"""
app.py - EduGuardian with Groq 2-Way AI Voice
"""
import csv
import os
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from core.models import StudentRecord, CallPayload
from agents.attendance_agent import AttendanceAgent
from agents.performance_agent import PerformanceAgent
from agents.behavior_agent import BehaviorAgent

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "attendance-guardian-secret")

voice_service = None
last_results  = {}
silence_count = {}

os.makedirs('data', exist_ok=True)
CSV_PATH = os.path.join('data', 'students.csv')

from services.database import init_db, get_all_summaries, delete_summary
init_db()

from services.schedule_manager import ScheduleManager, sync_from_weekly_csv
sync_from_weekly_csv()
ScheduleManager().reset_week_if_needed()


def get_voice_service():
    global voice_service
    if voice_service is None:
        twilio_ready = bool(os.getenv("TWILIO_ACCOUNT_SID"))
        groq_ready   = bool(os.getenv("GROQ_API_KEY"))

        if twilio_ready and groq_ready:
            from services.twilio_groq_voice import TwoWayAIVoiceService
            voice_service = TwoWayAIVoiceService()
            print("[OK] 2-Way AI Voice (Twilio + Groq)")
        elif twilio_ready:
            from services.twilio_voice_service import TwilioVoiceService
            voice_service = TwilioVoiceService()
            print("[OK] Twilio Voice (1-Way)")
        else:
            from services.twilio_groq_voice import TwoWayDemoService
            voice_service = TwoWayDemoService()
            print("[OK] Demo Mode (no API keys)")
    return voice_service


def load_students():
    students = []
    if not os.path.exists(CSV_PATH):
        return students
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                students.append(StudentRecord(
                    registration=row.get("registration", "").strip(),
                    name=row.get("name", "").strip(),
                    parent_name=row.get("parent_name", "").strip(),
                    parent_phone=row.get("parent_phone", "").strip(),
                    attendance_pct=float(row.get("attendance_pct", 0)),
                    attendance_total=int(row.get("attendance_total", 0)),
                    attendance_attended=int(row.get("attendance_attended", 0)),
                    consecutive_absences=int(row.get("consecutive_absences", 0)),
                    performance_grade=row.get("performance_grade", "").strip(),
                    performance_remarks=row.get("performance_remarks", "").strip(),
                    behavior_incidents=int(row.get("behavior_incidents", 0)),
                    behavior_status=row.get("behavior_status", "").strip(),
                    language=row.get("language", "en").strip(),
                ))
        return students
    except Exception as e:
        print(f"Error loading students: {e}")
        return []


@app.route('/')
def index():
    students = load_students()
    at_risk_count = 0
    if students:
        agents = [AttendanceAgent(), PerformanceAgent(), BehaviorAgent()]
        at_risk_set = set()
        for agent in agents:
            for r in agent.find_at_risk(students):
                at_risk_set.add(r.registration)
        at_risk_count = len(at_risk_set)
    voice = get_voice_service()
    calls = len(voice.calls_made) if hasattr(voice, 'calls_made') else 0
    return render_template('index.html',
                           total_students=len(students),
                           at_risk_count=at_risk_count,
                           calls_made=calls)


@app.route('/analyze/<dimension>')
def analyze(dimension):
    students = load_students()
    if not students:
        flash("No data! Upload CSV.", "error")
        return redirect(url_for('index'))

    agents = {
        "attendance": (AttendanceAgent(), "📊 Attendance Risk"),
        "performance": (PerformanceAgent(), "📝 Academic Performance"),
        "behavior":   (BehaviorAgent(),    "⚠️ Behavior Issues"),
    }

    if dimension == "all":
        all_risk = []
        for d, (ag, t) in agents.items():
            all_risk.extend(ag.find_at_risk(students))
        last_results['current'] = all_risk
        last_results['dimension'] = 'all'
        risks = [r.risk_level for r in all_risk]
        high = "HIGH" if "HIGH" in risks else ("MEDIUM" if "MEDIUM" in risks else "LOW")
        return render_template('results.html', at_risk=all_risk,
                               dimension_title="Complete Analysis", dimension="all", highest_risk=high)
    else:
        agent, title = agents.get(dimension, (None, None))
        if not agent:
            return redirect(url_for('index'))
        at_risk = agent.find_at_risk(students)
        last_results['current'] = at_risk
        last_results['dimension'] = dimension
        risks = [r.risk_level for r in at_risk]
        high = "HIGH" if "HIGH" in risks else ("MEDIUM" if "MEDIUM" in risks else "LOW")
        return render_template('results.html', at_risk=at_risk,
                               dimension_title=title, dimension=dimension, highest_risk=high)


@app.route('/call/selective', methods=['POST'])
def call_selective():
    selected  = request.form.getlist('selected_students')
    dimension = request.form.get('dimension', 'unknown')

    if not selected:
        flash("Select students!", "warning")
        return redirect(url_for('analyze', dimension=dimension))

    at_risk = last_results.get('current', [])
    if not at_risk:
        flash("Run analysis first!", "warning")
        return redirect(url_for('index'))

    selected_students = [at_risk[int(i)] for i in selected if int(i) < len(at_risk)]

    payloads = [CallPayload(
        to_number=s.parent_phone,
        registration=s.registration,
        student_name=s.student_name,
        parent_name=s.parent_name,
        dimension=s.dimension,
        risk_level=s.risk_level,
        details=s.details,
        recommended_action=s.recommended_action,
        language="en"
    ) for s in selected_students]

    voice   = get_voice_service()
    results = voice.make_batch_calls(payloads)

    return render_template('call_status.html', results=results, dimension=dimension)


def _resolve_registration(voice, call_sid: str, to_number: str) -> str | None:
    """
    Resolve registration number from a Twilio CallSid.
    
    Three lookups in order of reliability:
    1. voice._call_sid_map  — set inside make_call() after call.create() returns.
       May miss the very first webhook due to race condition (Twilio fires before
       make_call() sets the map). Subsequent turns always hit this.
    2. voice._phone_to_reg  — set BEFORE call.create() using payload.to_number.
       Catches the race-condition webhook reliably.
    3. Single active conversation fallback — works when only 1 call is active.
    """
    # Primary: SID map (set after call.create)
    if hasattr(voice, '_call_sid_map'):
        reg = voice._call_sid_map.get(call_sid)
        if reg:
            # Opportunistically sync SID → reg from phone map if not already set
            return reg

    # Secondary: phone number map (set BEFORE call.create — no race condition)
    if hasattr(voice, '_phone_to_reg') and to_number:
        reg = voice._phone_to_reg.get(to_number)
        if reg:
            print(f"   [INFO] Resolved via phone map: {to_number} → {reg}")
            # Backfill the SID map so future turns use the fast path
            if hasattr(voice, '_call_sid_map'):
                voice._call_sid_map[call_sid] = reg
            return reg

    # Fallback: only one conversation active
    if hasattr(voice, '_conversations'):
        active = [k for k, v in voice._conversations.items() if not getattr(v, 'ended', False)]
        if len(active) == 1:
            print(f"   [WARN] Using sole active conversation: {active[0]}")
            if hasattr(voice, '_call_sid_map'):
                voice._call_sid_map[call_sid] = active[0]
            return active[0]

    return None


@app.route('/handle-parent-response', methods=['GET', 'POST'])
def handle_parent_response():
    # Wrap the ENTIRE handler so Flask never returns a 500 to Twilio
    try:
        return _handle_parent_response_inner()
    except Exception:
        traceback.print_exc()
        phone = os.getenv('SCHOOL_PHONE', '033-4805-1910')
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">I'm sorry, there was a technical issue. Please call us at {phone}. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}


def _handle_parent_response_inner():
    parent_speech = request.form.get('SpeechResult', '').strip()
    call_sid      = request.form.get('CallSid', '')
    # Twilio sends the dialled number as "To" or "Called"
    to_number     = request.form.get('To', '') or request.form.get('Called', '')

    print(f"\n[CALL] CallSid : {call_sid}")
    print(f"   Parent  : \"{parent_speech}\"")

    voice  = get_voice_service()
    ngrok  = os.getenv('NGROK_URL', '')
    phone  = os.getenv('SCHOOL_PHONE', '033-4805-1910')

    # ── Silence / no speech ───────────────────────────────────────
    if not parent_speech:
        count = silence_count.get(call_sid, 0) + 1
        silence_count[call_sid] = count

        if count == 1:
            print(f"   [Silence #{count}] — re-prompting")
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{ngrok}/handle-parent-response"
            method="POST"
            timeout="8"
            speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">I'm sorry, I couldn't catch that. Please go ahead whenever you're ready.</Say>
    </Gather>
    <Redirect method="POST">{ngrok}/handle-parent-response</Redirect>
</Response>""", 200, {'Content-Type': 'text/xml'}

        elif count == 2:
            print(f"   [Silence #{count}] — second reprompt")
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{ngrok}/handle-parent-response"
            method="POST"
            timeout="10"
            speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">I'm still here. Take your time, please go ahead.</Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">I wasn't able to hear a response. I will note this and follow up. Thank you and have a good day.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

        else:
            print(f"   [Silence #{count}] — closing")
            silence_count.pop(call_sid, None)
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">I couldn't hear a response, so I'll conclude here. Please call us at {phone}. Thank you and have a good day.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

    silence_count.pop(call_sid, None)

    # ── Resolve registration ──────────────────────────────────────
    registration = _resolve_registration(voice, call_sid, to_number)

    if not registration:
        print(f"   [ERROR] Could not resolve registration for SID={call_sid} To={to_number}")
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">I'm sorry, there was a technical issue. Please contact the college at {phone}. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

    # ── Delegate to voice service ─────────────────────────────────
    if hasattr(voice, 'generate_followup_twiml'):
        try:
            twiml = voice.generate_followup_twiml(registration, parent_speech)
            return twiml, 200, {'Content-Type': 'text/xml'}
        except Exception:
            traceback.print_exc()
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">I'm sorry, there's a technical issue on our end. Please call us at {phone}. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Thank you. Please contact the college at {phone}. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}


# ── Summaries ──────────────────────────────────────────────────

@app.route('/summaries')
def summaries():
    all_summaries = get_all_summaries()
    total       = len(all_summaries)
    meetings    = sum(1 for s in all_summaries if s.get('meeting_booked'))
    high_risk   = sum(1 for s in all_summaries if s.get('risk_level') == 'HIGH')
    medium_risk = sum(1 for s in all_summaries if s.get('risk_level') == 'MEDIUM')
    return render_template('dashboard.html',
                           summaries=all_summaries,
                           total=total,
                           meetings=meetings,
                           high_risk=high_risk,
                           medium_risk=medium_risk)


@app.route('/summaries/delete/<call_id>', methods=['POST'])
def delete_summary_route(call_id):
    success = delete_summary(call_id)
    return jsonify({"success": success})


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            file.save(CSV_PATH)
            global voice_service, last_results, silence_count
            voice_service = None
            last_results  = {}
            silence_count = {}
            flash("✅ Uploaded!", "success")
            return redirect(url_for('index'))
        flash("Invalid file!", "error")
    return render_template('upload.html')


@app.route('/reset')
def reset():
    global voice_service, last_results, silence_count
    voice_service = None
    last_results  = {}
    silence_count = {}
    flash("✅ Reset!", "success")
    return redirect(url_for('index'))


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("\n" + "="*50)
    print("  EDUGUARDIAN -- GROQ AI VOICE")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)