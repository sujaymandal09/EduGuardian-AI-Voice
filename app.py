"""
app.py - EduGuardian with Groq 2-Way AI Voice
"""
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from core.models import StudentRecord, CallPayload
from agents.attendance_agent import AttendanceAgent
from agents.performance_agent import PerformanceAgent
from agents.behavior_agent import BehaviorAgent
from services.call_history import get_call_history_repository

load_dotenv()

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "attendance-guardian-secret")

voice_service = None
last_results = {}
call_sid_map = {}       # maps Twilio CallSid to its voice conversation key
silence_count = {}      # maps Twilio CallSid → number of consecutive no-speech events
summary_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="call-summary")

os.makedirs('data', exist_ok=True)
CSV_PATH = os.path.join('data', 'students.csv')


def get_voice_service():
    """Pick the right service based on available API keys."""
    global voice_service
    if voice_service is None:
        twilio_ready = bool(os.getenv("TWILIO_ACCOUNT_SID"))
        groq_ready   = bool(os.getenv("GROQ_API_KEY"))

        if twilio_ready and groq_ready:
            from services.twilio_groq_voice import TwoWayAIVoiceService
            voice_service = TwoWayAIVoiceService()
            print("🤖 2-Way AI Voice (Twilio + Groq)")
        elif twilio_ready:
            from services.twilio_voice_service import TwilioVoiceService
            voice_service = TwilioVoiceService()
            print("📞 Twilio Voice (1-Way)")
        else:
            from services.twilio_groq_voice import TwoWayDemoService
            voice_service = TwoWayDemoService()
            print("🖥️  Demo Mode (no API keys)")
    return voice_service


def _json_safe(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


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
    global call_sid_map
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
        language="en",
        teacher_id=os.getenv("DEFAULT_TEACHER_ID", "default")
    ) for s in selected_students]

    voice   = get_voice_service()
    results = voice.make_batch_calls(payloads)

    # Store CallSid → registration so webhook finds the right conversation
    for detail in results.get("details", []):
        sid = detail.get("sid")
        if sid:
            call_sid_map[sid] = sid

    return render_template('call_status.html', results=results, dimension=dimension)


@app.route('/handle-parent-response', methods=['GET', 'POST'])
def handle_parent_response():
    """Twilio webhook — receives parent speech, returns next AI reply as TwiML."""
    parent_speech = request.form.get('SpeechResult', '').strip()
    call_sid      = request.form.get('CallSid', '')

    print(f"\n📞 CallSid : {call_sid}")
    print(f"   Parent  : \"{parent_speech}\"")

    voice  = get_voice_service()
    ngrok  = os.getenv('NGROK_URL', '')
    phone  = os.getenv('SCHOOL_PHONE', '033-4805-1910')

    # ── No speech detected ────────────────────────────────────────
    if not parent_speech:
        count = silence_count.get(call_sid, 0) + 1
        silence_count[call_sid] = count

        if count == 1:
            # First silence — one short clarification, then re-open Gather once more
            print(f"   [Silence #{count}] — asking once")
            if hasattr(voice, 'record_system_turn'):
                voice.record_system_turn(call_sid, "[No speech detected]")
            if hasattr(voice, 'record_call_turn'):
                voice.record_call_turn(
                    call_sid, "agent",
                    "I'm sorry, I couldn't catch that. Please go ahead whenever you're ready.",
                )
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{ngrok}/handle-parent-response"
            method="POST"
            timeout="5"
            speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">I'm sorry, I couldn't catch that. Please go ahead whenever you're ready.</Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">I wasn't able to hear a response. I will note this and follow up if needed. Thank you for your time. Have a good day.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

        else:
            # Second (or more) silence — close gracefully, no more retries
            print(f"   [Silence #{count}] — closing gracefully")
            if hasattr(voice, 'record_system_turn'):
                voice.record_system_turn(call_sid, "[Call closed after repeated silence]")
            if hasattr(voice, 'record_call_turn'):
                voice.record_call_turn(
                    call_sid, "agent",
                    "I couldn't clearly hear the response, so I will conclude here for now.",
                )
            silence_count.pop(call_sid, None)   # clean up
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">I couldn't clearly hear the response, so I will conclude here for now. Please feel free to contact us at {phone}. Thank you and have a good day.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

    # ── Speech received — reset silence counter ───────────────────
    silence_count.pop(call_sid, None)

    # A live Groq call is stored under CallSid as soon as Twilio creates it.
    # Use that key directly so a webhook does not wait for the batch route to finish.
    registration = None
    if call_sid and hasattr(voice, '_conversations') and call_sid in voice._conversations:
        registration = call_sid
    else:
        registration = call_sid_map.get(call_sid)
    # Fallback: if only one call active, use that
    if not registration and hasattr(voice, '_conversations'):
        active = list(voice._conversations.keys())
        if len(active) == 1:
            registration = active[0]

    # Hand off to AI
    if registration and hasattr(voice, 'generate_followup_twiml'):
        started = time.monotonic()
        try:
            twiml = voice.generate_followup_twiml(registration, parent_speech)
            elapsed = time.monotonic() - started
            if elapsed > 8:
                logger.warning("Slow voice webhook for %s: %.2fs", call_sid, elapsed)
            return twiml, 200, {'Content-Type': 'text/xml'}
        except Exception:
            logger.exception("Voice webhook failed for CallSid %s", call_sid)
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">I'm sorry, I couldn't complete that request right now. Please contact the school at {phone}. Thank you.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

    # Hard fallback (no registration found)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Thank you. Please contact the college at {phone}. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}


@app.post('/twilio/call-status')
def twilio_call_status():
    """Persist Twilio lifecycle updates and queue one post-call summary."""
    call_sid = request.form.get('CallSid', '').strip()
    status = request.form.get('CallStatus', '').strip().lower()
    duration_text = request.form.get('CallDuration', '').strip()
    if not call_sid or not status:
        return "", 400
    try:
        duration = int(duration_text) if duration_text else None
    except ValueError:
        duration = None
    voice = get_voice_service()
    try:
        if hasattr(voice, 'update_call_status'):
            voice.update_call_status(call_sid, status, duration)
        if status == 'completed' and hasattr(voice, 'finalize_call_summary'):
            summary_executor.submit(voice.finalize_call_summary, call_sid)
    except Exception:
        logger.exception("Could not process Twilio status for %s", call_sid)
    return "", 204


@app.get('/calls')
def call_history():
    repository = get_call_history_repository()
    calls = repository.list_calls(limit=100) if repository.enabled else []
    return render_template('call_history.html', calls=calls, history_enabled=repository.enabled)


@app.get('/calls/<call_sid>')
def call_detail(call_sid):
    repository = get_call_history_repository()
    call = repository.get_call(call_sid) if repository.enabled else None
    if not call:
        return "Call not found", 404
    return render_template(
        'call_detail.html', call=call, turns=repository.get_turns(call_sid)
    )


@app.get('/api/calls/<call_sid>')
def call_detail_api(call_sid):
    repository = get_call_history_repository()
    call = repository.get_call(call_sid) if repository.enabled else None
    if not call:
        return jsonify({"error": "Call not found"}), 404
    return jsonify(_json_safe(call))


@app.post('/calls/<call_sid>/retry-summary')
def retry_call_summary(call_sid):
    repository = get_call_history_repository()
    if not repository.enabled or not repository.get_call(call_sid):
        return "Call not found", 404
    voice = get_voice_service()
    if hasattr(voice, 'finalize_call_summary'):
        summary_executor.submit(voice.finalize_call_summary, call_sid)
    return redirect(url_for('call_detail', call_sid=call_sid))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            file.save(CSV_PATH)
            global voice_service, last_results, call_sid_map, silence_count
            voice_service = None
            last_results  = {}
            call_sid_map  = {}
            silence_count = {}
            flash("✅ Uploaded!", "success")
            return redirect(url_for('index'))
        flash("Invalid file!", "error")
    return render_template('upload.html')


@app.route('/reset')
def reset():
    global voice_service, last_results, call_sid_map, silence_count
    voice_service = None
    last_results  = {}
    call_sid_map  = {}
    silence_count = {}
    flash("✅ Reset!", "success")
    return redirect(url_for('index'))

@app.get("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("\n" + "="*50)
    print("  EDUGUARDIAN -- GROQ AI VOICE")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
