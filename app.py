"""
app.py - Attendance Guardian with 2-Way AI Voice
"""
import sys
import csv
import os

# Windows cp1252 can't print emoji — force UTF-8 for the whole process
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from core.models import StudentRecord, CallPayload
from agents.attendance_agent import AttendanceAgent
from agents.performance_agent import PerformanceAgent
from agents.behavior_agent import BehaviorAgent

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "attendance-guardian-secret")

voice_service = None
last_results = {}

os.makedirs('data', exist_ok=True)
CSV_PATH = os.path.join('data', 'students.csv')


def get_voice_service():
    """Smart service selector."""
    global voice_service
    
    if voice_service is None:
        # 1. Twilio + Groq (2-Way AI - BEST)
        if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("GROQ_API_KEY"):
            from services.twilio_gemini_voice import TwoWayAIVoiceService
            voice_service = TwoWayAIVoiceService()
            print("🤖 2-Way AI Voice (Twilio + Gemini)")
        
        # 2. Twilio only (1-Way)
        elif os.getenv("TWILIO_ACCOUNT_SID"):
            from services.twilio_voice_service import TwilioVoiceService
            voice_service = TwilioVoiceService()
            print("📞 Twilio Voice (1-Way)")
        
        # 3. Demo mode
        else:
            from services.twilio_gemini_voice import TwoWayDemoService
            voice_service = TwoWayDemoService()
            print("🤖 AI Demo Mode")
    
    return voice_service


def load_students():
    """Load students from CSV."""
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
        print(f"Error: {e}")
        return []


@app.route('/')
def index():
    """Dashboard."""
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
    calls = len(voice.calls_made) if hasattr(voice, 'calls_made') else len(voice._conversations) if hasattr(voice, '_conversations') else 0
    
    return render_template('index.html',
                         total_students=len(students),
                         at_risk_count=at_risk_count,
                         calls_made=calls)


@app.route('/analyze/<dimension>')
def analyze(dimension):
    """Analyze students."""
    students = load_students()
    if not students:
        flash("No data! Upload CSV.", "error")
        return redirect(url_for('index'))
    
    agents = {
        "attendance": (AttendanceAgent(), "📊 Attendance Risk"),
        "performance": (PerformanceAgent(), "📝 Academic Performance"),
        "behavior": (BehaviorAgent(), "⚠️ Behavior Issues"),
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
    """Start AI conversations."""
    selected = request.form.getlist('selected_students')
    dimension = request.form.get('dimension', 'unknown')
    
    if not selected:
        flash("Select students!", "warning")
        return redirect(url_for('analyze', dimension=dimension))
    
    at_risk = last_results.get('current', [])
    if not at_risk:
        flash("Run analysis first!", "warning")
        return redirect(url_for('index'))
    
    selected_students = [at_risk[int(i)] for i in selected if int(i) < len(at_risk)]

    # Build a lookup so the AI gets real CSV data during the call
    all_students = {st.registration: st for st in load_students()}

    payloads = []
    for s in selected_students:
        st = all_students.get(s.registration)
        payloads.append(CallPayload(
            to_number=s.parent_phone,
            registration=s.registration,
            student_name=s.student_name,
            parent_name=s.parent_name,
            dimension=s.dimension,
            risk_level=s.risk_level,
            details=s.details,
            recommended_action=s.recommended_action,
            language="en",
            attendance_pct=st.attendance_pct if st else None,
            attendance_total=st.attendance_total if st else None,
            attendance_attended=st.attendance_attended if st else None,
            consecutive_absences=st.consecutive_absences if st else None,
            performance_grade=st.performance_grade if st else None,
            performance_remarks=st.performance_remarks if st else None,
            behavior_incidents=st.behavior_incidents if st else None,
            behavior_status=st.behavior_status if st else None,
        ))
    
    voice = get_voice_service()
    results = voice.make_batch_calls(payloads)
    
    return render_template('call_status.html', results=results, dimension=dimension)

@app.route('/handle-parent-response', methods=['GET', 'POST'])
def handle_parent_response():
    """Twilio webhook - receives parent's speech."""

    call_sid = request.form.get('CallSid', '')
    parent_speech = request.form.get('SpeechResult', '').strip()
    ngrok_url = os.getenv("NGROK_URL", "")
    phone = os.getenv("SCHOOL_PHONE", "033-4805-1910")

    if not parent_speech:
        print("⚠️  No speech detected from parent")
        retry_url = f"{ngrok_url}/handle-parent-response"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN" action="{retry_url}" method="POST" timeout="5" speechTimeout="auto" bargeIn="true">
        <Say voice="alice" language="en-IN">I didn't catch that. Please go ahead and speak.</Say>
    </Gather>
    <Say voice="alice" language="en-IN">Please contact the college at {phone}. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

    print(f"\n📞 PARENT SAID: \"{parent_speech}\"")

    voice = get_voice_service()

    # Resolve registration using CallSid sent by Twilio in every webhook POST
    registration = None
    if call_sid and hasattr(voice, 'get_registration_for_call'):
        registration = voice.get_registration_for_call(call_sid)

    # Fallback: pick the only active conversation (single-call scenario)
    if not registration and hasattr(voice, '_conversations') and voice._conversations:
        registration = next(iter(voice._conversations))
        print(f"⚠️  CallSid lookup missed — falling back to first conversation ({registration})")

    if registration and hasattr(voice, 'generate_followup_twiml'):
        twiml = voice.generate_followup_twiml(registration, parent_speech)
        return twiml, 200, {'Content-Type': 'text/xml'}

    # Fallback when conversation state is gone (e.g. app restarted mid-call)
    print(f"⚠️  No active conversation found for CallSid={call_sid}")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">Thank you for your response. Please contact the college at {phone} for further assistance. Goodbye.</Say>
</Response>""", 200, {'Content-Type': 'text/xml'}

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload CSV."""
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            file.save(CSV_PATH)
            global voice_service, last_results
            voice_service = None
            last_results = {}
            flash("✅ Uploaded!", "success")
            return redirect(url_for('index'))
        flash("Invalid file!", "error")
    return render_template('upload.html')


@app.route('/reset')
def reset():
    """Reset."""
    global voice_service, last_results
    voice_service = None
    last_results = {}
    flash("✅ Reset!", "success")
    return redirect(url_for('index'))


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("\n" + "="*55)
    print("  🤖 ATTENDANCE GUARDIAN + GEMINI AI")
    print("  2-Way AI Voice Conversations")
    print("="*55)
    if os.path.exists(CSV_PATH):
        s = load_students()
        print(f"  📂 {len(s)} students loaded")
    print(f"  👉 http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)