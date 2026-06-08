"""
services/twilio_groq_voice.py
──────────────────────────────
2-Way AI Voice — Groq (llama-3.3-70b) + Twilio
"""
import html
import logging
import os
import time
from groq import Groq
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  CALL STAGES  — tracked so closing signals mean different things
#  at different points in the conversation
# ─────────────────────────────────────────────────────────────────
STAGE_INTRO        = "intro"         # Just confirmed who they are
STAGE_AVAILABILITY = "availability"  # Asked if free — waiting for answer
STAGE_CONVERSATION = "conversation"  # Main discussion happening
STAGE_SOLUTION     = "solution"      # Advice / meeting suggested
STAGE_CLOSING      = "closing"       # Wrapping up


# ─────────────────────────────────────────────────────────────────
#  CLOSING SIGNAL DETECTOR
#  Only triggers AFTER the conversation has actually happened
#  (stage = solution or closing). Early "okay" and "yes" should
#  never end the call — they are just acknowledgements.
# ─────────────────────────────────────────────────────────────────
DEFINITE_CLOSING = [
    "thank you", "thanks", "thank you so much", "thanks so much",
    "okay thank you", "ok thank you", "okay thanks", "ok thanks",
    "theek hai", "theek hain", "thik hai",
    "alright then", "all right then",
    "got it", "understood", "i understand",
    "bye", "goodbye", "good bye", "bye bye",
    "dhanyawad", "dhanyavaad", "shukriya",
    "bas ji", "bas theek hai",
    "haan ji okay", "haan ji theek",
    "no questions", "no more questions", "no doubts",
    "koi sawaal nahi", "koi doubt nahi",
    "samajh gaya", "samajh gayi", "samajh liya",
    "that's all", "that is all", "no more",
    "i'll come", "we'll come", "i will come", "we will come",
    "see you", "see you then", "see you tomorrow",
    "day after tomorrow works", "that works", "that's fine",
    "okay i'll come", "okay we'll come",
    "noted", "will do",
]

# These only mean "end call" AFTER a solution/meeting has been proposed
# In early stages they just mean "I'm listening, go on"
LATE_STAGE_CLOSING = [
    "okay", "ok", "alright", "sure", "of course",
    "yes", "yeah", "yep", "yup",
    "haan", "ha", "accha", "acha", "achha",
    "fine", "all good",
]

def _parent_wants_to_end(speech: str, stage: str) -> bool:
    text = speech.lower().strip()

    # Definite closing phrases work at any stage
    if any(sig in text for sig in DEFINITE_CLOSING):
        return True

    # Short affirmatives only close the call AFTER solution has been given
    if stage in (STAGE_SOLUTION, STAGE_CLOSING):
        if text in LATE_STAGE_CLOSING:
            return True

    return False


# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────
def _build_counselor_prompt(school, phone, student_name, parent_name,
                             dimension, risk_level, details, recommended_action):

    dim_key = dimension.lower().replace("behaviour", "behavior").strip()

    dimension_context = {
        "attendance": f"""
CONCERN: Attendance
{student_name}'s attendance is low and they risk not being allowed to sit exams.
Details: {details}

Natural flow:
- After they confirm they're free, briefly explain the attendance concern.
- Ask warmly: do they know why their child has been missing classes? Listen fully.
- If reason is valid — be understanding, give practical advice.
- If reason is unclear — suggest a meeting. See MEETING RULES.
- Answer any questions, then close warmly.
""",
        "performance": f"""
CONCERN: Academic Performance
{student_name} is struggling with grades. Details: {details}

Natural flow:
- After they confirm they're free, gently explain the academic concern.
- Understand the home situation naturally: study habits, distractions, extra coaching.
  Only ask what hasn't been mentioned. Don't interrogate.
- Suggest meeting if needed. See MEETING RULES.
- Close with encouragement.
""",
        "behavior": f"""
CONCERN: Behaviour
{student_name} has had behavioural incidents. Details: {details}

Natural flow:
- After they confirm they're free, raise the concern gently, framing it as caring.
- Ask how the child has been at home. Listen genuinely.
- If parent mentions stress, sadness, depression — acknowledge with empathy FIRST.
- Suggest meeting. See MEETING RULES.
- Close warmly.
""",
    }.get(dim_key, f"""
CONCERN: {dimension.upper()}
Details: {details}
Warm, natural conversation. Understand, offer help, suggest meeting if needed.
""")

    meeting_rules = {
        "HIGH": """
MEETING RULES (HIGH risk):
- Suggest tomorrow at 10 AM.
- If parent says not available: persuade ONCE warmly — like a caring teacher.
  Example: "I understand completely. I just want to mention that the situation
  is quite urgent and the sooner we can meet, the better for the child.
  Is there any possibility this week at all?"
- After that ONE gentle push, IMMEDIATELY accept whatever time they give.
- Confirm their date warmly and move to closing.
""",
        "MEDIUM": """
MEETING RULES (MEDIUM risk):
- Suggest meeting this week if helpful.
- If parent proposes any other time → accept immediately.
""",
        "LOW": """
MEETING RULES (LOW risk):
- Meeting is optional. Offer only if it feels right.
- Accept or skip with no pressure.
""",
    }.get(risk_level.upper(), "Suggest meeting if helpful. Accept whatever time parent proposes.")

    return f"""You are Priya — a warm, experienced school counselor calling from {school}.
You are speaking with {parent_name}, parent of {student_name}.
School phone: {phone}

YOUR PERSONALITY:
- You sound like a real human. Warm, calm, genuinely caring.
- Natural English with occasional Hindi words (ji, haan, accha, bilkul).
- You LISTEN first. You acknowledge before moving forward.
- Never robotic. Never scripted. Never repeat yourself.
- 2 to 3 sentences per reply. Phone calls need space.
- Use contractions: I'll, we'll, that's, it's, don't.
- Vary your sentence starters. Don't always begin with the parent's name.

STRICT CALL FLOW — FOLLOW THIS ORDER:
1. Parent confirms who they are → acknowledge warmly (e.g. "So glad I reached you ji.")
2. Ask if it is a good time to talk → wait for answer
   - If YES / free → move to step 3
   - If NO / busy → apologise, offer to call back, say goodbye [END_CALL]
3. Briefly introduce yourself and explain the concern — 1 to 2 sentences
4. Ask ONE open question, listen genuinely
5. Continue conversation based on what they actually said
6. Suggest next steps or meeting
7. Answer any questions
8. Close warmly

THIS IS CRITICAL — DO NOT SKIP STEP 2.
After the parent confirms their name, you MUST ask if it is a good time.
Do not jump straight to the concern. The parent needs the chance to say
they are busy before you start discussing sensitive matters.

{dimension_context}

{meeting_rules}

INTERRUPTION HANDLING:
- If parent speaks mid-reply — respond only to what they said.
- Drop what you were saying. React to their words directly.
- Never repeat a sentence they interrupted.

CLOSING:
- Only close the call when the conversation has naturally reached its end —
  after you have explained the concern, had a discussion, and suggested next steps.
- Do NOT close after just 1 or 2 turns. The conversation should feel complete.
- When parent gives a clear goodbye signal AFTER the main conversation —
  give ONE warm closing sentence then [END_CALL].

SPEECH STYLE:
- Spoken, natural sentences. Not formal written English.
- Contractions always. Lists never.
- Occasional Hindi: bilkul, ji, haan, accha.

STRICT RULES:
1. NEVER greet again after the first message.
2. NEVER re-introduce yourself mid-call.
3. ONLY discuss {dimension.upper()}.
4. 2 to 3 sentences per reply maximum.
5. End EVERY reply with one control tag on its own line:
   [CONTINUE]  — keep going
   [END_CALL]  — end now (only after a complete conversation)
6. Tag is for system only — never spoken aloud.

You are in a REAL phone call. React naturally. Be human.
""".strip()


# ─────────────────────────────────────────────────────────────────
#  CONVERSATION STATE
# ─────────────────────────────────────────────────────────────────
class ConversationState:
    def __init__(self, payload: CallPayload, school: str, phone: str):
        self.payload        = payload
        self.school         = school
        self.phone          = phone
        self.ended          = False
        self.turn_count     = 0
        self.stage          = STAGE_INTRO    # starts at intro
        self.messages: list[dict] = []
        self.system_prompt  = _build_counselor_prompt(
            school=school,
            phone=phone,
            student_name=payload.student_name,
            parent_name=payload.parent_name,
            dimension=payload.dimension,
            risk_level=payload.risk_level,
            details=payload.details,
            recommended_action=getattr(payload, "recommended_action", ""),
        )

    def advance_stage(self):
        """Move stage forward based on turn count as a rough heuristic."""
        if self.stage == STAGE_INTRO and self.turn_count >= 1:
            self.stage = STAGE_AVAILABILITY
        elif self.stage == STAGE_AVAILABILITY and self.turn_count >= 2:
            self.stage = STAGE_CONVERSATION
        elif self.stage == STAGE_CONVERSATION and self.turn_count >= 5:
            self.stage = STAGE_SOLUTION
        elif self.stage == STAGE_SOLUTION and self.turn_count >= 7:
            self.stage = STAGE_CLOSING


# ─────────────────────────────────────────────────────────────────
#  MAIN SERVICE
# ─────────────────────────────────────────────────────────────────
class TwoWayAIVoiceService:
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self._twilio_sid   = os.getenv("TWILIO_ACCOUNT_SID")
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self._from_number  = os.getenv("TWILIO_FROM_NUMBER")
        self._groq_key     = os.getenv("GROQ_API_KEY")
        self._ngrok_url    = os.getenv("NGROK_URL", "")
        self._school       = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone        = os.getenv("SCHOOL_PHONE", "033-4805-1910")

        self._twilio_ready = all([self._twilio_sid, self._twilio_token, self._from_number])
        self._ai_ready     = bool(self._groq_key)

        if self._twilio_ready:
            from twilio.rest import Client
            self._client = Client(self._twilio_sid, self._twilio_token)
            print("✅ Twilio Connected")

        if self._ai_ready:
            self._groq = Groq(api_key=self._groq_key)
            print(f"✅ Groq Connected  [{self.MODEL}]")
        else:
            print("⚠️  GROQ_API_KEY not set — Demo mode active")

        self._conversations: dict[str, ConversationState] = {}
        self.calls_made = []

    # ── Ask Groq ──────────────────────────────────────────────────
    def _ask_groq(self, state: ConversationState, parent_speech: str) -> str:
        state.messages.append({"role": "user", "content": parent_speech})
        state.turn_count += 1
        state.advance_stage()

        try:
            response = self._groq.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": state.system_prompt}
                ] + state.messages,
                temperature=0.75,
                max_tokens=120,
            )
            ai_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            ai_text = (
                f"I'm so sorry, there seems to be a technical issue. "
                f"Please call us at {self._phone}. Goodbye.\n[END_CALL]"
            )

        state.messages.append({"role": "assistant", "content": ai_text})
        return ai_text

    # ── Parse control tag ─────────────────────────────────────────
    def _parse_reply(self, ai_text: str) -> tuple[str, bool]:
        end_call = "[END_CALL]" in ai_text
        spoken = (ai_text
                  .replace("[END_CALL]", "")
                  .replace("[CONTINUE]", "")
                  .strip())
        return spoken, end_call

    # ── Build TwiML — Say inside Gather for interruptibility ──────
    def _twiml(self, spoken: str, end_call: bool) -> str:
        safe    = html.escape(spoken)
        safe_ph = html.escape(self._phone)
        webhook = f"{self._ngrok_url}/handle-parent-response"

        if end_call:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">{safe}</Say>
    <Pause length="1"/>
</Response>"""
        else:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{webhook}"
            method="POST"
            timeout="6"
            speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">{safe}</Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">I didn't catch that. Please call us at {safe_ph}. Goodbye.</Say>
</Response>"""

    # ── Opening TwiML ─────────────────────────────────────────────
    def _opening_twiml(self, payload: CallPayload) -> str:
        safe_parent  = html.escape(payload.parent_name)
        safe_student = html.escape(payload.student_name)
        safe_school  = html.escape(self._school)
        safe_phone   = html.escape(self._phone)
        webhook      = f"{self._ngrok_url}/handle-parent-response"

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{webhook}"
            method="POST"
            timeout="8"
            speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">Hello! Am I speaking with {safe_parent} ji? This is Priya calling from {safe_school}, regarding your child {safe_student}.</Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">I didn't hear a response. Please call us at {safe_phone}. Goodbye.</Say>
</Response>"""

    # ── Webhook handler ───────────────────────────────────────────
    def generate_followup_twiml(self, registration: str, parent_speech: str) -> str:
        state   = self._conversations.get(registration)
        safe_ph = html.escape(self._phone)

        if not state:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Thank you for your time. Please contact us at {safe_ph}. Goodbye.</Say>
</Response>"""

        if state.ended:
            return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Goodbye.</Say>
</Response>"""

        if not self._ai_ready:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Thank you ji. Please contact the college at {safe_ph}. Goodbye.</Say>
</Response>"""

        # Stage-aware closing detection
        if _parent_wants_to_end(parent_speech, state.stage):
            # Only hint to close if we've had a real conversation (turn 4+)
            if state.turn_count >= 4:
                speech_for_ai = (
                    parent_speech +
                    " [Note: parent is wrapping up. Give ONE warm closing sentence and use [END_CALL].]"
                )
            else:
                # Too early — treat as normal reply, don't close
                speech_for_ai = parent_speech
        else:
            speech_for_ai = parent_speech

        ai_text = self._ask_groq(state, speech_for_ai)
        spoken, end_call = self._parse_reply(ai_text)

        if end_call:
            state.ended = True

        print(f"  🎤 Parent : \"{parent_speech}\"")
        print(f"  🤖 Priya  : \"{spoken[:90]}...\"")
        print(f"  📊 Turn: {state.turn_count} | Stage: {state.stage} | End: {end_call}")

        return self._twiml(spoken, end_call)

    # ── make_call ─────────────────────────────────────────────────
    def make_call(self, payload: CallPayload) -> NotificationResult:
        if not self._twilio_ready:
            return NotificationResult(
                success=False,
                error_message="Twilio not configured.",
                student_id=payload.registration
            )

        state = ConversationState(payload, self._school, self._phone)
        self._conversations[payload.registration] = state
        print(f"✅ Ready: {payload.student_name} [{payload.dimension.upper()} / {payload.risk_level}]")

        twiml = self._opening_twiml(payload)

        try:
            print(f"\n📞 Calling {payload.parent_name} ({payload.to_number})...")
            call = self._client.calls.create(
                to=payload.to_number,
                from_=self._from_number,
                twiml=twiml
            )
            self.calls_made.append(payload)
            print(f"   ✅ SID: {call.sid}\n")
            return NotificationResult(
                success=True, sid=call.sid,
                student_id=payload.registration, channel="ai_2way_groq"
            )
        except Exception as e:
            print(f"   ❌ {str(e)[:120]}")
            return NotificationResult(
                success=False, error_message=str(e),
                student_id=payload.registration
            )

    # ── make_batch_calls ──────────────────────────────────────────
    def make_batch_calls(self, payloads: list) -> dict:
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        print(f"\n🤖 Groq 2-Way Calls: {len(payloads)}\n")
        for i, p in enumerate(payloads, 1):
            print(f"[{i}/{len(payloads)}] {p.student_name} [{p.dimension.upper()}]")
            r = self.make_call(p)
            results["details"].append({
                "registration": p.registration,
                "student": p.student_name,
                "parent":  p.parent_name,
                "success": r.success,
                "sid":     getattr(r, "sid", None),
                "channel": "ai_2way_groq",
            })
            if r.success: results["successful"] += 1
            else:         results["failed"] += 1
            time.sleep(2)
        print(f"\n✅ {results['successful']}/{results['total']}\n")
        return results


# ─────────────────────────────────────────────────────────────────
#  DEMO SERVICE
# ─────────────────────────────────────────────────────────────────
class TwoWayDemoService:
    def __init__(self):
        self._school = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone  = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self.calls_made     = []
        self._conversations = {}

    def make_call(self, payload: CallPayload) -> NotificationResult:
        self.calls_made.append(payload)
        print(f"\n{'═'*56}")
        print(f"  🖥️  DEMO MODE — {payload.dimension.upper()}")
        print(f"  Student: {payload.student_name}   Parent: {payload.parent_name}")
        print(f"  (Set GROQ_API_KEY for real AI calls)")
        print(f"{'═'*56}\n")
        return NotificationResult(
            success=True, sid=f"DEMO_{len(self.calls_made):03d}",
            student_id=payload.registration, channel="demo"
        )

    def generate_followup_twiml(self, registration: str, parent_speech: str) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Thank you for your response. Goodbye.</Say>
</Response>"""

    def make_batch_calls(self, payloads: list) -> dict:
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        for p in payloads:
            r = self.make_call(p)
            results["successful"] += 1
            results["details"].append({
                "student": p.student_name, "parent": p.parent_name,
                "success": True, "sid": r.sid, "channel": "demo",
            })
        return results