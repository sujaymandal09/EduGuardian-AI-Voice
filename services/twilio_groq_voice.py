"""
services/twilio_groq_voice.py
──────────────────────────────
2-Way AI Voice — Groq (llama-3.1-8b-instant) + Twilio
"""
import html
import logging
import os
import re
import time
from groq import Groq
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  CALL STAGES
# ─────────────────────────────────────────────────────────────────
STAGE_INTRO        = "intro"
STAGE_AVAILABILITY = "availability"
STAGE_CONVERSATION = "conversation"
STAGE_SOLUTION     = "solution"
STAGE_FAREWELL     = "farewell"
STAGE_CLOSING      = "closing"


# ─────────────────────────────────────────────────────────────────
#  CLOSING SIGNAL DETECTOR
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

LATE_STAGE_CLOSING = [
    "okay", "ok", "alright", "sure", "of course",
    "yes", "yeah", "yep", "yup",
    "haan", "ha", "accha", "acha", "achha",
    "fine", "all good",
]

def _parent_wants_to_end(speech: str, stage: str) -> bool:
    text = speech.lower().strip()
    if any(sig in text for sig in DEFINITE_CLOSING):
        return True
    if stage in (STAGE_SOLUTION, STAGE_FAREWELL, STAGE_CLOSING):
        if text in LATE_STAGE_CLOSING:
            return True
    return False


# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────
def _build_counselor_prompt(school, phone, student_name, parent_name,
                             dimension, risk_level, details, recommended_action):

    dim_key = dimension.lower().replace("behaviour", "behavior").strip()

    risk_resolution = {
        "HIGH": f"""
RESOLUTION (HIGH risk):
- A face-to-face meeting is essential. Propose it firmly but warmly.
- Suggest tomorrow at 10 AM as the first option.
- If unavailable, accept any time this week — but make clear it is important.
- Do NOT accept monitoring alone as the outcome for a high-risk case.
""",
        "MEDIUM": f"""
RESOLUTION (MEDIUM risk):
- Reassure the parent that you will monitor the situation closely.
- Say clearly: "We'll keep a very close eye on this over the next week or two,
  and if things don't improve we will need to meet in person."
- Offer a meeting only if the parent wants one — do not push.
- The outcome is: monitoring with escalation path, not a meeting by default.
""",
        "LOW": f"""
RESOLUTION (LOW risk):
- End on a genuinely positive note.
- Acknowledge what {student_name} is doing well, even while raising the concern.
- No meeting needed. Just encouragement and practical home advice.
- Example close: "I just wanted to flag it early so we can nip it in the bud together.
  {student_name} is doing well overall and I have every confidence they'll turn it around."
""",
    }.get(risk_level.upper(), f"""
RESOLUTION:
- Suggest meeting if helpful. Accept whatever time parent proposes.
""")

    dimension_context = {
        "attendance": f"""
CONCERN: Attendance
{student_name}'s attendance is low and they risk not being allowed to sit exams.
Details (USE ONLY THESE FACTS — do not add any other specifics): {details}

Natural flow:
- After they confirm they're free, briefly introduce the attendance concern in 1-2 sentences.
- CRITICAL: Your first message about the concern must ALWAYS end with an open question —
  never a plain statement. Invite the parent to share what's been going on.
  Example: "...I was hoping you might be able to shed some light on what's been happening?"
- Listen fully to their answer before suggesting any next steps.
- If reason is valid — be understanding, give practical advice.
- If reason is unclear — follow RESOLUTION below.
- Answer any questions, then wait for the system to handle closing.

{risk_resolution}
""",
        "performance": f"""
CONCERN: Academic Performance
{student_name} is struggling with grades.
Details (USE ONLY THESE FACTS — do not add any other specifics): {details}

Natural flow:
- After they confirm they're free, briefly introduce the academic concern in 1-2 sentences.
- CRITICAL: Your first message about the concern must ALWAYS end with an open question —
  never a plain statement. Invite the parent to share the home situation.
  Example: "...I was wondering if you've noticed anything at home that might be affecting him?"
- Only ask what hasn't been mentioned. Don't interrogate.
- Follow RESOLUTION below.
- Close with encouragement.

{risk_resolution}
""",
        "behavior": f"""
CONCERN: Behaviour
{student_name} has had behavioural incidents.
Details (USE ONLY THESE FACTS — do not add any other specifics): {details}

Natural flow:
- After they confirm they're free, gently introduce the behavioural concern in 1-2 sentences.
- CRITICAL: Your first message about the concern must ALWAYS end with an open question —
  never a plain statement. Invite the parent to share how the child has been at home.
  Example: "...I wanted to ask — how has he been at home lately?"
- Refer ONLY to the incident details given above. Do NOT invent names of clubs, teams,
  teachers, or events that were not mentioned in those details.
- If parent mentions stress, sadness, depression — acknowledge with empathy FIRST.
- Follow RESOLUTION below.
- Close warmly.

{risk_resolution}
""",
    }.get(dim_key, f"""
CONCERN: {dimension.upper()}
Details (USE ONLY THESE FACTS): {details}
- Introduce the concern briefly, then ALWAYS end with an open question.
- Never deliver the concern as a plain statement with nothing for the parent to respond to.

{risk_resolution}
""")

    meeting_rules = {
        "HIGH": """
MEETING RULES (HIGH risk):
- Suggest tomorrow at 10 AM.
- If parent says not available: persuade ONCE warmly — like a caring teacher.
  Example: "I completely understand. I just want to mention that the situation
  is quite urgent and the sooner we can meet, the better it will be for your child.
  Is there any possibility this week at all?"
- After that ONE gentle push, IMMEDIATELY accept whatever time they give.
- Confirm their date warmly and move to farewell.
""",
        "MEDIUM": """
MEETING RULES (MEDIUM risk):
- The default outcome is monitoring, not a meeting.
- Tell the parent clearly: "We'll monitor closely, and if this continues we will
  need to meet." Only offer a meeting if the parent requests one.
""",
        "LOW": """
MEETING RULES (LOW risk):
- No meeting needed. Close with encouragement.
- Offer the school contact number so parents can reach out if they want.
""",
    }.get(risk_level.upper(), "Suggest meeting if helpful. Accept whatever time parent proposes.")

    scheduling_section = """
MEETING SCHEDULING:
- When a parent agrees to a meeting, DO NOT suggest any specific time yourself.
- Instead, say: "Please wait for a moment. I need to check my schedule." then on a NEW LINE
  emit ONLY this tag (nothing else on that line):
  [SCHEDULE_MEETING: <day_preference>]
  Where <day_preference> is one of:
    - "tomorrow"      — parent said yes/tomorrow/soonest
    - "next week"     — parent said next week
    - "<day name>"    — parent named a specific day, e.g. "Thursday"
- The system will replace this tag with the real available slot and inject it into your reply.
- After the system injects the slot, confirm it warmly and move to farewell.
- NEVER guess or invent a time. NEVER hardcode a time. ALWAYS emit the tag first.
"""

    return f"""You are Priya — a warm, experienced school counselor calling from {school}.
You are speaking with {parent_name}, parent of {student_name}.
School phone: {phone}
Office hours: Monday to Friday, 9 AM to 4 PM

YOUR PERSONALITY:
- You sound like a real human. Warm, calm, genuinely caring.
- Speak in clear, natural, grammatically correct English only. No Hindi or mixed-language words whatsoever.
- Do NOT use words such as: ji, haan, accha, acha, bilkul, theek hai, nahi, or any other Hindi or regional language term — not even as filler or courtesy words.
- If the parent speaks in Hindi or mixed language, always reply in English only.
- You LISTEN first. You acknowledge before moving forward.
- Never robotic. Never scripted. Never repeat yourself.
- 2 to 3 sentences per reply. Phone calls need space.
- Use contractions: I'll, we'll, that's, it's, don't.
- Vary your sentence starters. Don't always begin with the parent's name.

STRICT CALL FLOW — FOLLOW THIS ORDER:
1. Parent confirms who they are → in ONE sentence: acknowledge warmly AND introduce
   yourself as Priya, school counselor at {school}.
   Example: "I'm so glad I reached you — I'm Priya, the school counselor at {school}."
2. In the SAME reply, ask if it is a good time to talk. Wait for answer.
   - If YES / free → move to step 3
   - If NO / busy → apologise, offer to call back, say goodbye [END_CALL]
3. Briefly introduce the concern — 1 to 2 sentences
4. Ask ONE open question, listen genuinely
5. Continue conversation based on what they actually said
6. Suggest next steps or meeting
7. Answer any questions
8. Close warmly

THIS IS CRITICAL — DO NOT SKIP THE AVAILABILITY CHECK.
Your first reply must introduce yourself AND ask if it is a good time — both in one breath.
Do not jump straight to the concern. The parent needs the chance to say
they are busy before you start discussing sensitive matters.

{dimension_context}

{meeting_rules}

{scheduling_section}

INTERRUPTION HANDLING:
- If parent speaks mid-reply — respond only to what they said.
- Drop what you were saying. React to their words directly.
- Never repeat a sentence they interrupted.

CLOSING:
- NEVER use [END_CALL] on your own — not even after confirming a meeting.
  The system controls when the call ends. Wait for the system tag.
- When the system adds [SYSTEM: ask about more questions], ask the parent
  if they have any other questions or concerns, in ONE warm sentence. Then use [CONTINUE].
- When the system adds [SYSTEM: end call now], give ONE warm goodbye sentence that
  includes the office hours (Monday to Friday, 9 AM to 4 PM) and the school phone
  number, then use [END_CALL]. This is the ONLY time you may use [END_CALL].
- NEVER output or mention [SYSTEM:...] tags in your spoken response.
- NEVER ask "anything else?" unless you see [SYSTEM: ask about more questions].

SPEECH STYLE:
- Spoken, natural sentences. Not formal written English.
- Contractions always. Lists never.
- English only — no Hindi, no regional language words of any kind.

STRICT RULES:
1. NEVER greet again after the first message.
2. NEVER re-introduce yourself mid-call.
3. ONLY discuss {dimension.upper()}.
4. 2 to 3 sentences per reply maximum.
5. Every single word in your reply must be English. Remove any non-English word before responding.
6. NEVER invent ANY detail not explicitly stated in the concern details above.
   This means: no exam names (SSE, ICSE, boards, etc.), no report cards, no subject
   names, no grades, no test scores, no teacher names, no student names, no club or
   team names, no incident descriptions beyond what was given. If you were not told
   it, do not say it. If a parent asks about something you have no data on, say:
   "I'd prefer we go over those details when we meet in person."
7. End EVERY reply with one control tag on its own line:
   [CONTINUE]  — keep going
   [END_CALL]  — end now (only after farewell step 2 is complete)
   [SCHEDULE_MEETING: <day_preference>]  — only when scheduling a meeting
8. Tags are for system only — never spoken aloud.

You are in a REAL phone call. React naturally. Be human. Speak English only.
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
        self.stage          = STAGE_INTRO
        self.messages: list[dict] = []
        self.meeting_pending: bool = False   # True while awaiting slot injection
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
        if self.stage in (STAGE_FAREWELL, STAGE_CLOSING):
            return
        if self.stage == STAGE_INTRO and self.turn_count >= 1:
            self.stage = STAGE_AVAILABILITY
        elif self.stage == STAGE_AVAILABILITY and self.turn_count >= 2:
            self.stage = STAGE_CONVERSATION
        elif self.stage == STAGE_CONVERSATION and self.turn_count >= 5:
            self.stage = STAGE_SOLUTION


# ─────────────────────────────────────────────────────────────────
#  SCHEDULE TAG PARSER
# ─────────────────────────────────────────────────────────────────
_SCHEDULE_TAG_RE = re.compile(
    r'\[SCHEDULE_MEETING:\s*([^\]]+)\]', re.IGNORECASE
)

def _extract_schedule_tag(ai_text: str) -> tuple[str, str | None]:
    """
    Returns (cleaned_text, day_preference | None).
    cleaned_text has the tag removed; day_preference is the raw preference string.
    """
    match = _SCHEDULE_TAG_RE.search(ai_text)
    if not match:
        return ai_text, None
    day_pref = match.group(1).strip().lower()
    cleaned = _SCHEDULE_TAG_RE.sub("", ai_text).strip()
    return cleaned, day_pref


def _resolve_slot(day_pref: str) -> dict | None:
    """
    Use ScheduleManager to resolve a day preference to a concrete slot.
    Imported here (lazy) so it is never loaded during normal call flow.
    """
    from services.schedule_manager import ScheduleManager, DAY_INDEX
    sm = ScheduleManager()

    # "next week" preference
    if "next week" in day_pref:
        slot = sm.get_next_available_slot(prefer_next_week=True)
        return slot

    # Named day (e.g. "thursday")
    for day_name in DAY_INDEX:
        if day_name in day_pref:
            # Determine whether to look at current or next week
            from datetime import date, timedelta
            today = date.today()
            target_weekday = DAY_INDEX[day_name]
            days_ahead = (target_weekday - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # same day → next occurrence
            target_date = today + timedelta(days=days_ahead)
            from services.schedule_manager import _next_weeks_monday
            use_next = target_date >= _next_weeks_monday()
            slots = sm.get_available_slots_for_day(day_name.capitalize(), next_week=use_next)
            if slots:
                return {**slots[0], "date": target_date.isoformat()}
            return None  # named day requested but fully booked

    # "tomorrow" or generic "yes"
    slot = sm.get_next_available_slot(prefer_next_week=False)
    return slot


# ─────────────────────────────────────────────────────────────────
#  MAIN SERVICE
# ─────────────────────────────────────────────────────────────────
class TwoWayAIVoiceService:
    MODEL = "llama-3.1-8b-instant"
    MAX_HISTORY = 12

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
            print("[OK] Twilio Connected")

        if self._ai_ready:
            self._groq = Groq(api_key=self._groq_key)
            print(f"[OK] Groq Connected  [{self.MODEL}]")
            try:
                self._groq.chat.completions.create(
                    model=self.MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                )
                print("[OK] Groq Pre-warmed")
            except Exception:
                pass
        else:
            print("[WARN] GROQ_API_KEY not set — Demo mode active")

        self._conversations: dict[str, ConversationState] = {}
        self.calls_made = []

    # ── Ask Groq ──────────────────────────────────────────────────
    def _ask_groq(self, state: ConversationState, parent_speech: str) -> str:
        state.messages.append({"role": "user", "content": parent_speech})
        state.turn_count += 1
        state.advance_stage()

        try:
            recent = state.messages[-self.MAX_HISTORY:]
            response = self._groq.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": state.system_prompt}
                ] + recent,
                temperature=0.5,
                max_tokens=80,
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

    # ── Parse control tags ────────────────────────────────────────
    def _parse_reply(self, ai_text: str) -> tuple[str, bool]:
        end_call = "[END_CALL]" in ai_text
        spoken = (ai_text
                  .replace("[END_CALL]", "")
                  .replace("[CONTINUE]", "")
                  .strip())
        return spoken, end_call

    # ── Build TwiML ───────────────────────────────────────────────
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
        safe_parent = html.escape(payload.parent_name)
        safe_phone  = html.escape(self._phone)
        webhook     = f"{self._ngrok_url}/handle-parent-response"

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{webhook}"
            method="POST"
            timeout="8"
            speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">Hello! Am I speaking with {safe_parent}?</Say>
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
    <Say voice="Polly.Aditi" language="en-IN">Thank you for your time. Please contact the college at {safe_ph}. Goodbye.</Say>
</Response>"""

        # Stage-aware closing detection
        if _parent_wants_to_end(parent_speech, state.stage):
            if state.turn_count >= 4:
                if state.stage == STAGE_FAREWELL:
                    speech_for_ai = (
                        parent_speech +
                        " [SYSTEM: end call now - mention school hours and phone number]"
                    )
                else:
                    state.stage = STAGE_FAREWELL
                    speech_for_ai = (
                        parent_speech +
                        " [SYSTEM: ask about more questions]"
                    )
            else:
                speech_for_ai = parent_speech
        else:
            speech_for_ai = parent_speech

        ai_text = self._ask_groq(state, speech_for_ai)

        # ── Schedule meeting tag detection ────────────────────────
        pre_schedule_text, day_pref = _extract_schedule_tag(ai_text)

        if day_pref is not None:
            # Parent agreed to a meeting — resolve a real slot
            slot = _resolve_slot(day_pref)

            if slot:
                # Book the slot immediately
                from services.schedule_manager import ScheduleManager
                sm = ScheduleManager()
                sm.book_slot(slot["day"], slot["start_time"], next_week=slot.get("use_next_week", False))

                slot_str = f"{slot['day']} at {slot['start_time']}"
                logger.info(f"[Scheduler] Booked: {slot_str} for reg={registration}")

                # Inject confirmed slot into Priya's reply
                if pre_schedule_text:
                    spoken = (
                        f"{pre_schedule_text} "
                        f"I've checked and we have a slot available on {slot_str}. "
                        f"Does that work for you?"
                    )
                else:
                    spoken = (
                        f"I've checked my schedule and we have a slot available on {slot_str}. "
                        f"Does that work for you?"
                    )
                # Mark stage as solution since a meeting was proposed
                state.stage = STAGE_SOLUTION
                return self._twiml(spoken, end_call=False)

            else:
                # No slots available — tell the parent
                spoken = (
                    f"{pre_schedule_text} "
                    f"I'm afraid we don't have any free slots available in the next few days. "
                    f"Please call us directly at {self._phone} to arrange a time that suits you."
                ).strip()
                return self._twiml(spoken, end_call=False)

        # ── Normal flow ───────────────────────────────────────────
        spoken, end_call = self._parse_reply(ai_text)

        if end_call:
            state.ended = True

        print(f"  [P] Parent : \"{parent_speech}\"")
        print(f"  [AI] Priya  : \"{spoken[:90]}{'...' if len(spoken) > 90 else ''}\"")
        print(f"  [INFO] Turn: {state.turn_count} | Stage: {state.stage} | End: {end_call}")

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
        print(f"[OK] Ready: {payload.student_name} [{payload.dimension.upper()} / {payload.risk_level}]")

        twiml = self._opening_twiml(payload)

        try:
            print(f"\n[CALL] Calling {payload.parent_name} ({payload.to_number})...")
            call = self._client.calls.create(
                to=payload.to_number,
                from_=self._from_number,
                twiml=twiml
            )
            self.calls_made.append(payload)
            print(f"   [OK] SID: {call.sid}\n")
            return NotificationResult(
                success=True, sid=call.sid,
                student_id=payload.registration, channel="ai_2way_groq"
            )
        except Exception as e:
            print(f"   [ERR] {str(e)[:120]}")
            return NotificationResult(
                success=False, error_message=str(e),
                student_id=payload.registration
            )

    # ── make_batch_calls ──────────────────────────────────────────
    def make_batch_calls(self, payloads: list) -> dict:
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        print(f"\n[AI] Groq 2-Way Calls: {len(payloads)}\n")
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
        print(f"\n[OK] {results['successful']}/{results['total']}\n")
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