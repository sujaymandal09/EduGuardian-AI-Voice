"""
services/twilio_groq_voice.py
──────────────────────────────
2-Way AI Voice — Groq (llama-3.1-8b-instant) + Twilio
Now with SQLite conversation logging + auto-summary on call end.
"""
import html
import logging
import os
import re
import threading
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
        "HIGH": f"""
MEETING RULES (HIGH risk):
- Suggest tomorrow at 10 AM.
- If parent says not available: persuade ONCE warmly — like a caring teacher.
  Example: "I completely understand. I just want to mention that the situation
  is quite urgent and the sooner we can meet, the better it will be for your child.
  Is there any possibility this week at all?"
- After that ONE gentle push, IMMEDIATELY accept whatever time they give.
- EXCEPTION — if parent proposes a far-future date like "next month", "next year",
  or "in a few weeks", do NOT accept and do NOT emit [SCHEDULE_MEETING:...].
  Instead say something like:
  "I really do understand you're very busy, but I want to be honest — this situation
  is quite serious and waiting that long could have a significant impact on your child.
  I'd strongly encourage you to call us directly at {phone} as soon as you can so
  we can find a time that works for you."
  Then use [CONTINUE] and let the parent respond.
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

    if risk_level.upper() == "HIGH":
        scheduling_section = f"""
MEETING SCHEDULING:
- When a parent agrees to a meeting, DO NOT suggest any specific time yourself.
- Instead, say exactly this sentence first: "Please wait for a moment, I need to check my schedule."
  Then on a NEW LINE emit ONLY this control tag (nothing else on that line, no other text):
  [SCHEDULE_MEETING: <day_preference>]
  Where <day_preference> is ONE of:
    - "tomorrow"        — parent said yes/tomorrow/as soon as possible
    - "next week"       — parent said next week
    - "<day name>"      — parent named a specific day, e.g. "Thursday" or "Friday"
    - "next_available"  — parent rejected a time but did NOT name a specific new day
- CRITICAL: The tag must appear on its OWN line.
- NEVER guess, invent, or hardcode any time. NEVER say "scheduling_meeting" aloud.
- If the parent says "today" or "right now", emit [SCHEDULE_MEETING: today].
- CRITICAL: This rule applies at ALL stages including farewell. If the parent names
  a day or asks about scheduling at any point, ALWAYS emit [SCHEDULE_MEETING: <day>].
  Never verbally confirm a meeting time without first emitting the tag — you do not
  know what times are available. The system provides the actual time.
- If a meeting has already been booked (system confirmed a time to the parent) and
  the parent asks "what time?" or "what was the time?", repeat the confirmed time
  from the most recent system-confirmed booking in the conversation. Do not invent
  a new time.
"""
    else:
        scheduling_section = f"""
MEETING SCHEDULING (DISABLED for this call):
- NEVER use [SCHEDULE_MEETING:...] — it is disabled for {risk_level.upper()} risk calls.
- If the parent asks to meet in person, say:
  "Of course, please call us at {phone} and we'll arrange a time that suits you."
"""

    return f"""You are Priya — a warm, experienced school counselor calling from {school}.
You are speaking with {parent_name}, parent of {student_name}.
School phone: {phone}
Office hours: Monday to Friday, 9 AM to 4 PM

YOUR PERSONALITY:
- You sound like a real human. Warm, calm, genuinely caring.
- Speak in clear, natural, grammatically correct English only. No Hindi or mixed-language words whatsoever.
- Do NOT use words such as: ji, haan, accha, acha, bilkul, theek hai, nahi, or any other Hindi or regional language term.
- If the parent speaks in Hindi or mixed language, always reply in English only.
- You LISTEN first. You acknowledge before moving forward.
- Never robotic. Never scripted. Never repeat yourself.
- 2 to 3 sentences per reply. Phone calls need space.
- Use contractions: I'll, we'll, that's, it's, don't.
- Vary your sentence starters. Don't always begin with the parent's name.

STRICT CALL FLOW — FOLLOW THIS ORDER:
1. Parent confirms who they are → in ONE sentence: acknowledge warmly AND introduce
   yourself as Priya, school counselor at {school}.
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

{dimension_context}

{meeting_rules}

{scheduling_section}

INTERRUPTION HANDLING:
- If parent speaks mid-reply — respond only to what they said.
- Drop what you were saying. React to their words directly.
- Never repeat a sentence they interrupted.

CLOSING:
- NEVER use [END_CALL] on your own.
- When the system adds [SYSTEM: ask about more questions], ask the parent
  if they have any other questions or concerns, in ONE warm sentence. Then use [CONTINUE].
- When the system adds [SYSTEM: end call now], give ONE warm goodbye sentence that
  includes the office hours and the school phone number, then use [END_CALL].
- NEVER output or mention [SYSTEM:...] tags in your spoken response.

SPEECH STYLE:
- Spoken, natural sentences. Not formal written English.
- Contractions always. Lists never.
- English only.

STRICT RULES:
1. NEVER greet again after the first message.
2. NEVER re-introduce yourself mid-call.
3. ONLY discuss {dimension.upper()}.
4. 2 to 3 sentences per reply maximum.
5. Every single word in your reply must be English.
6. NEVER invent ANY detail not explicitly stated in the concern details above.
7. End EVERY reply with one control tag on its own line:
   [CONTINUE]  — keep going
   [END_CALL]  — end now (only after farewell)
   [SCHEDULE_MEETING: <day_preference>]  — only when scheduling
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
        self.meeting_pending: bool = False
        self.pending_slots: list[dict] = []
        self.awaiting_slot_choice: bool = False
        self.last_booked_slot: dict | None = None

        # ── DB tracking ───────────────────────────────────────────
        # call_id is set by TwoWayAIVoiceService after Twilio returns a SID.
        # Until then we use a temporary placeholder.
        self.call_id: str = f"PENDING_{payload.registration}"
        self.db_turn_number: int = 0   # incremented with every saved turn

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

    def log_turn(self, role: str, message: str) -> None:
        """Save a turn to SQLite. role = 'assistant' (Priya) or 'user' (parent)."""
        if not isinstance(self.call_id, int):
            return   # DB record not yet created — skip silently
        from services.database import save_turn
        try:
            save_turn(
                call_id=self.call_id,
                role=role,
                content=message,
                turn_order=self.db_turn_number,
            )
            self.db_turn_number += 1
        except Exception as e:
            logger.error(f"[DB] log_turn failed: {e}")


# ─────────────────────────────────────────────────────────────────
#  SCHEDULE TAG PARSER
# ─────────────────────────────────────────────────────────────────
_SCHEDULE_TAG_RE = re.compile(
    r'\[SCHEDULE_MEETING[^\]]*\]', re.IGNORECASE
)
_SCHEDULE_TAG_CAP_RE = re.compile(
    r'\[SCHEDULE_MEETING:\s*([^\]]+)\]', re.IGNORECASE
)

def _extract_schedule_tag(ai_text: str) -> tuple[str, str | None]:
    match = _SCHEDULE_TAG_CAP_RE.search(ai_text)
    if not match:
        cleaned = _SCHEDULE_TAG_RE.sub("", ai_text).strip()
        return cleaned, None
    day_pref = match.group(1).strip().lower()
    cleaned = _SCHEDULE_TAG_RE.sub("", ai_text).strip()
    cleaned = re.sub(r'\bscheduling[_\s]meeting\b', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned, day_pref


_MEETING_PROPOSAL_KEYWORDS = (
    "meet", "come in", "schedule", "arrange", "in person", "appointment"
)

_SIMPLE_AFFIRMATIVE = {
    "yes", "yes!", "yes?", "yeah", "yep", "yup", "sure", "okay", "ok",
    "of course", "available", "i'm available", "i am available", "why not",
    "can do", "works", "that's fine", "that works", "alright", "will do",
    "no problem", "sounds good", "happy to", "i'll come", "we'll come",
    "please", "go ahead", "let's do it", "let's do that", "sure thing",
    "absolutely", "certainly", "definitely", "by all means",
}

def _last_ai_proposed_meeting(state: "ConversationState") -> bool:
    last_ai = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "assistant"),
        ""
    ).lower()
    return any(kw in last_ai for kw in _MEETING_PROPOSAL_KEYWORDS)


def _is_simple_affirmative(speech: str) -> bool:
    text = speech.lower().strip().rstrip("?,!.")
    return text in _SIMPLE_AFFIRMATIVE


def _format_time_spoken(t: str) -> str:
    try:
        h, m = map(int, t.split(":"))
        period = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        if m == 0:
            return f"{h12} {period}"
        return f"{h12}:{m:02d} {period}"
    except Exception:
        return t


def _sanitise_spoken(text: str) -> str:
    text = _SCHEDULE_TAG_RE.sub("", text)
    text = re.sub(r'\[CONTINUE\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[END_CALL\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SYSTEM:[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bscheduling[_\s]meeting\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SCHEDULE_MEETING[^\]]*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SCHEDULE_[^\]]*$', '', text, flags=re.IGNORECASE)
    return text.strip()


def _resolve_slot(day_pref: str) -> dict | None:
    from services.schedule_manager import ScheduleManager, DAY_INDEX
    sm = ScheduleManager()
    if any(kw in day_pref for kw in ("today", "right now", "now", "immediately")):
        return sm.get_today_available_slot()
    if "next week" in day_pref:
        return sm.get_next_available_slot(prefer_next_week=True)
    for day_name in DAY_INDEX:
        if day_name in day_pref:
            slots = _resolve_slots_for_day(day_name)
            return slots[0] if slots else None
    return sm.get_next_available_slot(prefer_next_week=False)


def _resolve_next_available(state: "ConversationState") -> tuple[list[dict], str]:
    from services.schedule_manager import ScheduleManager, _next_weeks_monday
    from datetime import date, timedelta
    sm = ScheduleManager()
    today = date.today()
    if state.last_booked_slot:
        try:
            start_date = date.fromisoformat(state.last_booked_slot["date"])
        except (KeyError, ValueError):
            start_date = today + timedelta(days=1)
        excluded_time = state.last_booked_slot["start_time"]
        excluded_day  = state.last_booked_slot["day"].lower()
    else:
        start_date    = today + timedelta(days=1)
        excluded_time = None
        excluded_day  = None
    for offset in range(8):
        candidate = start_date + timedelta(days=offset)
        if candidate <= today:
            continue
        if candidate.weekday() == 6:
            continue
        day_name = candidate.strftime("%A")
        use_next = candidate >= _next_weeks_monday()
        slots = sm.get_available_slots_for_day(day_name, next_week=use_next)
        if excluded_day and day_name.lower() == excluded_day and excluded_time:
            slots = [s for s in slots if s["start_time"] != excluded_time]
        if slots:
            return [
                {**s, "date": candidate.isoformat(), "use_next_week": use_next}
                for s in slots
            ], day_name
    return [], ""


def _try_next_day_slots(after_day: str, after_date: str) -> tuple[list[dict], str]:
    from services.schedule_manager import ScheduleManager, _next_weeks_monday
    from datetime import date, timedelta
    sm = ScheduleManager()
    today = date.today()
    try:
        start = date.fromisoformat(after_date) + timedelta(days=1)
    except (ValueError, TypeError):
        start = today + timedelta(days=1)
    for offset in range(7):
        candidate = start + timedelta(days=offset)
        if candidate.weekday() == 6:
            continue
        day_name = candidate.strftime("%A")
        use_next = candidate >= _next_weeks_monday()
        slots = sm.get_available_slots_for_day(day_name, next_week=use_next)
        if slots:
            return [
                {**s, "date": candidate.isoformat(), "use_next_week": use_next}
                for s in slots
            ], day_name
    return [], ""


def _resolve_slots_for_day(day_name: str) -> list[dict]:
    from services.schedule_manager import ScheduleManager, DAY_INDEX, _next_weeks_monday
    from datetime import date, timedelta
    sm = ScheduleManager()
    today = date.today()
    target_weekday = DAY_INDEX.get(day_name.lower())
    if target_weekday is None:
        return []
    days_ahead = (target_weekday - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    target_date = today + timedelta(days=days_ahead)
    use_next = target_date >= _next_weeks_monday()
    raw_slots = sm.get_available_slots_for_day(day_name.capitalize(), next_week=use_next)
    return [{**s, "date": target_date.isoformat()} for s in raw_slots]


# ─────────────────────────────────────────────────────────────────
#  BACKGROUND SUMMARY TRIGGER
# ─────────────────────────────────────────────────────────────────
def _trigger_summary(state, groq_api_key: str) -> None:
    """
    Called in a background thread when the call ends.
    Generates and saves the summary without blocking the TwiML response.
    """
    try:
        from groq import Groq
        from services.summary_agent import SummaryAgent
        agent = SummaryAgent(Groq(api_key=groq_api_key))
        agent.generate_and_save(
            call_id=state.call_id,
            payload=state.payload,
            last_booked_slot=state.last_booked_slot,
        )
        logger.info(f"[Summary] Done for call_id={state.call_id}")
    except Exception as e:
        logger.error(f"[Summary] Failed for call_id={state.call_id}: {e}")


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
        # Log parent turn to DB
        state.log_turn("user", parent_speech)

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

        # Log Priya's reply (cleaned) to DB
        spoken_for_log = _sanitise_spoken(
            ai_text.replace("[END_CALL]", "").replace("[CONTINUE]", "")
        )
        state.log_turn("assistant", spoken_for_log)

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
    def _twiml(self, spoken: str, end_call: bool, state: ConversationState | None = None) -> str:
        safe    = html.escape(spoken)
        safe_ph = html.escape(self._phone)
        webhook = f"{self._ngrok_url}/handle-parent-response"

        if end_call:
            # Trigger summary in background
            if state and self._groq_key:
                threading.Thread(
                    target=_trigger_summary,
                    args=(state, self._groq_key),
                    daemon=True
                ).start()
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">{safe}</Say>
    <Pause length="1"/>
    <Hangup/>
</Response>"""
        else:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN"
            action="{webhook}"
            method="POST"
            timeout="8"
            speechTimeout="auto"
            bargeIn="true">
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
    <Hangup/>
</Response>"""

        if not self._ai_ready:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Thank you for your time. Please contact the college at {safe_ph}. Goodbye.</Say>
</Response>"""

        # ── Named-day slot-choice waiting state ──────────────────────
        if state.awaiting_slot_choice and state.pending_slots:
            return self._handle_slot_choice(state, parent_speech, registration)

        # ── Meeting-agreement shortcut (HIGH risk only) ───────────
        if (
            state.payload.risk_level.upper() == "HIGH"
            and state.stage in (STAGE_AVAILABILITY, STAGE_CONVERSATION, STAGE_SOLUTION)
            and not state.meeting_pending
            and _is_simple_affirmative(parent_speech)
            and _last_ai_proposed_meeting(state)
        ):
            slot = _resolve_slot("tomorrow")
            if slot:
                from services.schedule_manager import ScheduleManager
                sm = ScheduleManager()
                if state.last_booked_slot:
                    sm.cancel_slot(
                        state.last_booked_slot["day"],
                        state.last_booked_slot["start_time"],
                        next_week=state.last_booked_slot.get("use_next_week", False),
                    )
                sm.book_slot(slot["day"], slot["start_time"], next_week=slot.get("use_next_week", False))
                state.last_booked_slot = slot
                slot_str = f"{slot['day']} at {_format_time_spoken(slot['start_time'])}"
                spoken = (
                    f"That's great to hear! Let me just check my schedule. "
                    f"I've booked you in for {slot_str}. "
                    f"We really look forward to seeing you then. "
                    f"Is there anything else you'd like to discuss before we wrap up?"
                )
                state.log_turn("user", parent_speech)
                state.log_turn("assistant", spoken)
                state.messages.append({"role": "user", "content": parent_speech})
                state.messages.append({"role": "assistant", "content": spoken})
                state.turn_count += 1
                state.stage = STAGE_FAREWELL
                return self._twiml(spoken, end_call=False, state=state)

        # ── Stage-aware closing detection ─────────────────────────
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
        pre_schedule_text = _sanitise_spoken(pre_schedule_text)

        # MEDIUM/LOW: suppress scheduling
        if day_pref is not None and state.payload.risk_level.upper() != "HIGH":
            pre_schedule_text = re.sub(
                r"please wait[^.!?]*[.!?]?\s*", "", pre_schedule_text, flags=re.IGNORECASE
            ).strip()
            pre_schedule_text = re.sub(
                r"let me check my schedule[^.!?]*[.!?]?\s*", "", pre_schedule_text, flags=re.IGNORECASE
            ).strip()
            spoken, end_call = self._parse_reply(pre_schedule_text)
            spoken = _sanitise_spoken(spoken)
            if end_call:
                state.ended = True
            return self._twiml(spoken, end_call, state=state)

        # Far-future rejection
        _FAR_FUTURE = ("next month", "month", "next year", "year", "few weeks", "some time")
        if (
            day_pref is not None
            and state.payload.risk_level.upper() == "HIGH"
            and any(kw in day_pref for kw in _FAR_FUTURE)
            and "next week" not in day_pref
        ):
            spoken = (
                f"I completely understand you're very busy, and I don't want to add to your stress. "
                f"But I do want to be honest — this situation is quite serious, and waiting that long "
                f"could have a real impact on your child's future. "
                f"Please do call us directly at {self._phone} as soon as possible "
                f"and we'll do everything we can to find a time that works for you."
            )
            spoken = _sanitise_spoken(spoken)
            state.stage = STAGE_SOLUTION
            return self._twiml(spoken, end_call=False, state=state)

        if day_pref is not None:
            from services.schedule_manager import DAY_INDEX

            named_day = None
            if "next week" not in day_pref and day_pref not in ("tomorrow", "yes", "soonest"):
                for d in DAY_INDEX:
                    if d in day_pref:
                        named_day = d
                        break

            if "next_available" in day_pref:
                if state.last_booked_slot:
                    from services.schedule_manager import ScheduleManager as _SM
                    _SM().cancel_slot(
                        state.last_booked_slot["day"],
                        state.last_booked_slot["start_time"],
                        next_week=state.last_booked_slot.get("use_next_week", False),
                    )
                    state.last_booked_slot = None
                slots, day_name = _resolve_next_available(state)
                if slots:
                    state.pending_slots       = slots
                    state.awaiting_slot_choice = True
                    state.stage               = STAGE_SOLUTION
                    times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                    spoken = (
                        f"{pre_schedule_text} "
                        f"I've checked and on {day_name} we have slots available at {times}. "
                        f"Which time works best for you?"
                    ).strip()
                    return self._twiml(spoken, end_call=False, state=state)
                else:
                    spoken = (
                        f"{pre_schedule_text} "
                        f"I'm afraid we don't have any free slots available over the next few days. "
                        f"Please call us directly at {self._phone} to arrange a time that suits you."
                    ).strip()
                    return self._twiml(spoken, end_call=False, state=state)

            if named_day:
                if state.last_booked_slot:
                    from services.schedule_manager import ScheduleManager as _SM
                    _SM().cancel_slot(
                        state.last_booked_slot["day"],
                        state.last_booked_slot["start_time"],
                        next_week=state.last_booked_slot.get("use_next_week", False),
                    )
                    state.last_booked_slot = None
                slots = _resolve_slots_for_day(named_day)
                if slots:
                    state.pending_slots = slots
                    state.awaiting_slot_choice = True
                    state.stage = STAGE_SOLUTION
                    times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                    spoken = (
                        f"{pre_schedule_text} "
                        f"I've checked and on {named_day.capitalize()} we have slots available at {times}. "
                        f"Which time works best for you?"
                    ).strip()
                    return self._twiml(spoken, end_call=False, state=state)
                else:
                    spoken = (
                        f"{pre_schedule_text} "
                        f"I'm sorry, we don't have any free slots on {named_day.capitalize()}. "
                        f"Would another day work for you?"
                    ).strip()
                    return self._twiml(spoken, end_call=False, state=state)

            else:
                is_today = any(kw in day_pref for kw in ("today", "right now", "now", "immediately"))
                slot = _resolve_slot(day_pref)
                if slot:
                    from services.schedule_manager import ScheduleManager
                    sm = ScheduleManager()
                    if state.last_booked_slot:
                        sm.cancel_slot(
                            state.last_booked_slot["day"],
                            state.last_booked_slot["start_time"],
                            next_week=state.last_booked_slot.get("use_next_week", False),
                        )
                    sm.book_slot(slot["day"], slot["start_time"], next_week=slot.get("use_next_week", False))
                    state.last_booked_slot = slot
                    slot_str = f"{slot['day']} at {_format_time_spoken(slot['start_time'])}"
                    spoken = (
                        f"{pre_schedule_text} "
                        f"I've booked you in for {slot_str}. "
                        f"We look forward to seeing you then. "
                        f"Is there anything else you'd like to discuss before we wrap up?"
                    ).strip()
                    state.stage = STAGE_FAREWELL
                    return self._twiml(spoken, end_call=False, state=state)
                else:
                    if is_today:
                        spoken = (
                            f"{pre_schedule_text} "
                            f"I'm afraid all of today's slots have already passed or been filled. "
                            f"Could we look at tomorrow, or would you prefer to call us directly "
                            f"at {self._phone} to arrange something?"
                        ).strip()
                    else:
                        spoken = (
                            f"{pre_schedule_text} "
                            f"I'm afraid we don't have any free slots available in the next few days. "
                            f"Please call us directly at {self._phone} to arrange a time that suits you."
                        ).strip()
                    return self._twiml(spoken, end_call=False, state=state)

        # ── Normal flow ───────────────────────────────────────────
        spoken, end_call = self._parse_reply(ai_text)
        spoken = _sanitise_spoken(spoken)

        if end_call:
            state.ended = True

        print(f"  [P] Parent : \"{parent_speech}\"")
        print(f"  [AI] Priya  : \"{spoken[:90]}{'...' if len(spoken) > 90 else ''}\"")
        print(f"  [INFO] Turn: {state.turn_count} | Stage: {state.stage} | End: {end_call}")

        return self._twiml(spoken, end_call, state=state)

    # ── Slot choice handler ───────────────────────────────────────
    def _handle_slot_choice(self, state: "ConversationState", parent_speech: str, registration: str) -> str:
        speech_lower = parent_speech.lower().strip()

        from services.schedule_manager import DAY_INDEX
        current_day = state.pending_slots[0]["day"].lower() if state.pending_slots else ""
        requested_day = None
        for d in DAY_INDEX:
            if d in speech_lower:
                requested_day = d
                break

        if requested_day and requested_day != current_day:
            if state.last_booked_slot:
                from services.schedule_manager import ScheduleManager
                sm = ScheduleManager()
                sm.cancel_slot(
                    state.last_booked_slot["day"],
                    state.last_booked_slot["start_time"],
                    next_week=state.last_booked_slot.get("use_next_week", False),
                )
                state.last_booked_slot = None
            state.pending_slots        = []
            state.awaiting_slot_choice = False
            slots = _resolve_slots_for_day(requested_day)
            if slots:
                state.pending_slots        = slots
                state.awaiting_slot_choice = True
                state.stage                = STAGE_SOLUTION
                times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                spoken = (
                    f"Of course! On {requested_day.capitalize()} we have slots available at {times}. "
                    f"Which time works best for you?"
                )
                return self._twiml(spoken, end_call=False, state=state)
            else:
                spoken = (
                    f"I'm sorry, we don't have any free slots on {requested_day.capitalize()}. "
                    f"Would another day work for you?"
                )
                return self._twiml(spoken, end_call=False, state=state)

        decline_signals = ["no", "none", "different day", "another day", "cancel", "forget it",
                           "never mind", "not available", "can't make", "cannot make",
                           "won't work", "doesn't work"]
        if any(sig in speech_lower for sig in decline_signals):
            rejected_day  = state.pending_slots[0]["day"]  if state.pending_slots else ""
            rejected_date = state.pending_slots[0].get("date", "") if state.pending_slots else ""
            state.pending_slots        = []
            state.awaiting_slot_choice = False
            next_slots, next_day = _try_next_day_slots(rejected_day, rejected_date)
            if next_slots:
                state.pending_slots        = next_slots
                state.awaiting_slot_choice = True
                state.stage                = STAGE_SOLUTION
                times = ", ".join(_format_time_spoken(s["start_time"]) for s in next_slots)
                spoken = (
                    f"No problem. I've checked and on {next_day} we have slots available at {times}. "
                    f"Would any of those work for you?"
                )
                return self._twiml(spoken, end_call=False, state=state)
            else:
                state.stage = STAGE_FAREWELL
                spoken = (
                    f"I completely understand. Unfortunately we don't have any free slots available "
                    f"over the next few days. Please call us at {self._phone} and we'll do our best "
                    f"to find a time that works for you."
                )
                return self._twiml(spoken, end_call=False, state=state)

        matched_slot = None
        for slot in state.pending_slots:
            st = slot["start_time"]
            if st.replace(":", "") in speech_lower.replace(":", ""):
                matched_slot = slot
                break
            spoken_fmt = _format_time_spoken(st).lower()
            spoken_nsp = spoken_fmt.replace(" ", "")
            speech_nsp = speech_lower.replace(" ", "")
            if spoken_fmt in speech_lower or spoken_nsp in speech_nsp:
                matched_slot = slot
                break
            hour24 = st.split(":")[0].lstrip("0") or "0"
            if f" {hour24} " in f" {speech_lower} " or speech_lower.startswith(hour24):
                matched_slot = slot
                break
            hour12 = spoken_fmt.split(":")[0].split(" ")[0]
            if hour12 != hour24:
                if f" {hour12} " in f" {speech_lower} " or speech_lower.startswith(hour12 + " ") or speech_lower.startswith(hour12 + ":"):
                    matched_slot = slot
                    break

        if matched_slot:
            from services.schedule_manager import ScheduleManager
            sm = ScheduleManager()
            if state.last_booked_slot:
                sm.cancel_slot(
                    state.last_booked_slot["day"],
                    state.last_booked_slot["start_time"],
                    next_week=state.last_booked_slot.get("use_next_week", False),
                )
                state.last_booked_slot = None
            booked = sm.book_slot(
                matched_slot["day"],
                matched_slot["start_time"],
                next_week=matched_slot.get("use_next_week", False),
            )
            state.pending_slots = []
            state.awaiting_slot_choice = False
            if booked:
                state.last_booked_slot = matched_slot
                slot_str = f"{matched_slot['day']} at {_format_time_spoken(matched_slot['start_time'])}"
                state.stage = STAGE_FAREWELL
                spoken = (
                    f"Perfect, I've booked you in for {slot_str}. "
                    f"We look forward to seeing you then. "
                    f"Is there anything else you'd like to discuss before we wrap up?"
                )
                return self._twiml(spoken, end_call=False, state=state)
            else:
                remaining = _resolve_slots_for_day(matched_slot["day"].lower())
                if remaining:
                    state.pending_slots = remaining
                    state.awaiting_slot_choice = True
                    times = ", ".join(_format_time_spoken(s["start_time"]) for s in remaining)
                    spoken = (
                        f"I'm sorry, that slot was just taken. "
                        f"The remaining times available on {matched_slot['day']} are {times}. "
                        f"Which would you prefer?"
                    )
                else:
                    state.awaiting_slot_choice = False
                    spoken = (
                        f"I'm sorry, all slots on {matched_slot['day']} have just been filled. "
                        f"Please call us at {self._phone} to arrange an alternative time."
                    )
                return self._twiml(spoken, end_call=False, state=state)

        times = ", ".join(_format_time_spoken(s["start_time"]) for s in state.pending_slots)
        day = state.pending_slots[0]["day"] if state.pending_slots else "that day"
        spoken = (
            f"I didn't quite catch which time you prefer. "
            f"The available slots on {day} are {times}. "
            f"Which one works best for you?"
        )
        return self._twiml(spoken, end_call=False, state=state)

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

        # Create DB record before the call so turns save correctly
        try:
            from services.database import create_call_record
            db_call_id = create_call_record(
                registration=payload.registration,
                student_name=payload.student_name,
                parent_name=payload.parent_name,
                dimension=payload.dimension,
                risk_level=payload.risk_level,
            )
            state.call_id = db_call_id
            print(f"   [DB] Call record created: db_call_id={db_call_id}")
        except Exception as e:
            logger.error(f"[DB] create_call_record failed: {e}")

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
        print(f"  DEMO MODE — {payload.dimension.upper()}")
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