"""
services/twilio_groq_voice.py
──────────────────────────────
2-Way AI Voice — Groq (llama-3.1-8b-instant) + Twilio
"""
import html
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from groq import Groq
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)

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
    "theek hai", "theek hain", "thik hai", "alright then", "all right then",
    "got it", "understood", "i understand",
    "bye", "goodbye", "good bye", "bye bye",
    "dhanyawad", "dhanyavaad", "shukriya",
    "bas ji", "bas theek hai", "haan ji okay", "haan ji theek",
    "no questions", "no more questions", "no doubts",
    "koi sawaal nahi", "koi doubt nahi",
    "samajh gaya", "samajh gayi", "samajh liya",
    "that's all", "that is all", "no more",
    "i'll come", "we'll come", "i will come", "we will come",
    "see you", "see you then", "see you tomorrow",
    "day after tomorrow works", "that works", "that's fine",
    "okay i'll come", "okay we'll come", "noted", "will do",
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
#  FRAGMENTED SPEECH DETECTOR
# ─────────────────────────────────────────────────────────────────
_RECORDING_FRAGMENTS = {
    "be recorded", "recorded", "call will be recorded", "this call",
    "will be recorded", "call may be recorded",
}

def _is_recording_notice_fragment(speech: str) -> bool:
    return speech.lower().strip() in _RECORDING_FRAGMENTS

def _is_too_short_to_process(speech: str) -> bool:
    words = [w for w in speech.lower().split() if len(w) > 1]
    return len(words) <= 2 and not any(
        kw in speech.lower() for kw in (
            "yes", "no", "bye", "okay", "ok", "sure", "meet",
            "tomorrow", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "busy", "free", "time", "can",
            "quick", "done", "finished", "that's all",
        )
    )


# ─────────────────────────────────────────────────────────────────
#  SINGLE-SLOT AUTO-CONFIRM
# ─────────────────────────────────────────────────────────────────
_SINGLE_SLOT_CONFIRM = {
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "fine",
    "alright", "that's fine", "that works", "sounds good", "good",
    "perfect", "great", "go ahead", "confirmed", "that one",
    "yes please", "that's good", "works for me", "i'll take it",
}

def _is_single_slot_confirm(speech: str) -> bool:
    text = speech.lower().strip().rstrip(".,!?")
    if text in _SINGLE_SLOT_CONFIRM:
        return True
    words = set(text.split())
    meaningful = {w for w in words if len(w) > 1}
    return bool(meaningful) and meaningful.issubset(_SINGLE_SLOT_CONFIRM | {"that's", "it's"})


# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────
def _build_counselor_prompt(school, phone, student_name, parent_name,
                             dimension, risk_level, details, recommended_action):

    dim_key   = dimension.lower().replace("behaviour", "behavior").strip()
    today_str = date.today().strftime("%A, %B %d, %Y")

    risk_resolution = {
        "HIGH": f"""
MEETING RULES (HIGH risk):
- Suggest meeting as soon as possible — emit [SCHEDULE_MEETING: soonest].
  Do NOT say a specific time; the system provides real slots.
- If parent says busy: persuade ONCE warmly then IMMEDIATELY accept whatever day they give.
- EXCEPTION — far-future dates (next month, next year, few weeks): do NOT accept.
  Say: "I really understand you're busy, but this is quite serious. Please call us at {phone}."
  Then use [CONTINUE].
- Do NOT accept monitoring alone as the outcome for a high-risk case.
""",
        "MEDIUM": f"""
RESOLUTION (MEDIUM risk):
- Reassure the parent that you will monitor the situation closely.
- Offer a meeting only if the parent wants one — do not push.
""",
        "LOW": f"""
RESOLUTION (LOW risk):
- End on a genuinely positive note. No meeting needed.
""",
    }.get(risk_level.upper(), "Suggest meeting if helpful.")

    dimension_context = {
        "attendance": f"""
CONCERN: Attendance
{student_name}'s attendance is low and they risk not being allowed to sit exams.
Details (USE ONLY THESE FACTS): {details}

Natural flow:
- After they confirm they're free, introduce the concern in 1-2 sentences.
- CRITICAL: First concern message must ALWAYS end with an open question.
- Listen fully before suggesting next steps.

{risk_resolution}
""",
        "performance": f"""
CONCERN: Academic Performance
{student_name} is struggling with grades.
Details (USE ONLY THESE FACTS): {details}

Natural flow:
- Introduce concern in 1-2 sentences, always end with an open question.

{risk_resolution}
""",
        "behavior": f"""
CONCERN: Behaviour
{student_name} has had behavioural incidents.
Details (USE ONLY THESE FACTS): {details}

Natural flow:
- Gently introduce concern, always end with an open question.

{risk_resolution}
""",
    }.get(dim_key, f"CONCERN: {dimension.upper()}\nDetails: {details}\n{risk_resolution}")

    if risk_level.upper() == "HIGH":
        scheduling_section = f"""
MEETING SCHEDULING:
- When a parent agrees to a meeting OR asks to reschedule, say:
  "Please wait for a moment, I need to check my schedule."
  Then on a NEW LINE emit ONLY this tag:
  [SCHEDULE_MEETING: <day_preference>]
  Where <day_preference> is ONE of:
    - "soonest"        — parent said yes/as soon as possible
    - "tomorrow"       — parent explicitly said tomorrow
    - "today"          — parent explicitly said today
    - "next week"      — parent said next week
    - "<day name>"     — parent named a specific day e.g. "monday"
    - "next_available" — parent rejected a time but gave no specific day
- Tag must be on its OWN line. NEVER inline.
- NEVER invent times. The SYSTEM provides real slots.
- NEVER emit [SCHEDULE_MEETING:...] if a meeting is already confirmed in the conversation.
  If asked "what time?", repeat the confirmed time from earlier in the conversation.
- This rule applies at ALL stages including farewell.
"""
    else:
        scheduling_section = f"""
MEETING SCHEDULING (DISABLED):
- NEVER use [SCHEDULE_MEETING:...].
- If parent asks to meet, say: "Please call us at {phone} and we'll arrange a time."
"""

    return f"""You are Priya — a warm, experienced school counselor calling from {school}.
You are speaking with {parent_name}, parent of {student_name}.
School phone: {phone}
Office hours: Monday to Friday, 9 AM to 4 PM

TODAY'S DATE: {today_str}
- Use this exact date if the parent asks what day it is.
- NEVER guess or make up the date.

YOUR PERSONALITY:
- Warm, calm, genuinely caring. Sound like a real human.
- Clear, natural English only. No Hindi/regional words.
- 2 to 3 sentences per reply maximum.
- Use contractions: I'll, we'll, that's, it's, don't.

STRICT CALL FLOW:
1. Parent confirms identity → ONE sentence: acknowledge warmly + introduce yourself as Priya.
2. Same reply: ask if it's a good time. Wait.
   - YES → step 3  |  NO/busy → apologise, offer callback, goodbye [END_CALL]
3. Introduce concern (1-2 sentences)
4. Ask ONE open question, listen
5. Continue based on what they said
6. Suggest next steps / meeting
7. Answer questions
8. Close warmly

{dimension_context}

{scheduling_section}

EMOTIONAL INTELLIGENCE:
- If parent shares something sad or difficult, acknowledge with empathy FIRST.
- NEVER say "I'm glad" or "That's great" when parent said something difficult.

WHEN PARENT SAYS "BUSY":
- Treat as "I cannot do that time." Do NOT ask why.
- Say one warm sentence then emit [SCHEDULE_MEETING: next_available].

CLOSING:
- NEVER use [END_CALL] on your own.
- [SYSTEM: ask about more questions] → ask if they have other questions, then [CONTINUE].
- [SYSTEM: end call now] → ONE warm goodbye with hours and phone, then [END_CALL].
- Never output [SYSTEM:...] tags aloud.

STRICT RULES:
1. Never greet again after the first message.
2. Never re-introduce yourself mid-call.
3. Only discuss {dimension.upper()}.
4. 2-3 sentences max per reply.
5. English only.
6. Never invent details not in the concern above.
7. Never invent times — always use [SCHEDULE_MEETING:...].
8. NEVER emit [SCHEDULE_MEETING:...] once a meeting is already confirmed.
9. End EVERY reply with one tag on its own line:
   [CONTINUE] | [END_CALL] | [SCHEDULE_MEETING: <day>]
""".strip()


# ─────────────────────────────────────────────────────────────────
#  CONVERSATION STATE
# ─────────────────────────────────────────────────────────────────
class ConversationState:
    def __init__(self, payload: CallPayload, school: str, phone: str):
        self.payload               = payload
        self.school                = school
        self.phone                 = phone
        self.ended                 = False
        self.turn_count            = 0
        self.stage                 = STAGE_INTRO
        self.messages: list[dict]  = []
        self.meeting_pending       = False
        self.pending_slots: list[dict] = []
        self.awaiting_slot_choice  = False
        self.last_booked_slot: dict | None = None
        self.call_id               = f"PENDING_{payload.registration}"
        self.db_turn_number        = 0

        self.system_prompt = _build_counselor_prompt(
            school=school, phone=phone,
            student_name=payload.student_name, parent_name=payload.parent_name,
            dimension=payload.dimension, risk_level=payload.risk_level,
            details=payload.details,
            recommended_action=getattr(payload, "recommended_action", ""),
        )

    def advance_stage(self):
        if self.stage in (STAGE_FAREWELL, STAGE_CLOSING):
            return
        if   self.stage == STAGE_INTRO        and self.turn_count >= 1: self.stage = STAGE_AVAILABILITY
        elif self.stage == STAGE_AVAILABILITY and self.turn_count >= 2: self.stage = STAGE_CONVERSATION
        elif self.stage == STAGE_CONVERSATION and self.turn_count >= 5: self.stage = STAGE_SOLUTION

    def log_turn(self, role: str, message: str) -> None:
        if not isinstance(self.call_id, int):
            return
        # Run DB write in a background thread so it never adds latency to the
        # Twilio response. The turn_order counter is captured by value here.
        import threading
        from services.database import save_turn
        turn_num = self.db_turn_number
        self.db_turn_number += 1
        call_id  = self.call_id
        def _write():
            try:
                save_turn(call_id=call_id, role=role,
                          content=message, turn_order=turn_num)
            except Exception as e:
                logger.error(f"[DB] log_turn failed: {e}")
        threading.Thread(target=_write, daemon=True).start()


# ─────────────────────────────────────────────────────────────────
#  TAG PARSERS & HELPERS
# ─────────────────────────────────────────────────────────────────
_SCHEDULE_TAG_RE     = re.compile(r'\[SCHEDULE_MEETING[^\]]*\]', re.IGNORECASE)
_SCHEDULE_TAG_CAP_RE = re.compile(r'\[SCHEDULE_MEETING:\s*([^\]]+)\]', re.IGNORECASE)

def _extract_schedule_tag(ai_text: str) -> tuple[str, str | None]:
    match = _SCHEDULE_TAG_CAP_RE.search(ai_text)
    if not match:
        return _SCHEDULE_TAG_RE.sub("", ai_text).strip(), None
    day_pref = match.group(1).strip().lower()
    cleaned  = _SCHEDULE_TAG_RE.sub("", ai_text).strip()
    cleaned  = re.sub(r'\bscheduling[_\s]meeting\b', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned, day_pref

_MEETING_PROPOSAL_KEYWORDS = ("meet", "come in", "schedule", "arrange", "in person", "appointment")

_SIMPLE_AFFIRMATIVE = {
    "yes","yes!","yes?","yeah","yep","yup","sure","okay","ok",
    "of course","available","i'm available","i am available","why not",
    "can do","works","that's fine","that works","alright","will do",
    "no problem","sounds good","happy to","i'll come","we'll come",
    "please","go ahead","let's do it","sure thing",
    "absolutely","certainly","definitely","by all means",
}

def _last_ai_proposed_meeting(state: "ConversationState") -> bool:
    last_ai = next(
        (m["content"] for m in reversed(state.messages) if m["role"] == "assistant"), ""
    ).lower()
    return any(kw in last_ai for kw in _MEETING_PROPOSAL_KEYWORDS)

def _is_simple_affirmative(speech: str) -> bool:
    return speech.lower().strip().rstrip("?,!.") in _SIMPLE_AFFIRMATIVE

def _format_time_spoken(t: str) -> str:
    try:
        h, m   = map(int, t.split(":"))
        period = "AM" if h < 12 else "PM"
        h12    = h % 12 or 12
        return f"{h12} {period}" if m == 0 else f"{h12}:{m:02d} {period}"
    except Exception:
        return t

def _sanitise_spoken(text: str) -> str:
    text = _SCHEDULE_TAG_RE.sub("", text)
    for pat in (r'\[CONTINUE\]', r'\[END_CALL\]', r'\[SYSTEM:[^\]]*\]',
                r'\bscheduling[_\s]meeting\b', r'\[SCHEDULE_MEETING[^\]]*$',
                r'\[SCHEDULE_[^\]]*$'):
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
#  SCHEDULE HELPERS
# ─────────────────────────────────────────────────────────────────
def _resolve_slots_for_day(day_name: str, force_next_week: bool = False) -> list[dict]:
    from services.schedule_manager import ScheduleManager, DAY_INDEX, _next_weeks_monday
    sm    = ScheduleManager()
    today = date.today()
    now_str = datetime.now().strftime("%H:%M")

    target_weekday = DAY_INDEX.get(day_name.lower())
    if target_weekday is None:
        return []

    if target_weekday == today.weekday() and not force_next_week:
        use_next = today >= _next_weeks_monday()
        raw = sm.get_available_slots_for_day(day_name.capitalize(), next_week=use_next)
        today_slots = [s for s in raw if s["start_time"] > now_str]
        if today_slots:
            return [{**s, "date": today.isoformat(), "use_next_week": use_next}
                    for s in today_slots]

    days_ahead = (target_weekday - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    target_date = today + timedelta(days=days_ahead)

    if force_next_week and target_date < _next_weeks_monday():
        target_date += timedelta(weeks=1)

    use_next  = target_date >= _next_weeks_monday()
    raw_slots = sm.get_available_slots_for_day(day_name.capitalize(), next_week=use_next)
    return [{**s, "date": target_date.isoformat(), "use_next_week": use_next}
            for s in raw_slots]


def _resolve_slots_for_day_or_next(day_pref: str) -> list[dict]:
    from services.schedule_manager import ScheduleManager, DAY_INDEX, _next_weeks_monday
    sm    = ScheduleManager()
    today = date.today()
    now_str = datetime.now().strftime("%H:%M")

    if "next week" in day_pref:
        nw_monday = _next_weeks_monday()
        for offset in range(7):
            candidate = nw_monday + timedelta(days=offset)
            if candidate.weekday() == 6:
                continue
            day_name = candidate.strftime("%A")
            slots = sm.get_available_slots_for_day(day_name, next_week=True)
            if slots:
                return [{**s, "date": candidate.isoformat(), "use_next_week": True} for s in slots]
        return []

    for day_name in DAY_INDEX:
        if day_name in day_pref:
            return _resolve_slots_for_day(day_name, force_next_week="next week" in day_pref)

    start_offset = 1 if "tomorrow" in day_pref else 0
    for offset in range(start_offset, 14):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() == 6:
            continue
        day_name = candidate.strftime("%A")
        use_next = candidate >= _next_weeks_monday()
        slots    = sm.get_available_slots_for_day(day_name, next_week=use_next)
        if candidate == today:
            slots = [s for s in slots if s["start_time"] > now_str]
        if slots:
            return [{**s, "date": candidate.isoformat(), "use_next_week": use_next} for s in slots]
    return []


def _resolve_next_available(state: "ConversationState") -> tuple[list[dict], str]:
    from services.schedule_manager import ScheduleManager, _next_weeks_monday
    sm    = ScheduleManager()
    today = date.today()

    if state.last_booked_slot:
        try:
            start_date = date.fromisoformat(state.last_booked_slot["date"])
        except (KeyError, ValueError):
            start_date = today + timedelta(days=1)
        excluded_time = state.last_booked_slot.get("start_time")
        excluded_day  = state.last_booked_slot.get("day", "").lower()
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
        slots    = sm.get_available_slots_for_day(day_name, next_week=use_next)
        if excluded_day and day_name.lower() == excluded_day and excluded_time:
            slots = [s for s in slots if s["start_time"] != excluded_time]
        if slots:
            return (
                [{**s, "date": candidate.isoformat(), "use_next_week": use_next} for s in slots],
                day_name,
            )
    return [], ""


def _try_next_day_slots(after_day: str, after_date: str) -> tuple[list[dict], str]:
    from services.schedule_manager import ScheduleManager, _next_weeks_monday
    sm    = ScheduleManager()
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
        slots    = sm.get_available_slots_for_day(day_name, next_week=use_next)
        if slots:
            return (
                [{**s, "date": candidate.isoformat(), "use_next_week": use_next} for s in slots],
                day_name,
            )
    return [], ""


# ─────────────────────────────────────────────────────────────────
#  SLOT BOOKING / CANCELLATION
#  FIX: All book_slot and cancel_slot calls now pass registration
#       and date_iso so the DB query targets the correct row.
# ─────────────────────────────────────────────────────────────────
def _book_csv_slot(slot: dict, state: "ConversationState") -> bool:
    from services.schedule_manager import ScheduleManager
    sm = ScheduleManager()
    registration = state.payload.registration

    # Cancel any existing booking first
    if state.last_booked_slot:
        sm.cancel_slot(
            state.last_booked_slot["day"],
            state.last_booked_slot["start_time"],
            registration,                                        # FIX: was missing
            date_iso=state.last_booked_slot.get("date"),        # FIX: was missing
            next_week=state.last_booked_slot.get("use_next_week", False),
        )
        state.last_booked_slot = None

    booked = sm.book_slot(
        slot["day"],
        slot["start_time"],
        registration,                                            # FIX: was missing
        date_iso=slot.get("date"),                              # FIX: was missing
        next_week=slot.get("use_next_week", False),
    )
    if booked:
        state.last_booked_slot = slot
    return booked


def _cancel_last_booking(state: "ConversationState") -> None:
    """Helper to cancel the current booking cleanly in one place."""
    if not state.last_booked_slot:
        return
    from services.schedule_manager import ScheduleManager
    ScheduleManager().cancel_slot(
        state.last_booked_slot["day"],
        state.last_booked_slot["start_time"],
        state.payload.registration,                              # FIX: was missing
        date_iso=state.last_booked_slot.get("date"),            # FIX: was missing
        next_week=state.last_booked_slot.get("use_next_week", False),
    )
    state.last_booked_slot = None


# ─────────────────────────────────────────────────────────────────
#  BACKGROUND SUMMARY
# ─────────────────────────────────────────────────────────────────
def _trigger_summary(state, groq_api_key: str) -> None:
    try:
        from groq import Groq
        from services.summary_agent import SummaryAgent
        SummaryAgent(Groq(api_key=groq_api_key)).generate_and_save(
            call_id=state.call_id,
            payload=state.payload,
            last_booked_slot=state.last_booked_slot,
        )
    except Exception as e:
        logger.error(f"[Summary] Failed: {e}")


# ─────────────────────────────────────────────────────────────────
#  SLOT TIME MATCHER
# ─────────────────────────────────────────────────────────────────
_NEGATION_RE = re.compile(
    r"\b(not|no|never|can't|cant|cannot|won't|wont|don't|dont)\b", re.IGNORECASE)

def _prefix_is_negated(speech_lower: str, match_pos: int) -> bool:
    prefix = speech_lower[:match_pos]
    for sep in (",", ";", ".", " but ", " and "):
        last = prefix.rfind(sep)
        if last >= 0:
            prefix = prefix[last + len(sep):]
    prefix_words = prefix.split()
    recent = " ".join(prefix_words[-2:]) if prefix_words else ""
    return bool(_NEGATION_RE.search(recent))


def _match_slot_in_speech(speech: str, slots: list[dict]) -> dict | None:
    speech_lower = speech.lower().strip()

    if len(slots) == 1 and _is_single_slot_confirm(speech):
        return slots[0]

    speech_nsp = speech_lower.replace(" ", "").replace(".", "").replace(":", "")

    _neg_token_re = re.compile(
        r'\b(?:not|no|never|can\'t|cant|cannot|won\'t|wont|don\'t|dont)\s+(\d+)', re.IGNORECASE)
    _negated_digits = set()
    for m in _neg_token_re.finditer(speech_lower):
        _negated_digits.update(m.group(1))
    _effective_digit_count = sum(
        1 for c in speech_lower if c.isdigit() and c not in _negated_digits)

    for slot in slots:
        st         = slot["start_time"]
        spoken_fmt = _format_time_spoken(st).lower()
        spoken_nsp = spoken_fmt.replace(" ", "").replace(":", "")
        h, m       = map(int, st.split(":"))
        h12        = h % 12 or 12

        pos = speech_lower.find(spoken_fmt)
        if pos >= 0:
            if _prefix_is_negated(speech_lower, pos):
                continue
            return slot

        nsp_pos = speech_nsp.find(spoken_nsp)
        if nsp_pos >= 0 and not _prefix_is_negated(speech_lower, speech_lower.find(str(h12))):
            return slot

        raw24  = st.replace(":", "")
        pos24  = speech_nsp.find(raw24)
        if pos24 >= 0:
            approx = speech_lower.find(raw24[0])
            if not _prefix_is_negated(speech_lower, approx if approx >= 0 else pos24):
                return slot

        raw12  = f"{h12}{m:02d}"
        pos12  = speech_nsp.find(raw12)
        if pos12 >= 0:
            approx = speech_lower.find(str(h12))
            if not _prefix_is_negated(speech_lower, approx if approx >= 0 else pos12):
                return slot

        h12_colon = f"{h12}:{m:02d}"
        posc = speech_lower.find(h12_colon)
        if posc >= 0 and not _prefix_is_negated(speech_lower, posc):
            return slot

        hour12_str = str(h12)
        if ":" not in speech_lower and _effective_digit_count <= len(hour12_str):
            padded = f" {speech_lower} "
            posh   = padded.find(f" {hour12_str} ")
            if posh < 0 and speech_lower.startswith(hour12_str + " "):
                posh = 0
            if posh < 0 and speech_lower == hour12_str:
                posh = 0
            if posh >= 0 and not _prefix_is_negated(padded, posh):
                return slot

    return None


# ─────────────────────────────────────────────────────────────────
#  MAIN SERVICE
# ─────────────────────────────────────────────────────────────────
class TwoWayAIVoiceService:
    MODEL       = "llama-3.1-8b-instant"
    MAX_HISTORY = 8   # was 12; fewer = lower latency

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
                    model=self.MODEL, messages=[{"role":"user","content":"Hi"}], max_tokens=10)
                print("[OK] Groq Pre-warmed")
            except Exception:
                pass
        else:
            print("[WARN] GROQ_API_KEY not set — Demo mode active")

        self._conversations: dict[str, ConversationState] = {}
        self._call_sid_map:  dict[str, str] = {}
        self._phone_to_reg:  dict[str, str] = {}
        self.calls_made = []

    def _ask_groq(self, state: ConversationState, parent_speech: str) -> str:
        state.log_turn("user", parent_speech)
        state.messages.append({"role": "user", "content": parent_speech})
        state.turn_count += 1
        state.advance_stage()
        try:
            r = self._groq.chat.completions.create(
                model=self.MODEL,
                messages=[{"role":"system","content":state.system_prompt}]
                         + state.messages[-self.MAX_HISTORY:],
                temperature=0.3, max_tokens=80,
            )
            ai = r.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            ai = f"I'm sorry, there's a technical issue. Please call us at {self._phone}. Goodbye.\n[END_CALL]"
        state.messages.append({"role":"assistant","content":ai})
        state.log_turn("assistant", _sanitise_spoken(ai.replace("[END_CALL]","").replace("[CONTINUE]","")))
        return ai

    def _parse_reply(self, ai_text: str) -> tuple[str, bool]:
        return ai_text.replace("[END_CALL]","").replace("[CONTINUE]","").strip(), "[END_CALL]" in ai_text

    def _twiml(self, spoken: str, end_call: bool, state: ConversationState | None = None) -> str:
        safe    = html.escape(spoken)
        safe_ph = html.escape(self._phone)
        wh      = f"{self._ngrok_url}/handle-parent-response"
        if end_call:
            if state and self._groq_key:
                threading.Thread(target=_trigger_summary, args=(state, self._groq_key), daemon=True).start()
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">{safe}</Say>
    <Pause length="1"/><Hangup/>
</Response>"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN" action="{wh}" method="POST"
            timeout="8" speechTimeout="auto" bargeIn="true">
        <Say voice="Polly.Aditi" language="en-IN">{safe}</Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">I didn't catch that. Please call us at {safe_ph}. Goodbye.</Say>
</Response>"""

    def _reprompt_twiml(self, msg: str) -> str:
        safe = html.escape(msg)
        wh   = f"{self._ngrok_url}/handle-parent-response"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN" action="{wh}" method="POST"
            timeout="10" speechTimeout="auto" bargeIn="true">
        <Say voice="Polly.Aditi" language="en-IN">{safe}</Say>
    </Gather>
    <Redirect method="POST">{wh}</Redirect>
</Response>"""

    def _opening_twiml(self, payload: CallPayload) -> str:
        safe_p  = html.escape(payload.parent_name)
        safe_ph = html.escape(self._phone)
        wh      = f"{self._ngrok_url}/handle-parent-response"
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" language="en-IN" action="{wh}" method="POST"
            timeout="8" speechTimeout="auto">
        <Say voice="Polly.Aditi" language="en-IN">Hello! Am I speaking with {safe_p}?</Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">I didn't hear a response. Please call us at {safe_ph}. Goodbye.</Say>
</Response>"""

    def generate_followup_twiml(self, registration: str, parent_speech: str) -> str:
        state   = self._conversations.get(registration)
        safe_ph = html.escape(self._phone)

        if not state:
            return f'<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="Polly.Aditi" language="en-IN">Thank you. Please contact us at {safe_ph}. Goodbye.</Say></Response>'
        if state.ended:
            return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
        if not self._ai_ready:
            return f'<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="Polly.Aditi" language="en-IN">Thank you. Please contact the college at {safe_ph}. Goodbye.</Say></Response>'

        if _is_recording_notice_fragment(parent_speech):
            return self._reprompt_twiml("Hello! Am I speaking with " + html.escape(state.payload.parent_name) + "?")

        if state.turn_count < 2 and _is_too_short_to_process(parent_speech):
            return self._reprompt_twiml("I'm sorry, I didn't quite catch that. Could you please say that again?")

        if state.awaiting_slot_choice and state.pending_slots:
            return self._handle_slot_choice(state, parent_speech)

        # Meeting-agreement shortcut
        if (state.payload.risk_level.upper() == "HIGH"
                and state.stage in (STAGE_AVAILABILITY, STAGE_CONVERSATION, STAGE_SOLUTION)
                and not state.meeting_pending
                and _is_simple_affirmative(parent_speech)
                and _last_ai_proposed_meeting(state)):
            slots = _resolve_slots_for_day_or_next("soonest")
            if slots:
                slot = slots[0]
                if _book_csv_slot(slot, state):  # FIX: _book_csv_slot now passes registration
                    slot_str = f"{slot['day']} at {_format_time_spoken(slot['start_time'])}"
                    spoken   = (f"That's great! I've booked you in for {slot_str}. "
                                f"We look forward to seeing you then. "
                                f"Is there anything else you'd like to discuss?")
                    state.log_turn("user", parent_speech)
                    state.log_turn("assistant", spoken)
                    state.messages += [{"role":"user","content":parent_speech},
                                       {"role":"assistant","content":spoken}]
                    state.turn_count += 1
                    state.stage = STAGE_FAREWELL
                    return self._twiml(spoken, False, state)

        # Closing detection
        if _parent_wants_to_end(parent_speech, state.stage):
            if state.turn_count >= 4:
                if state.stage == STAGE_FAREWELL:
                    speech_for_ai = parent_speech + " [SYSTEM: end call now - mention school hours and phone number]"
                else:
                    state.stage   = STAGE_FAREWELL
                    speech_for_ai = parent_speech + " [SYSTEM: ask about more questions]"
            else:
                speech_for_ai = parent_speech
        else:
            speech_for_ai = parent_speech

        ai_text = self._ask_groq(state, speech_for_ai)

        pre, day_pref = _extract_schedule_tag(ai_text)
        pre = _sanitise_spoken(pre)

        # Guard: ignore scheduling tags if meeting already booked
        if day_pref is not None and state.last_booked_slot is not None:
            print(f"   [GUARD] Ignoring [SCHEDULE_MEETING:{day_pref}] — slot already booked: "
                  f"{state.last_booked_slot['day']} {state.last_booked_slot['start_time']}")
            day_pref = None

        # Suppress scheduling for MEDIUM/LOW
        if day_pref is not None and state.payload.risk_level.upper() != "HIGH":
            spoken, ec = self._parse_reply(_sanitise_spoken(pre))
            if ec: state.ended = True
            return self._twiml(spoken, ec, state)

        # Far-future rejection
        _FAR_FUTURE = ("next year", "year", "few weeks", "some time")
        if (day_pref is not None
                and any(kw in day_pref for kw in _FAR_FUTURE)
                and "next week" not in day_pref):
            spoken = (f"I completely understand you're very busy. But this situation is quite serious "
                      f"and waiting that long could really impact your child's future. "
                      f"Please call us at {self._phone} as soon as possible — we'll find a time.")
            state.stage = STAGE_SOLUTION
            return self._twiml(_sanitise_spoken(spoken), False, state)

        if day_pref is not None:
            from services.schedule_manager import DAY_INDEX
            named_day      = None
            want_next_week = "next week" in day_pref
            if day_pref not in ("soonest", "tomorrow", "today", "yes"):
                for d in DAY_INDEX:
                    if d in day_pref:
                        named_day = d
                        break

            if "next_available" in day_pref:
                _cancel_last_booking(state)  # FIX: use helper with registration
                slots, day_name = _resolve_next_available(state)
                if slots:
                    state.pending_slots = slots; state.awaiting_slot_choice = True
                    state.stage = STAGE_SOLUTION
                    times  = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                    spoken = (f"{pre} I've checked and on {day_name} we have "
                              f"slots at {times}. Which time works best for you?").strip()
                    return self._twiml(spoken, False, state)
                else:
                    spoken = (f"{pre} I'm afraid we don't have any free slots in the "
                              f"next few days. Please call us at {self._phone}.").strip()
                    return self._twiml(spoken, False, state)

            if named_day:
                _cancel_last_booking(state)  # FIX: use helper with registration
                slots = _resolve_slots_for_day(named_day, force_next_week=want_next_week)
                if slots:
                    state.pending_slots = slots; state.awaiting_slot_choice = True
                    state.stage = STAGE_SOLUTION
                    times      = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                    week_label = " next week" if want_next_week else ""
                    spoken = (f"{pre} I've checked and on {named_day.capitalize()}{week_label} "
                              f"we have slots at {times}. Which time works best for you?").strip()
                    return self._twiml(spoken, False, state)
                else:
                    week_label = " next week" if want_next_week else ""
                    spoken = (f"{pre} I'm sorry, we don't have any free slots on "
                              f"{named_day.capitalize()}{week_label}. Would another day work?").strip()
                    return self._twiml(spoken, False, state)

            else:
                # soonest / tomorrow / today
                is_today = any(kw in day_pref for kw in ("today", "right now", "now", "immediately"))
                slots = _resolve_slots_for_day_or_next(day_pref)
                if slots:
                    _cancel_last_booking(state)  # FIX: use helper with registration
                    state.pending_slots = slots; state.awaiting_slot_choice = True
                    state.stage = STAGE_SOLUTION
                    day_label = slots[0]["day"]
                    times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                    spoken = (f"{pre} I have slots on {day_label} at {times}. "
                              f"Which time works best for you?").strip()
                    return self._twiml(spoken, False, state)
                else:
                    spoken = (
                        f"{pre} I'm afraid all of today's slots have passed or been filled. "
                        f"Could we look at another day, or call us at {self._phone}?"
                        if is_today else
                        f"{pre} I'm afraid we don't have any free slots in the next few days. "
                        f"Please call us at {self._phone}."
                    ).strip()
                    return self._twiml(spoken, False, state)

        # Normal flow
        spoken, end_call = self._parse_reply(ai_text)
        spoken = _sanitise_spoken(spoken)
        if end_call:
            state.ended = True

        print(f"  [P] Parent : \"{parent_speech}\"")
        print(f"  [AI] Priya  : \"{spoken[:90]}{'...' if len(spoken)>90 else ''}\"")
        print(f"  [INFO] Turn: {state.turn_count} | Stage: {state.stage} | End: {end_call}")

        return self._twiml(spoken, end_call, state)

    def _handle_slot_choice(self, state: ConversationState, parent_speech: str) -> str:
        speech_lower = parent_speech.lower().strip()
        from services.schedule_manager import DAY_INDEX
        current_day = state.pending_slots[0]["day"].lower() if state.pending_slots else ""

        # Today / tomorrow redirect
        for kw, pref in (("today", "today"), ("tomorrow", "tomorrow")):
            if (kw in speech_lower and f"not {kw}" not in speech_lower
                    and not _is_single_slot_confirm(parent_speech)
                    and not _match_slot_in_speech(parent_speech, state.pending_slots)):
                _cancel_last_booking(state)  # FIX: use helper with registration
                state.pending_slots = []; state.awaiting_slot_choice = False
                slots = _resolve_slots_for_day_or_next(pref)
                if slots:
                    state.pending_slots = slots; state.awaiting_slot_choice = True
                    day_label = slots[0]["day"]
                    times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                    label = "today" if pref == "today" else f"tomorrow, {day_label}"
                    return self._twiml(f"For {label} we have slots at {times}. Which time works best?",
                                       False, state)
                else:
                    alt = "tomorrow" if pref == "today" else "another day"
                    return self._twiml(f"I'm sorry, there are no slots for {pref}. Would {alt} work?",
                                       False, state)

        # Named different day
        requested_day = None
        for d in DAY_INDEX:
            if d in speech_lower:
                requested_day = d
                break

        if requested_day and requested_day == current_day and not _is_single_slot_confirm(parent_speech) and not _match_slot_in_speech(parent_speech, state.pending_slots):
            times = ", ".join(_format_time_spoken(s["start_time"]) for s in state.pending_slots)
            return self._twiml(f"For {requested_day.capitalize()} we have slots at {times}. Which time works best?",
                               False, state)

        if requested_day and requested_day != current_day:
            _cancel_last_booking(state)  # FIX: use helper with registration
            state.pending_slots = []; state.awaiting_slot_choice = False
            force_nw = "next week" in speech_lower
            slots = _resolve_slots_for_day(requested_day, force_next_week=force_nw)
            if slots:
                state.pending_slots = slots; state.awaiting_slot_choice = True
                state.stage = STAGE_SOLUTION
                times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                wl = " next week" if force_nw else ""
                return self._twiml(f"Of course! On {requested_day.capitalize()}{wl} we have slots at {times}. Which time works best?",
                                   False, state)
            else:
                return self._twiml(f"I'm sorry, we don't have any free slots on {requested_day.capitalize()}. Would another day work?",
                                   False, state)

        # Next week shift
        _already_next_week = bool(state.pending_slots and state.pending_slots[0].get("use_next_week", False))
        if ("next week" in speech_lower and not _already_next_week
                and not _match_slot_in_speech(parent_speech, state.pending_slots)):
            _cancel_last_booking(state)  # FIX: use helper with registration
            state.pending_slots = []; state.awaiting_slot_choice = False
            slots = _resolve_slots_for_day_or_next("next week")
            if slots:
                state.pending_slots = slots; state.awaiting_slot_choice = True
                state.stage = STAGE_SOLUTION
                day_label = slots[0]["day"]
                times = ", ".join(_format_time_spoken(s["start_time"]) for s in slots)
                return self._twiml(f"Of course! For next week I have slots on {day_label} at {times}. Which time works best?",
                                   False, state)
            else:
                return self._twiml(f"I'm sorry, we don't have any free slots next week. Please call us at {self._phone}.",
                                   False, state)

        # Try to match a specific time
        matched = _match_slot_in_speech(parent_speech, state.pending_slots)
        if matched:
            booked = _book_csv_slot(matched, state)  # FIX: now passes registration via state
            state.pending_slots = []; state.awaiting_slot_choice = False
            if booked:
                slot_str = f"{matched['day']} at {_format_time_spoken(matched['start_time'])}"
                state.stage = STAGE_FAREWELL
                spoken = (f"Perfect, I've booked you in for {slot_str}. "
                          f"We look forward to seeing you then. "
                          f"Is there anything else you'd like to discuss before we wrap up?")
                state.messages += [{"role":"user","content":parent_speech},
                                   {"role":"assistant","content":spoken}]
                try:
                    state.log_turn("user", parent_speech)
                    state.log_turn("assistant", spoken)
                except Exception as e:
                    logger.warning(f"[DB] log_turn skipped: {e}")
                state.turn_count += 1
                return self._twiml(spoken, False, state)
            else:
                remaining = _resolve_slots_for_day(matched["day"].lower())
                if remaining:
                    state.pending_slots = remaining; state.awaiting_slot_choice = True
                    times = ", ".join(_format_time_spoken(s["start_time"]) for s in remaining)
                    return self._twiml(f"I'm sorry, that slot was just taken. Remaining on {matched['day']}: {times}. Which would you prefer?",
                                       False, state)
                else:
                    state.awaiting_slot_choice = False
                    return self._twiml(f"I'm sorry, all slots on {matched['day']} have just been filled. Please call us at {self._phone}.",
                                       False, state)

        # Decline check
        _decline_word_re = re.compile(r'\b(no|none|cancel)\b', re.IGNORECASE)
        _decline_phrases = ["different day","another day","forget it","never mind",
                            "not available","won't work","doesn't work"]
        _is_decline = (bool(_decline_word_re.search(speech_lower))
                       or any(sig in speech_lower for sig in _decline_phrases))
        if (_is_decline and not _is_single_slot_confirm(parent_speech)
                and not requested_day and not any(c.isdigit() for c in speech_lower)):
            rejected_day  = state.pending_slots[0]["day"]         if state.pending_slots else ""
            rejected_date = state.pending_slots[0].get("date","") if state.pending_slots else ""
            state.pending_slots = []; state.awaiting_slot_choice = False
            next_slots, next_day = _try_next_day_slots(rejected_day, rejected_date)
            if next_slots:
                state.pending_slots = next_slots; state.awaiting_slot_choice = True
                state.stage = STAGE_SOLUTION
                times = ", ".join(_format_time_spoken(s["start_time"]) for s in next_slots)
                return self._twiml(f"No problem. On {next_day} we have slots at {times}. Would any of those work?",
                                   False, state)
            else:
                state.stage = STAGE_FAREWELL
                return self._twiml(f"I understand. Unfortunately we don't have any free slots in the next few days. Please call us at {self._phone}.",
                                   False, state)

        # Fallback
        times = ", ".join(_format_time_spoken(s["start_time"]) for s in state.pending_slots)
        day   = state.pending_slots[0]["day"] if state.pending_slots else "that day"
        if _is_single_slot_confirm(parent_speech):
            return self._twiml(f"We have times on {day} at {times}. Which specific time works best?", False, state)
        if any(c.isdigit() for c in parent_speech):
            if len(state.pending_slots) == 1:
                closest = state.pending_slots[0]
                return self._twiml(f"I think there may be a small mix-up — the slot we have on {day} is at {_format_time_spoken(closest['start_time'])}. Would that work?",
                                   False, state)
            return self._twiml(f"I don't think we have a slot at that time. On {day} we have {times} — would either suit you?",
                               False, state)
        return self._twiml(f"I didn't quite catch that. The available slots on {day} are {times}. Which one works best?",
                           False, state)

    def make_call(self, payload: CallPayload) -> NotificationResult:
        if not self._twilio_ready:
            return NotificationResult(success=False, error_message="Twilio not configured.",
                                      student_id=payload.registration)
        state = ConversationState(payload, self._school, self._phone)
        self._conversations[payload.registration] = state
        self._phone_to_reg[payload.to_number]     = payload.registration
        print(f"[OK] Ready: {payload.student_name} [{payload.dimension.upper()} / {payload.risk_level}]")
        try:
            from services.database import create_call_record
            state.call_id = create_call_record(
                registration=payload.registration, student_name=payload.student_name,
                parent_name=payload.parent_name, dimension=payload.dimension,
                risk_level=payload.risk_level)
            print(f"   [DB] Call record created: db_call_id={state.call_id}")
        except Exception as e:
            logger.error(f"[DB] create_call_record failed: {e}")
        try:
            print(f"\n[CALL] Calling {payload.parent_name} ({payload.to_number})...")
            call = self._client.calls.create(
                to=payload.to_number, from_=self._from_number, twiml=self._opening_twiml(payload))
            self._call_sid_map[call.sid] = payload.registration
            self.calls_made.append(payload)
            print(f"   [OK] SID: {call.sid}\n")
            return NotificationResult(success=True, sid=call.sid,
                                      student_id=payload.registration, channel="ai_2way_groq")
        except Exception as e:
            print(f"   [ERR] {str(e)[:120]}")
            self._phone_to_reg.pop(payload.to_number, None)
            return NotificationResult(success=False, error_message=str(e),
                                      student_id=payload.registration)

    def make_batch_calls(self, payloads: list) -> dict:
        results = {"total":len(payloads),"successful":0,"failed":0,"details":[]}
        print(f"\n[AI] Groq 2-Way Calls: {len(payloads)}\n")
        for i, p in enumerate(payloads, 1):
            print(f"[{i}/{len(payloads)}] {p.student_name} [{p.dimension.upper()}]")
            r = self.make_call(p)
            results["details"].append({"registration":p.registration,"student":p.student_name,
                                       "parent":p.parent_name,"success":r.success,
                                       "sid":getattr(r,"sid",None),"channel":"ai_2way_groq"})
            if r.success: results["successful"] += 1
            else:         results["failed"] += 1
            time.sleep(2)
        print(f"\n[OK] {results['successful']}/{results['total']}\n")
        return results


class TwoWayDemoService:
    def __init__(self):
        self._school = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone  = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self.calls_made     = []
        self._conversations = {}
        self._call_sid_map  = {}
        self._phone_to_reg  = {}

    def make_call(self, payload: CallPayload) -> NotificationResult:
        self.calls_made.append(payload)
        print(f"\n{'═'*56}\n  DEMO — {payload.dimension.upper()} | {payload.student_name}\n{'═'*56}\n")
        return NotificationResult(success=True, sid=f"DEMO_{len(self.calls_made):03d}",
                                  student_id=payload.registration, channel="demo")

    def generate_followup_twiml(self, registration, parent_speech):
        return '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="Polly.Aditi" language="en-IN">Thank you. Goodbye.</Say></Response>'

    def make_batch_calls(self, payloads):
        results = {"total":len(payloads),"successful":0,"failed":0,"details":[]}
        for p in payloads:
            r = self.make_call(p)
            results["successful"] += 1
            results["details"].append({"student":p.student_name,"parent":p.parent_name,
                                       "success":True,"sid":r.sid,"channel":"demo"})
        return results