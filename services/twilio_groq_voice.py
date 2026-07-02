"""
services/twilio_groq_voice.py
──────────────────────────────
2-Way AI Voice — Groq (llama-3.3-70b) + Twilio
"""
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date as calendar_date, datetime, time as clock_time, timedelta
try:
    from groq import Groq
except ImportError:  # Calendar and demo mode can run without the AI dependency.
    Groq = None
from core.models import CallPayload, NotificationResult
from services.calendar_service import CalendarService, CalendarSlot, create_calendar_service
from services.call_history import get_call_history_repository
from services.intent_interpreter import (
    ContextualIntentInterpreter,
    ParentIntent,
    TurnUnderstanding,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  CALL STAGES  — tracked so closing signals mean different things
#  at different points in the conversation
# ─────────────────────────────────────────────────────────────────
STAGE_INTRO        = "intro"         # Just confirmed who they are
STAGE_AVAILABILITY = "availability"  # Asked if free — waiting for answer
STAGE_CONVERSATION = "conversation"  # Main discussion happening
STAGE_SOLUTION     = "solution"      # Advice / meeting suggested
STAGE_MEETING      = "meeting"       # Waiting for a verified slot choice
STAGE_RESCHEDULE   = "reschedule"    # Waiting for a replacement date or time
STAGE_FAREWELL     = "farewell"      # Asked "anything else?" — waiting for final answer
STAGE_CLOSING      = "closing"       # Wrapping up


# ─────────────────────────────────────────────────────────────────
#  CLOSING SIGNAL DETECTOR
#  Only triggers AFTER the conversation has actually happened
#  (stage = solution or closing). Early "okay" and "yes" should
#  never end the call — they are just acknowledgements.
#
#  NOTE: Hindi/mixed phrases are kept here intentionally —
#  they detect what the PARENT says, not what the AI speaks.
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
    if stage in (STAGE_SOLUTION, STAGE_FAREWELL, STAGE_CLOSING):
        if text in LATE_STAGE_CLOSING:
            return True

    return False


def _join_spoken_slots(slots: list[CalendarSlot]) -> str:
    labels = [slot.spoken() for slot in slots]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _working_days_label(days: frozenset[int]) -> str:
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected = [names[index] for index in sorted(days)]
    if selected == names[:5]:
        return "Monday through Friday"
    if len(selected) == 1:
        return f"on {selected[0]}"
    return "on " + ", ".join(selected[:-1]) + f", and {selected[-1]}"


def _declines_meeting(speech: str) -> bool:
    text = speech.lower()
    return any(phrase in text for phrase in (
        "no meeting", "don't want", "do not want", "none of those",
        "not this week", "can't meet", "cannot meet",
    ))


def _wants_to_reschedule(speech: str) -> bool:
    text = speech.lower()
    action = re.search(r"\b(?:re[- ]?schedul\w*|shift\w*|mov(?:e|ed|ing)|chang(?:e|ed|ing))\b", text)
    if action:
        return True
    if re.search(r"\b(?:instead|another\s+(?:time|slot|day)|different\s+(?:time|slot|day))\b", text):
        return True
    if re.search(r"\bschedul\w*\b.*\bmeeting\b.*\b(?:to|for)\b", text):
        return True
    return bool(re.search(
        r"\b(?:make|do)\s+(?:the\s+meeting\s+|it\s+|that\s+)?(?:for\s+|at\s+)?"
        r"(?:\d{1,2}(?::\d{2})?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        text,
    ))


def _wants_to_cancel(speech: str) -> bool:
    text = speech.lower()
    return any(phrase in text for phrase in (
        "cancel the meeting", "cancel my meeting", "cancel our meeting",
        "cancel it", "cancel that", "call it off", "call off the meeting",
        "delete the meeting", "remove the meeting", "can't attend the meeting",
        "cannot attend the meeting", "don't need the meeting", "do not need the meeting",
    ))


def _asks_current_date(speech: str) -> bool:
    text = speech.lower()
    return "date" in text and bool(re.search(
        r"\b(?:today|today'?s|current|what\s+date|which\s+date|date\s+today)\b",
        text,
    ))


def _asks_meeting_availability(speech: str) -> bool:
    text = speech.lower()
    return bool(
        re.search(r"\b(?:meeting|meetings|slot|slots|appointment)\b", text)
        and re.search(
            r"\b(?:any|available|availability|free|possible|want|need|check|loaded|"
            r"are\s+there|is\s+there|do\s+you\s+have|when\s+can)\b",
            text,
        )
    )


def _is_short_affirmative(speech: str) -> bool:
    text = re.sub(r"[^a-z\s]", " ", speech.lower())
    text = " ".join(text.split())
    return text in {
        "yes", "yes please", "sure", "okay", "ok", "that works",
        "works for me", "sounds good", "please do", "go ahead", "do it",
        "yes thank you", "sure thank you",
    }


def _expresses_time_limit(speech: str) -> bool:
    text = speech.lower()
    return bool(re.search(
        r"\b(?:quick|quickly|brief|briefly|hurry|busy|little time|not much time|"
        r"only (?:a |one |two |three )?(?:minute|minutes)|short on time)\b",
        text,
    ))


def _explicitly_requests_meeting(speech: str) -> bool:
    text = speech.lower()
    return bool(
        re.search(r"\b(?:meeting|meet|appointment|slot|slots)\b", text)
        or re.search(r"\b(?:speak|talk|discussion)\b.*\bteacher\b", text)
    )


def _asks_for_meeting_consent(speech: str) -> bool:
    text = speech.lower()
    return "?" in speech and bool(
        re.search(r"\b(?:meeting|meet)\b", text)
        or re.search(r"\b(?:speak|talk)\b.*\bteacher\b", text)
    )


def _declines_further_help(speech: str) -> bool:
    text = " ".join(re.sub(r"[^a-z\s']", " ", speech.lower()).split())
    return bool(re.search(
        r"\b(?:i )?(?:do not|don't) need (?:anything|anything else|a meeting)|"
        r"\bnothing else\b|\bno(?:,)? thanks?\b|\bthat(?:'s| is) all\b",
        text,
    ))


def _booking_change_from_temporal_reference(state, speech: str, timezone):
    if not state.booking:
        return None
    parsed = _parse_meeting_request(speech, timezone)
    if not parsed.date and not parsed.time and not parsed.range_start:
        return None
    changed_date = parsed.date and parsed.date != state.booking.start.date()
    changed_time = parsed.time and parsed.time != state.booking.start.timetz().replace(tzinfo=None)
    explicit_change = _wants_to_reschedule(speech) or "instead" in speech.lower()
    short_reference = len(speech.split()) <= 10
    changed_range = bool(parsed.range_start)
    return parsed if (changed_date or changed_time or changed_range) and (explicit_change or short_reference) else None


@dataclass(frozen=True)
class ParsedMeetingRequest:
    date: calendar_date | None = None
    time: clock_time | None = None
    range_start: calendar_date | None = None
    range_end: calendar_date | None = None


def _parse_meeting_request(speech: str, timezone, now: datetime | None = None) -> ParsedMeetingRequest:
    text = speech.lower().strip()
    today = (now or datetime.now(timezone)).astimezone(timezone).date()
    requested_date = None
    range_start = None
    range_end = None

    next_monday = today + timedelta(days=((7 - today.weekday()) or 7))
    if re.search(r"\b(?:next\s+to\s+next\s+week|week\s+after\s+next)\b", text):
        range_start = next_monday + timedelta(days=7)
        range_end = range_start + timedelta(days=6)
    elif re.search(r"\bnext\s+week\b", text):
        range_start = next_monday
        range_end = range_start + timedelta(days=6)
    elif re.search(r"\bnext\s+month\b", text):
        if today.month == 12:
            range_start = calendar_date(today.year + 1, 1, 1)
        else:
            range_start = calendar_date(today.year, today.month + 1, 1)
        following_month = (
            calendar_date(range_start.year + 1, 1, 1)
            if range_start.month == 12
            else calendar_date(range_start.year, range_start.month + 1, 1)
        )
        range_end = following_month - timedelta(days=1)

    iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    numeric_date = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", text)
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    month_names = "|".join(months)
    month_first = re.search(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b", text
    )
    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})(?:\s+(20\d{{2}}))?\b", text
    )
    scheduling_context = bool(re.search(r"\b(?:meeting|meetings|slot|slots|appointment|date)\b", text))
    day_only = None
    if scheduling_context and not any((iso, numeric_date, month_first, day_first)):
        day_only = (
            re.search(r"\b(?:on|at|for)\s+(3[01]|[12]\d)(?:st|nd|rd|th)?\b", text)
            or re.search(r"\b(?:on|for)\s+([1-9]|1[0-2])(?:st|nd|rd|th)\b", text)
            or re.search(r"\bdate\s+(?:is\s+)?([1-9]|[12]\d|3[01])\b", text)
        )

    try:
        if iso:
            requested_date = calendar_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        elif numeric_date:
            requested_date = calendar_date(
                int(numeric_date.group(3)), int(numeric_date.group(2)), int(numeric_date.group(1))
            )
        elif month_first or day_first:
            if month_first:
                month, day, year = months[month_first.group(1)], int(month_first.group(2)), month_first.group(3)
            else:
                month, day, year = months[day_first.group(2)], int(day_first.group(1)), day_first.group(3)
            requested_date = calendar_date(int(year or today.year), month, day)
            if not year and requested_date < today:
                requested_date = requested_date.replace(year=today.year + 1)
        elif day_only:
            requested_date = calendar_date(today.year, today.month, int(day_only.group(1)))
            if requested_date < today:
                if today.month == 12:
                    requested_date = calendar_date(today.year + 1, 1, int(day_only.group(1)))
                else:
                    requested_date = calendar_date(today.year, today.month + 1, int(day_only.group(1)))
        elif "day after tomorrow" in text:
            requested_date = today + timedelta(days=2)
        elif "tomorrow" in text:
            requested_date = today + timedelta(days=1)
        elif "today" in text:
            requested_date = today
        elif not range_start:
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            matches = [
                (match.start(), weekday, name)
                for weekday, name in enumerate(weekdays)
                for match in re.finditer(rf"\b{name}\b", text)
            ]
            if matches:
                _, weekday, name = matches[-1] if len(matches) > 1 else matches[0]
                distance = (weekday - today.weekday()) % 7
                if "next " + name in text and distance == 0:
                    distance = 7
                requested_date = today + timedelta(days=distance)
    except ValueError:
        requested_date = None

    time_text = text
    if day_only:
        time_text = text[:day_only.start()] + " " + text[day_only.end():]
    requested_time = None
    time_match = (
        re.search(r"\b(?:to|instead(?:\s+at)?|rather\s+at)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", time_text)
        or
        re.search(r"\b(?:at|around|from|by)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", time_text)
        or re.search(r"\b(\d{1,2})[:.](\d{2})\s*(a\.?m\.?|p\.?m\.?)?\b", time_text)
        or re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", time_text)
    )
    if time_match:
        groups = time_match.groups()
        hour = int(groups[0])
        minute = int(groups[1]) if len(groups) > 2 and groups[1] else 0
        meridiem = (groups[-1] or "").replace(".", "")
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            requested_time = clock_time(hour, minute)
    else:
        word_hours = {
            "twelve": 12, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11,
        }
        word_match = re.search(
            r"\b(?:at|around|from|by)\s+(" + "|".join(word_hours) + r")\b", time_text
        )
        if word_match:
            hour = word_hours[word_match.group(1)]
            if "pm" in text and hour < 12:
                hour += 12
            requested_time = clock_time(hour, 30 if "thirty" in text else 0)

    return ParsedMeetingRequest(requested_date, requested_time, range_start, range_end)


def _match_requested_slot(
    speech: str,
    offered: list[CalendarSlot],
    timezone,
    duration: timedelta,
) -> CalendarSlot | None:
    text = speech.lower().strip()
    ordinal_words = {"first": 0, "second": 1, "third": 2}
    for word, index in ordinal_words.items():
        if re.search(rf"\b{word}\b", text) and index < len(offered):
            return offered[index]
    numeric_choice = re.search(r"(?:option\s+([123])|\b([123])(?:st|nd|rd)\b)", text)
    if numeric_choice:
        index = int(numeric_choice.group(1) or numeric_choice.group(2)) - 1
        if index < len(offered):
            return offered[index]

    parsed = _parse_meeting_request(speech, timezone)
    requested_date = parsed.date
    hour = parsed.time.hour if parsed.time else None
    minute = parsed.time.minute if parsed.time else None

    candidates = offered
    if requested_date:
        candidates = [slot for slot in candidates if slot.start.date() == requested_date]
    if hour is not None:
        candidates = [slot for slot in candidates if slot.start.hour == hour and slot.start.minute == minute]
    if len(candidates) == 1:
        return candidates[0]

    if requested_date and hour is not None:
        start = datetime.combine(requested_date, datetime.min.time(), timezone).replace(hour=hour, minute=minute)
        return CalendarSlot(start, start + duration)
    return None


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
- Ask whether the parent would like you to check the teacher's calendar.
- If they agree, let the application offer verified openings.
- Do NOT accept monitoring as the outcome for a high-risk case.
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
- Ask whether the parent would like to meet.
- If parent says not available: persuade ONCE warmly — like a caring teacher.
  Example: "I completely understand. I just want to mention that the situation
  is quite urgent and the sooner we can meet, the better it will be for your child.
  Is there any possibility this week at all?"
- After that ONE gentle push, respect their decision.
- If they agree, use [CHECK_AVAILABILITY] and let the application offer times.
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
    }.get(risk_level.upper(), "Suggest a meeting if helpful, but never invent an available time.")

    return f"""You are Priya — a warm, experienced school counselor calling from {school}.
You are speaking with {parent_name}, parent of {student_name}.
School phone: {phone}

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
1. Parent confirms who they are → acknowledge warmly (e.g. "I'm so glad I reached you.")
2. Ask if it is a good time to talk → wait for answer
   - If YES / free → move to step 3
   - If BUSY but asks you to be quick → give a brief concern summary; do not end the call
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
- Do NOT proactively end the call or ask farewell questions on your own.
- The system will tell you exactly when to ask "anything else?" and when to say goodbye,
  by injecting a [Note: ...] instruction into the conversation.
- When you see [Note: ... ask if the parent has any other questions ... Use [CONTINUE]]:
  ask warmly in one sentence and use [CONTINUE]. Nothing else.
- When you see [Note: Parent has confirmed they have nothing more ... Use [END_CALL]]:
  give ONE warm closing sentence, mention school hours (Monday to Friday, 9 AM to 4 PM)
  and the school number {phone}, then use [END_CALL].
- Never ask "is there anything else?" unless the system note instructs you to.

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
6. NEVER invent specific details — such as team names, teacher names, club names, or
   incident specifics — that were not given to you in the concern details above.
   If a parent raises something you have no data on, say you'd like to discuss the
   full details when you meet rather than guessing.
7. End EVERY reply with one control tag on its own line:
   [CONTINUE]  — keep going
   [END_CALL]  — end now (only after farewell step 2 is complete)
8. Tag is for system only — never spoken aloud.

MEETING TOOL:
- When the parent agrees to a meeting or asks for available times, add
  [CHECK_AVAILABILITY] on its own line.
- Never state or confirm a meeting date or time yourself. Only the application
  can check the teacher's calendar and confirm a booking.
- Never claim that a meeting was rescheduled or cancelled. The application must
  update the calendar event before that change can be confirmed.
- For high-risk cases, automated slots are limited to the next two days.
- For medium-risk cases, only check meeting slots when the parent asks, and limit
  automated options to the next seven days.

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
        self.stage          = STAGE_INTRO    # starts at intro
        self.messages: list[dict] = []
        self.offered_slots: list[CalendarSlot] = []
        self.booking = None
        self.pending_meeting_date: calendar_date | None = None
        self.pending_meeting_time: clock_time | None = None
        self.pending_action: str | None = None
        self.parent_requested_meeting = False
        self.awaiting_meeting_consent = False
        self.brief_mode = False
        self.call_sid: str | None = None
        self.last_agent_message = (
            f"Hello. This is Priya calling from {school}. "
            f"Am I speaking with {payload.parent_name}?"
        )
        self.last_understanding: TurnUnderstanding | None = None
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
        # STAGE_FAREWELL and STAGE_CLOSING are set explicitly — never overwrite them here
        if self.stage in (STAGE_MEETING, STAGE_RESCHEDULE, STAGE_FAREWELL, STAGE_CLOSING):
            return
        if self.stage == STAGE_INTRO and self.turn_count >= 1:
            self.stage = STAGE_AVAILABILITY
        elif self.stage == STAGE_AVAILABILITY and self.turn_count >= 2:
            self.stage = STAGE_CONVERSATION
        elif self.stage == STAGE_CONVERSATION and self.turn_count >= 5:
            self.stage = STAGE_SOLUTION


# ─────────────────────────────────────────────────────────────────
#  MAIN SERVICE
# ─────────────────────────────────────────────────────────────────
class TwoWayAIVoiceService:
    MODEL = "llama-3.1-8b-instant"
    MAX_HISTORY = 12

    def __init__(
        self,
        calendar_service: CalendarService | None = None,
        intent_interpreter: ContextualIntentInterpreter | None = None,
    ):
        self._twilio_sid   = os.getenv("TWILIO_ACCOUNT_SID")
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self._from_number  = os.getenv("TWILIO_FROM_NUMBER")
        self._groq_key     = os.getenv("GROQ_API_KEY")
        self._ngrok_url    = os.getenv("NGROK_URL", "")
        self._school       = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone        = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self._calendar     = calendar_service or create_calendar_service()
        self._intent_interpreter = intent_interpreter
        self._call_history = get_call_history_repository()

        self._twilio_ready = all([self._twilio_sid, self._twilio_token, self._from_number])
        self._ai_ready     = bool(self._groq_key and Groq)

        if self._twilio_ready:
            from twilio.rest import Client
            self._client = Client(self._twilio_sid, self._twilio_token)
            print("[OK] Twilio Connected")

        if self._ai_ready:
            self._groq = Groq(api_key=self._groq_key, timeout=6.0, max_retries=0)
            if self._intent_interpreter is None:
                self._intent_interpreter = ContextualIntentInterpreter(self._groq, self.MODEL)
            print(f"[OK] Groq Connected  [{self.MODEL}]")
            # Pre-warm the API to reduce cold-start latency on first call
            try:
                self._groq.chat.completions.create(
                    model=self.MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                )
                print("[OK] Groq Pre-warmed")
            except Exception as e:
                pass  # Warmup is optional
        else:
            print("[WARN]  GROQ_API_KEY not set — Demo mode active")

        self._conversations: dict[str, ConversationState] = {}
        self.calls_made = []

    # ── Ask Groq ──────────────────────────────────────────────────
    def _ask_groq(self, state: ConversationState, parent_speech: str) -> str:
        state.messages.append({"role": "user", "content": parent_speech})
        state.turn_count += 1
        self._record_turn(state, "parent", parent_speech)
        state.advance_stage()

        try:
            # Cap message history to last 12 messages to limit input tokens
            recent = state.messages[-self.MAX_HISTORY:]
            system_messages = [{"role": "system", "content": state.system_prompt}]
            if state.brief_mode:
                system_messages.append({
                    "role": "system",
                    "content": (
                        "BRIEF MODE IS ACTIVE. The parent has very little time. Reply in one short "
                        "sentence only. Do not ask another exploratory question. Acknowledge what "
                        "they said and move directly to the appropriate next step or resolution."
                    ),
                })
            response = self._groq.chat.completions.create(
                model=self.MODEL,
                messages=system_messages + recent,
                temperature=0.3 if state.brief_mode else 0.5,
                max_tokens=45 if state.brief_mode else 80,
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

    def _interpret_parent_turn(
        self, state: ConversationState, parent_speech: str
    ) -> TurnUnderstanding:
        if getattr(self, "_intent_interpreter", None) is None:
            return TurnUnderstanding.unclear("No contextual interpreter is configured.")
        booking = None
        if state.booking:
            booking = {
                "event_id": state.booking.event_id,
                "date": state.booking.start.date().isoformat(),
                "start_time": state.booking.start.strftime("%H:%M"),
                "end_time": state.booking.end.strftime("%H:%M"),
            }
        context = {
            "stage": state.stage,
            "last_agent_message": state.last_agent_message,
            "active_booking": booking,
            "offered_slots": [
                {
                    "option": index,
                    "date": slot.start.date().isoformat(),
                    "time": slot.start.strftime("%H:%M"),
                }
                for index, slot in enumerate(state.offered_slots, 1)
            ],
            "pending_action": state.pending_action,
            "risk_level": state.payload.risk_level.upper(),
            "concern_dimension": state.payload.dimension,
            "student_name": state.payload.student_name,
            "parent_name": state.payload.parent_name,
            "concern_details": state.payload.details,
            "recommended_action": getattr(state.payload, "recommended_action", ""),
            "school_phone": self._phone,
            "parent_requested_meeting": state.parent_requested_meeting,
            "brief_mode": state.brief_mode,
            "pending_date": (
                state.pending_meeting_date.isoformat() if state.pending_meeting_date else None
            ),
            "pending_time": (
                state.pending_meeting_time.strftime("%H:%M") if state.pending_meeting_time else None
            ),
            "current_datetime": self._calendar.now().isoformat(),
            "timezone": str(self._calendar.timezone),
            "working_days": sorted(self._calendar.working_days),
            "recent_turns": state.messages[-6:],
        }
        understanding = self._intent_interpreter.interpret(parent_speech, context)
        state.last_understanding = understanding
        print(
            f"  [NLU] Intent: {understanding.intent.value} | "
            f"Confidence: {understanding.confidence:.2f} | "
            f"Target: {understanding.target_date or '-'} {understanding.target_time or '-'}"
        )
        return understanding

    def _route_contextual_action(
        self,
        state: ConversationState,
        parent_speech: str,
        understanding: TurnUnderstanding,
    ) -> tuple[str, bool] | None:
        action_intents = {
            ParentIntent.SCHEDULE_MEETING,
            ParentIntent.SELECT_SLOT,
            ParentIntent.CHECK_AVAILABILITY,
            ParentIntent.RESCHEDULE_MEETING,
            ParentIntent.CANCEL_MEETING,
            ParentIntent.CONFIRM_ACTION,
            ParentIntent.DECLINE_ACTION,
            ParentIntent.END_CALL,
        }
        if not understanding.interpreted:
            return None
        if understanding.intent in action_intents and (
            understanding.confidence < ContextualIntentInterpreter.MIN_ACTION_CONFIDENCE
        ):
            return (
                "I want to make sure I understood correctly. Are you asking to schedule, "
                "move, or cancel a meeting?",
                False,
            )

        intent = understanding.intent
        normalized = self._normalized_scheduling_request(state, parent_speech, understanding)

        if intent == ParentIntent.CONFIRM_IDENTITY:
            if state.stage == STAGE_INTRO:
                state.stage = STAGE_AVAILABILITY
                return (
                    f"Thank you, {state.payload.parent_name}. Do you have a few minutes "
                    f"to discuss {state.payload.student_name}'s progress?",
                    False,
                )
            if state.stage == STAGE_AVAILABILITY:
                return (self._begin_concern_discussion(state), False)

        if intent == ParentIntent.AVAILABLE_TO_TALK:
            return (self._begin_concern_discussion(state), False)

        if intent == ParentIntent.CONFIRM_ACTION and state.stage == STAGE_AVAILABILITY:
            return (self._begin_concern_discussion(state), False)

        if intent == ParentIntent.AVAILABLE_BRIEFLY:
            if not _expresses_time_limit(parent_speech):
                if state.stage in {STAGE_INTRO, STAGE_AVAILABILITY}:
                    return (self._begin_concern_discussion(state), False)
                return None
            return (self._brief_call_response(state), False)

        if intent == ParentIntent.ASK_CURRENT_DATE:
            return (
                f"Today is {self._calendar.now().strftime('%A, %B %d, %Y').replace(' 0', ' ')}. "
                "Is there another date you'd like me to check?",
                False,
            )

        if state.brief_mode and intent == ParentIntent.DISCUSS_CONCERN:
            return (self._brief_followup_response(state), False)

        scheduling_intents = {
            ParentIntent.SCHEDULE_MEETING,
            ParentIntent.CHECK_AVAILABILITY,
            ParentIntent.SELECT_SLOT,
            ParentIntent.CONFIRM_ACTION,
        }
        if state.stage == STAGE_MEETING and intent in scheduling_intents:
            if understanding.range_start and understanding.range_end:
                return (
                    self._offer_slots_for_range(
                        state,
                        calendar_date.fromisoformat(understanding.range_start),
                        calendar_date.fromisoformat(understanding.range_end),
                    ),
                    False,
                )
            if understanding.target_date and not understanding.target_time:
                return (
                    self._offer_slots_for_date(
                        state, calendar_date.fromisoformat(understanding.target_date)
                    ),
                    False,
                )
            return (self._handle_meeting_choice(state, normalized), False)

        if state.stage == STAGE_RESCHEDULE and state.booking and intent in scheduling_intents:
            if understanding.range_start and understanding.range_end:
                return (
                    self._offer_slots_for_range(
                        state,
                        calendar_date.fromisoformat(understanding.range_start),
                        calendar_date.fromisoformat(understanding.range_end),
                        for_reschedule=True,
                    ),
                    False,
                )
            if understanding.target_date and not understanding.target_time:
                return (
                    self._offer_slots_for_date(
                        state, calendar_date.fromisoformat(understanding.target_date),
                        for_reschedule=True,
                    ),
                    False,
                )
            return (self._handle_reschedule(state, normalized), False)

        if intent == ParentIntent.CANCEL_MEETING:
            if not state.booking:
                return ("There isn't an active meeting to cancel during this call.", False)
            return (self._handle_cancellation(state), False)

        if intent == ParentIntent.RESCHEDULE_MEETING:
            if not state.booking:
                return (
                    "There isn't an active meeting to move yet. Would you like me to check available times?",
                    False,
                )
            state.pending_action = "reschedule_meeting"
            state.parent_requested_meeting = True
            if understanding.range_start and understanding.range_end:
                return (
                    self._offer_slots_for_range(
                        state,
                        calendar_date.fromisoformat(understanding.range_start),
                        calendar_date.fromisoformat(understanding.range_end),
                        for_reschedule=True,
                    ),
                    False,
                )
            if understanding.target_date and not understanding.target_time:
                return (
                    self._offer_slots_for_date(
                        state, calendar_date.fromisoformat(understanding.target_date),
                        for_reschedule=True,
                    ),
                    False,
                )
            return (self._handle_reschedule(state, normalized), False)

        if intent in {ParentIntent.SCHEDULE_MEETING, ParentIntent.CHECK_AVAILABILITY}:
            if (
                state.payload.risk_level.upper() == "MEDIUM"
                and not state.awaiting_meeting_consent
                and not _explicitly_requests_meeting(parent_speech)
            ):
                state.stage = STAGE_SOLUTION
                return (
                    "Thank you for explaining. We'll monitor this closely and continue "
                    "supporting your child. Is there anything specific you'd like us to watch for?",
                    False,
                )
            if state.booking:
                return (
                    f"You already have a meeting booked for {CalendarSlot(state.booking.start, state.booking.end).spoken()}. "
                    "Would you like to move that meeting?",
                    False,
                )
            state.pending_action = "schedule_meeting"
            state.parent_requested_meeting = True
            state.awaiting_meeting_consent = False
            if understanding.range_start and understanding.range_end:
                return (
                    self._offer_slots_for_range(
                        state,
                        calendar_date.fromisoformat(understanding.range_start),
                        calendar_date.fromisoformat(understanding.range_end),
                    ),
                    False,
                )
            if understanding.target_date and not understanding.target_time:
                return (
                    self._offer_slots_for_date(
                        state, calendar_date.fromisoformat(understanding.target_date)
                    ),
                    False,
                )
            return (self._offer_available_slots(state), False)

        if intent == ParentIntent.SELECT_SLOT:
            if state.stage == STAGE_RESCHEDULE and state.booking:
                return (self._handle_reschedule(state, normalized), False)
            if state.stage == STAGE_MEETING:
                return (self._handle_meeting_choice(state, normalized), False)
            return None

        if intent == ParentIntent.CONFIRM_ACTION:
            if state.stage == STAGE_RESCHEDULE and state.booking:
                return (self._handle_reschedule(state, normalized), False)
            if state.stage == STAGE_MEETING:
                return (self._handle_meeting_choice(state, normalized), False)
            meeting_context = "meeting" in parent_speech.lower() or (
                "meeting" in state.last_agent_message.lower()
                and re.search(r"\b(?:yes|sure|okay|ok|please do|go ahead)\b", parent_speech.lower())
            )
            if meeting_context and not state.booking:
                state.parent_requested_meeting = True
                state.awaiting_meeting_consent = False
                state.pending_action = "schedule_meeting"
                return (self._offer_available_slots(state), False)
            return None

        if intent == ParentIntent.DECLINE_ACTION and state.stage in {
            STAGE_MEETING, STAGE_RESCHEDULE
        }:
            state.offered_slots = []
            state.pending_action = None
            self._clear_pending_meeting_request(state)
            state.stage = STAGE_FAREWELL
            return ("Of course, I understand. Is there anything else you'd like to discuss?", False)

        if intent == ParentIntent.END_CALL:
            if state.stage == STAGE_FAREWELL:
                state.ended = True
                return (
                    f"Thank you for your time. The school is available Monday to Friday, "
                    f"9 AM to 4 PM, at {self._phone}. Goodbye.",
                    True,
                )
            state.stage = STAGE_FAREWELL
            return ("Before we finish, is there anything else you'd like to discuss?", False)

        return None

    def _begin_concern_discussion(self, state: ConversationState) -> str:
        details = " ".join(state.payload.details.split()).strip()
        if len(details) > 220:
            details = details[:217].rsplit(" ", 1)[0] + "..."
        state.stage = STAGE_CONVERSATION
        dimension = state.payload.dimension.lower().replace("behaviour", "behavior")
        question = {
            "attendance": "Could you help me understand what may be affecting the attendance?",
            "performance": "Have you noticed anything that may be affecting their studies?",
            "behavior": "How have things been at home recently?",
        }.get(dimension, "Could you share what you have noticed at home?")
        return f"Thank you. I'm calling because {details} {question}"

    def _normalized_scheduling_request(
        self,
        state: ConversationState,
        original_speech: str,
        understanding: TurnUnderstanding,
    ) -> str:
        if understanding.selected_option:
            return f"option {understanding.selected_option}"
        target_date = understanding.target_date
        if not target_date and understanding.target_time and state.booking:
            target_date = state.booking.start.date().isoformat()
        if target_date and understanding.target_time:
            return f"{target_date} at {understanding.target_time}"
        if target_date:
            return target_date
        if understanding.target_time:
            return f"at {understanding.target_time}"
        return original_speech

    def _tracked_twiml(
        self, state: ConversationState, spoken: str, end_call: bool = False
    ) -> str:
        state.messages.append({"role": "assistant", "content": spoken})
        self._record_turn(state, "agent", spoken)
        state.last_agent_message = spoken
        if state.payload.risk_level.upper() == "HIGH" and _asks_for_meeting_consent(spoken):
            state.awaiting_meeting_consent = True
        state.advance_stage()
        return self._twiml(spoken, end_call)

    def _record_turn(
        self, state: ConversationState, speaker: str, message: str,
        *, intent: str | None = None,
    ) -> None:
        repository = getattr(self, "_call_history", None)
        if not repository or not repository.enabled or not state.call_sid:
            return
        try:
            repository.append_turn(
                state.call_sid, speaker, message, intent=intent, stage=state.stage
            )
        except Exception:
            logger.exception("Could not persist %s turn for %s", speaker, state.call_sid)

    def record_system_turn(self, call_sid: str, message: str) -> None:
        self.record_call_turn(call_sid, "system", message)

    def record_call_turn(self, call_sid: str, speaker: str, message: str) -> None:
        repository = getattr(self, "_call_history", None)
        if repository and repository.enabled:
            try:
                repository.append_turn(call_sid, speaker, message)
            except Exception:
                logger.exception("Could not persist %s turn for %s", speaker, call_sid)

    def update_call_status(
        self, call_sid: str, status: str, duration: int | None = None
    ) -> None:
        repository = getattr(self, "_call_history", None)
        if repository and repository.enabled:
            repository.update_status(call_sid, status, duration)

    def finalize_call_summary(self, call_sid: str) -> None:
        repository = getattr(self, "_call_history", None)
        if not repository or not repository.enabled or not repository.claim_summary(call_sid):
            return
        turns = repository.get_turns(call_sid)
        if not any(turn.get("speaker") == "parent" for turn in turns):
            repository.fail_summary(call_sid, "No parent speech was captured.")
            return
        call = repository.get_call(call_sid) or {}
        meeting_start = call.get("meeting_start")
        meeting_end = call.get("meeting_end")
        transcript = [
            {"speaker": turn["speaker"], "message": turn["message"]}
            for turn in turns
        ]
        try:
            if not self._ai_ready:
                raise RuntimeError("Groq is not configured for summary generation.")
            response = self._groq.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize a school-parent call as one JSON object using only explicit "
                            "facts. Never invent causes, diagnoses, promises, or meeting details. "
                            "The brief_summary must concisely state what was discussed, any reason "
                            "the parent explicitly gave, the agreed next action, and the meeting "
                            "outcome when meeting metadata is present. If the parent gave no reason, "
                            "do not infer one. Treat meeting metadata as authoritative. "
                            "Required keys: brief_summary, parent_concerns, school_observations, "
                            "agreed_actions, unresolved_questions, follow_up_required, parent_sentiment."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "metadata": {
                                "student_name": call.get("student_name"),
                                "dimension": call.get("dimension"),
                                "risk_level": call.get("risk_level"),
                                "meeting_status": call.get("meeting_status"),
                                "meeting_start": meeting_start.isoformat() if meeting_start else None,
                                "meeting_end": meeting_end.isoformat() if meeting_end else None,
                            },
                            "transcript": transcript,
                        }, ensure_ascii=True),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=500,
            )
            summary = json.loads(response.choices[0].message.content)
            for key in (
                "parent_concerns", "school_observations", "agreed_actions",
                "unresolved_questions",
            ):
                if not isinstance(summary.get(key), list):
                    summary[key] = []
            summary["brief_summary"] = str(summary.get("brief_summary", ""))[:2000]
            summary["follow_up_required"] = bool(summary.get("follow_up_required", False))
            summary["parent_sentiment"] = str(summary.get("parent_sentiment", "neutral"))[:40]
            repository.save_summary(call_sid, summary)
        except Exception as exc:
            logger.exception("Summary generation failed for %s", call_sid)
            repository.fail_summary(call_sid, str(exc))

    def _contextual_reply_twiml(
        self,
        state: ConversationState,
        parent_speech: str,
        understanding: TurnUnderstanding,
    ) -> str:
        if not understanding.interpreted:
            spoken = "I'm sorry, I didn't catch that clearly. Could you please say it once more?"
        else:
            spoken = understanding.assistant_reply or (
                "I want to make sure I understood you correctly. Could you please clarify?"
            )
        unsafe_calendar_claim = re.search(
            r"\b(?:booked|scheduled|rescheduled|cancelled|canceled)\b.*\bmeeting\b|"
            r"\bmeeting\b.*\b(?:booked|scheduled|rescheduled|cancelled|canceled)\b",
            spoken.lower(),
        )
        if unsafe_calendar_claim:
            spoken = "I need to confirm that through the teacher's calendar first. What date or time would you prefer?"
        risk = state.payload.risk_level.upper()
        if risk == "HIGH" and understanding.intent == ParentIntent.DISCUSS_CONCERN:
            if not _asks_for_meeting_consent(spoken):
                spoken = spoken.rstrip(" .?") + (
                    ". Because this is urgent, would you like me to check the teacher's "
                    "earliest available meeting times?"
                )
            state.stage = STAGE_SOLUTION
        elif (
            risk == "MEDIUM"
            and not state.parent_requested_meeting
            and _explicitly_requests_meeting(spoken)
        ):
            spoken = (
                "Thank you for explaining. We'll monitor this closely and continue "
                "supporting your child. Is there anything specific you'd like us to watch for at school?"
            )
            state.stage = STAGE_SOLUTION
        elif risk == "LOW" and _explicitly_requests_meeting(spoken):
            spoken = (
                f"Thank you for explaining. We'll continue encouraging {state.payload.student_name} "
                "and address this early. Is there anything else you'd like us to know?"
            )
            state.stage = STAGE_SOLUTION
        if not spoken.rstrip().endswith("?"):
            spoken = spoken.rstrip(" .") + ". Does that sound reasonable to you?"
        return self._tracked_twiml(state, spoken, False)

    # ── Parse control tag ─────────────────────────────────────────
    def _parse_reply(self, ai_text: str) -> tuple[str, bool, bool]:
        end_call = "[END_CALL]" in ai_text
        check_availability = "[CHECK_AVAILABILITY]" in ai_text
        spoken = (ai_text
                  .replace("[END_CALL]", "")
                  .replace("[CONTINUE]", "")
                  .replace("[CHECK_AVAILABILITY]", "")
                  .strip())
        return spoken, end_call, check_availability

    def _brief_call_response(self, state: ConversationState) -> str:
        if state.brief_mode:
            return self._brief_followup_response(state)
        state.brief_mode = True
        state.stage = STAGE_SOLUTION
        details = " ".join(state.payload.details.split())
        if len(details) > 220:
            details = details[:217].rsplit(" ", 1)[0] + "..."
        opening = f"I'll keep this brief. The reason for my call is that {details}"
        risk = state.payload.risk_level.upper()
        if risk == "HIGH":
            return (
                f"{opening} Because this is urgent, would you like me to check the "
                "teacher's earliest meeting times?"
            )
        if risk == "MEDIUM":
            return (
                f"{opening} We'll monitor this closely and follow up if it continues. "
                "Is there anything important you'd like us to know?"
            )
        return f"{opening} I wanted to make you aware so we can address it early. Does that sound reasonable?"

    def _brief_followup_response(self, state: ConversationState) -> str:
        risk = state.payload.risk_level.upper()
        state.stage = STAGE_SOLUTION
        if risk == "HIGH":
            return (
                "Thank you for explaining. Because this is urgent, would you like me to "
                "check the teacher's earliest meeting times?"
            )
        if risk == "MEDIUM":
            return (
                "Thank you for explaining. We'll monitor this closely and contact you if it continues. "
                "Is there anything specific you'd like us to watch for?"
            )
        return (
            "Thank you for explaining. We'll note this and continue supporting your child. "
            "Does that sound reasonable?"
        )

    def _meeting_horizon_days(self, state: ConversationState) -> int | None:
        risk = state.payload.risk_level.upper()
        if risk == "HIGH":
            return 2
        if risk == "MEDIUM" and state.parent_requested_meeting:
            return 7
        return None

    def _meeting_policy_message(self, state: ConversationState) -> str | None:
        risk = state.payload.risk_level.upper()
        if risk == "MEDIUM" and not state.parent_requested_meeting:
            return "We'll monitor this closely. If you would like a meeting, please ask and I can check the teacher's schedule."
        if risk == "LOW":
            return f"A meeting isn't normally needed for this concern. Please contact the school at {self._phone} if you'd still like to arrange one."
        return None

    def _horizon_error_for_date(
        self, state: ConversationState, requested_date: calendar_date
    ) -> str | None:
        horizon = self._meeting_horizon_days(state)
        if horizon is None:
            return self._meeting_policy_message(state)
        last_date = self._calendar.now().date() + timedelta(days=horizon)
        if requested_date > last_date:
            label = "two days" if horizon == 2 else "one week"
            return (
                f"Automated meeting options for this {state.payload.risk_level.lower()}-risk case "
                f"are limited to the next {label}. For a later date, please contact the "
                f"school at {self._phone}."
            )
        return None

    def _offer_slots_for_date(
        self,
        state: ConversationState,
        requested_date: calendar_date,
        *,
        for_reschedule: bool = False,
    ) -> str:
        if requested_date.weekday() not in self._calendar.working_days:
            return (
                f"{requested_date.strftime('%A')} is outside the teacher's working days. "
                f"Meetings are available {_working_days_label(self._calendar.working_days)}."
            )
        policy_message = self._horizon_error_for_date(state, requested_date)
        if policy_message:
            return policy_message
        day_start = datetime.combine(requested_date, clock_time.min, self._calendar.timezone)
        cursor = max(day_start, self._calendar.now())
        try:
            slots = self._calendar.find_available_slots(
                state.payload.teacher_id, start=cursor, days=0, limit=3
            )
        except Exception as exc:
            logger.error("Calendar date availability error: %s", exc)
            return f"I can't access the teacher's calendar right now. Please contact the school at {self._phone}."
        if not slots:
            return (
                f"There are no available meeting times on {requested_date.strftime('%A, %B %d')}. "
                f"Please contact the school at {self._phone} for another arrangement."
            )
        state.offered_slots = slots
        state.pending_meeting_date = requested_date
        state.pending_action = "reschedule_meeting" if for_reschedule else "schedule_meeting"
        state.stage = STAGE_RESCHEDULE if for_reschedule else STAGE_MEETING
        return (
            f"The available times on {requested_date.strftime('%A, %B %d')} are "
            f"{_join_spoken_slots(slots)}. Which works best for you?"
        )

    def _offer_slots_for_range(
        self,
        state: ConversationState,
        range_start: calendar_date,
        range_end: calendar_date,
        *,
        for_reschedule: bool = False,
    ) -> str:
        policy_message = self._meeting_policy_message(state)
        if policy_message:
            return policy_message
        horizon = self._meeting_horizon_days(state)
        last_allowed = self._calendar.now().date() + timedelta(days=horizon or 0)
        if range_start > last_allowed:
            return self._horizon_error_for_date(state, range_start) or (
                f"Please contact the school at {self._phone} for that date range."
            )
        effective_end = min(range_end, last_allowed)
        cursor = max(
            self._calendar.now(),
            datetime.combine(range_start, clock_time.min, self._calendar.timezone),
        )
        days = max(0, (effective_end - cursor.date()).days)
        try:
            slots = self._calendar.find_available_slots(
                state.payload.teacher_id, start=cursor, days=days, limit=3
            )
        except Exception as exc:
            logger.error("Calendar range availability error: %s", exc)
            return f"I can't access the teacher's calendar right now. Please contact the school at {self._phone}."
        if not slots:
            return (
                "There are no available meeting times in that period. "
                f"Please contact the school at {self._phone} for another arrangement."
            )
        state.offered_slots = slots
        state.pending_action = "reschedule_meeting" if for_reschedule else "schedule_meeting"
        state.stage = STAGE_RESCHEDULE if for_reschedule else STAGE_MEETING
        return f"The available times in that period are {_join_spoken_slots(slots)}. Which works best for you?"

    def _offer_available_slots(
        self, state: ConversationState, prefix: str = "", *, for_reschedule: bool = False
    ) -> str:
        policy_message = self._meeting_policy_message(state)
        if policy_message:
            return policy_message
        horizon = self._meeting_horizon_days(state)
        try:
            slots = self._calendar.find_available_slots(
                state.payload.teacher_id, days=horizon or 0, limit=3
            )
        except Exception as exc:
            logger.error("Calendar availability error: %s", exc)
            state.stage = STAGE_FAREWELL
            return (
                "I'm sorry, I can't access the teacher's calendar right now. "
                f"Please call the school at {self._phone} and we'll arrange the meeting for you."
            )
        state.offered_slots = slots
        if not slots:
            state.stage = STAGE_FAREWELL
            window = "two days" if horizon == 2 else "one week"
            return (
                f"I checked the teacher's calendar, but there aren't any openings in the next {window}. "
                f"Please call the school at {self._phone} and we'll help arrange another time."
            )

        state.stage = STAGE_RESCHEDULE if for_reschedule else STAGE_MEETING
        state.pending_action = "reschedule_meeting" if for_reschedule else "schedule_meeting"
        choices = _join_spoken_slots(slots)
        lead = f"{prefix.strip()} " if prefix.strip() else ""
        return f"{lead}I checked the teacher's calendar. The available times are {choices}. Which works best for you?"

    def _handle_meeting_choice(self, state: ConversationState, parent_speech: str) -> str:
        if _declines_meeting(parent_speech):
            state.stage = STAGE_FAREWELL
            state.offered_slots = []
            state.pending_action = None
            self._clear_pending_meeting_request(state)
            return "Of course, I understand. Is there anything else you'd like to discuss?"

        slot = self._requested_slot_with_context(state, parent_speech)
        if not slot:
            return self._clarify_meeting_request(state, parent_speech)

        policy_error = self._slot_policy_error(slot)
        if policy_error:
            choices = _join_spoken_slots(state.offered_slots[:2])
            return f"{policy_error} The next available options are {choices}. Which would you prefer?"
        horizon_error = self._horizon_error_for_date(state, slot.start.date())
        if horizon_error:
            return horizon_error

        try:
            booking = self._calendar.book_meeting(
                state.payload.teacher_id,
                slot,
                student_name=state.payload.student_name,
                parent_name=state.payload.parent_name,
                reason=state.payload.dimension,
            )
        except ValueError:
            return self._offer_available_slots(
                state, "That time was just taken, so I've refreshed the calendar."
            )
        except Exception as exc:
            logger.error("Calendar booking error: %s", exc)
            state.stage = STAGE_FAREWELL
            return (
                "I'm sorry, I couldn't complete the booking just now. "
                f"Please call the school at {self._phone} and we'll reserve it for you."
            )

        state.booking = booking
        repository = getattr(self, "_call_history", None)
        if repository and repository.enabled and state.call_sid:
            repository.update_meeting(state.call_sid, booking, "booked")
        state.offered_slots = []
        state.pending_action = None
        self._clear_pending_meeting_request(state)
        state.stage = STAGE_FAREWELL
        return f"Your meeting is booked for {slot.spoken()}. Is there anything else I can help with?"

    def _handle_reschedule(self, state: ConversationState, parent_speech: str) -> str:
        state.pending_action = "reschedule_meeting"
        state.parent_requested_meeting = True
        slot = self._requested_slot_with_context(state, parent_speech)
        if not slot:
            state.stage = STAGE_RESCHEDULE
            return self._clarify_meeting_request(state, parent_speech, action="move the meeting")

        policy_error = self._slot_policy_error(slot)
        if policy_error:
            return self._offer_available_slots(
                state, policy_error, for_reschedule=True
            )
        horizon_error = self._horizon_error_for_date(state, slot.start.date())
        if horizon_error:
            return horizon_error

        try:
            state.booking = self._calendar.reschedule_meeting(state.booking, slot)
        except ValueError:
            return self._offer_available_slots(
                state, "That requested time isn't available.", for_reschedule=True
            )
        except Exception as exc:
            logger.error("Calendar reschedule error: %s", exc)
            state.stage = STAGE_FAREWELL
            return (
                "I'm sorry, I couldn't update the calendar just now. "
                f"Please call the school at {self._phone} and we'll change it for you."
            )

        repository = getattr(self, "_call_history", None)
        if repository and repository.enabled and state.call_sid:
            repository.update_meeting(state.call_sid, state.booking, "rescheduled")

        state.offered_slots = []
        state.pending_action = None
        self._clear_pending_meeting_request(state)
        state.stage = STAGE_FAREWELL
        return f"Your meeting has been moved to {slot.spoken()}. Is there anything else I can help with?"

    def _handle_cancellation(self, state: ConversationState) -> str:
        try:
            self._calendar.cancel_meeting(state.booking)
        except Exception as exc:
            logger.error("Calendar cancellation error: %s", exc)
            return (
                "I'm sorry, I couldn't cancel the calendar event just now. "
                f"Please call the school at {self._phone} for help."
            )
        state.booking = None
        repository = getattr(self, "_call_history", None)
        if repository and repository.enabled and state.call_sid:
            repository.update_meeting(state.call_sid, None, "cancelled")
        state.offered_slots = []
        state.pending_action = None
        self._clear_pending_meeting_request(state)
        state.stage = STAGE_FAREWELL
        return "The meeting has been cancelled in the teacher's calendar. Is there anything else I can help with?"

    def _clarify_meeting_request(
        self, state: ConversationState, parent_speech: str, action: str = "schedule the meeting"
    ) -> str:
        parsed_now = _parse_meeting_request(parent_speech, self._calendar.timezone)
        parsed = ParsedMeetingRequest(
            parsed_now.date or state.pending_meeting_date,
            parsed_now.time or state.pending_meeting_time,
        )
        if parsed.date and parsed.date.weekday() not in self._calendar.working_days:
            valid_days = _working_days_label(self._calendar.working_days)
            return f"That day is outside the teacher's working days. Meetings are available {valid_days}. Which working day would you prefer?"
        if parsed.date and not parsed.time:
            return f"What time on {parsed.date.strftime('%A, %B %d')} would you prefer?"
        if parsed.time and not parsed.date:
            clock = parsed.time.strftime("%I:%M %p").lstrip("0")
            return f"Which working day would you prefer for {clock}?"
        if state.offered_slots:
            return f"Please choose one of these times: {_join_spoken_slots(state.offered_slots)}."
        return f"What day and time would you like to {action}?"

    def _requested_slot_with_context(
        self, state: ConversationState, parent_speech: str
    ) -> CalendarSlot | None:
        slot = _match_requested_slot(
            parent_speech, state.offered_slots, self._calendar.timezone,
            self._calendar.duration,
        )
        if slot:
            return slot
        parsed = _parse_meeting_request(parent_speech, self._calendar.timezone)
        if parsed.date:
            state.pending_meeting_date = parsed.date
        if parsed.time:
            state.pending_meeting_time = parsed.time
            if (
                not parsed.date
                and state.pending_action == "reschedule_meeting"
                and state.booking
            ):
                state.pending_meeting_date = state.booking.start.date()
        if state.pending_meeting_date and state.pending_meeting_time:
            start = datetime.combine(
                state.pending_meeting_date, state.pending_meeting_time,
                self._calendar.timezone,
            )
            return CalendarSlot(start, start + self._calendar.duration)
        return None

    @staticmethod
    def _clear_pending_meeting_request(state: ConversationState) -> None:
        state.pending_meeting_date = None
        state.pending_meeting_time = None

    def _slot_policy_error(self, slot: CalendarSlot) -> str | None:
        if slot.start.weekday() not in self._calendar.working_days:
            return f"Meetings are only available {_working_days_label(self._calendar.working_days)}."
        if slot.start <= self._calendar.now():
            return "That requested time has already passed."
        if (
            slot.start.timetz().replace(tzinfo=None) < self._calendar.meeting_start
            or slot.end.timetz().replace(tzinfo=None) > self._calendar.meeting_end
        ):
            start = self._calendar.meeting_start.strftime("%I:%M %p").lstrip("0")
            end = self._calendar.meeting_end.strftime("%I:%M %p").lstrip("0")
            return f"Meeting times are available between {start} and {end}."
        return None

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
        <Say voice="Polly.Aditi" language="en-IN">Hello. This is Priya calling from {safe_school}.</Say>
        <Pause length="1"/>
        <Say voice="Polly.Aditi" language="en-IN">Am I speaking with {safe_parent}?</Say>
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

        # Persist every recognized parent turn before routing it. This history is
        # supplied to the next interpretation so the agent does not repeat a
        # question that the parent has already answered.
        state.messages.append({"role": "user", "content": parent_speech})
        state.turn_count += 1
        self._record_turn(state, "parent", parent_speech)

        if _declines_further_help(parent_speech):
            state.awaiting_meeting_consent = False
            state.stage = STAGE_FAREWELL
            spoken = "I understand. We'll continue monitoring the concern. Thank you for your time, and goodbye."
            state.ended = True
            return self._tracked_twiml(state, spoken, True)

        if (
            not state.booking
            and state.awaiting_meeting_consent
            and _is_short_affirmative(parent_speech)
        ):
            state.parent_requested_meeting = True
            state.awaiting_meeting_consent = False
            state.pending_action = "schedule_meeting"
            spoken = self._offer_available_slots(state)
            return self._tracked_twiml(state, spoken, False)

        # Resolve deterministic calendar questions before any network AI call.
        if _asks_current_date(parent_speech):
            spoken = (
                f"Today is {self._calendar.now().strftime('%A, %B %d, %Y').replace(' 0', ' ')}. "
                "Is there another date you'd like me to check?"
            )
            return self._tracked_twiml(state, spoken, False)

        if state.booking and _wants_to_cancel(parent_speech):
            spoken = self._handle_cancellation(state)
            return self._tracked_twiml(state, spoken, False)

        if _asks_meeting_availability(parent_speech):
            parsed_request = _parse_meeting_request(parent_speech, self._calendar.timezone)
            state.parent_requested_meeting = True
            if parsed_request.range_start and parsed_request.range_end:
                spoken = self._offer_slots_for_range(
                    state, parsed_request.range_start, parsed_request.range_end,
                    for_reschedule=bool(state.booking),
                )
                return self._tracked_twiml(state, spoken, False)
            if parsed_request.date and not parsed_request.time:
                spoken = self._offer_slots_for_date(
                    state, parsed_request.date, for_reschedule=bool(state.booking)
                )
                return self._tracked_twiml(state, spoken, False)

        booking_change = _booking_change_from_temporal_reference(
            state, parent_speech, self._calendar.timezone
        )
        if booking_change:
            state.parent_requested_meeting = True
            if booking_change.range_start and booking_change.range_end:
                spoken = self._offer_slots_for_range(
                    state, booking_change.range_start, booking_change.range_end,
                    for_reschedule=True,
                )
            elif booking_change.date and not booking_change.time:
                spoken = self._offer_slots_for_date(
                    state, booking_change.date, for_reschedule=True
                )
            else:
                spoken = self._handle_reschedule(state, parent_speech)
            return self._tracked_twiml(state, spoken, False)

        if state.stage == STAGE_MEETING and _is_short_affirmative(parent_speech):
            spoken = self._handle_meeting_choice(state, parent_speech)
            return self._tracked_twiml(state, spoken, False)

        if (
            not state.booking
            and _is_short_affirmative(parent_speech)
            and "meeting" in state.last_agent_message.lower()
        ):
            state.parent_requested_meeting = True
            state.pending_action = "schedule_meeting"
            spoken = self._offer_available_slots(state)
            return self._tracked_twiml(state, spoken, False)

        understanding = self._interpret_parent_turn(state, parent_speech)
        routed = self._route_contextual_action(state, parent_speech, understanding)
        if routed:
            spoken, end_call = routed
            return self._tracked_twiml(state, spoken, end_call)

        if understanding.interpreted:
            return self._contextual_reply_twiml(
                state, parent_speech, understanding
            )

        return self._contextual_reply_twiml(state, parent_speech, understanding)

        # High-precision legacy checks remain as a fail-safe if interpretation fails.
        if state.booking and _wants_to_cancel(parent_speech):
            spoken = self._handle_cancellation(state)
            return self._tracked_twiml(state, spoken, False)

        if state.stage == STAGE_RESCHEDULE or (
            state.booking and _wants_to_reschedule(parent_speech)
        ):
            spoken = self._handle_reschedule(state, parent_speech)
            return self._tracked_twiml(state, spoken, False)

        if state.stage == STAGE_MEETING:
            spoken = self._handle_meeting_choice(state, parent_speech)
            print(f"  [P] Parent : \"{parent_speech}\"")
            print(f"  [CAL] Priya: \"{spoken[:90]}...\"")
            return self._tracked_twiml(state, spoken, False)

        # Stage-aware closing detection
        if _parent_wants_to_end(parent_speech, state.stage):
            if state.turn_count >= 4:
                if state.stage == STAGE_FAREWELL:
                    # Parent confirmed nothing more — give final goodbye with working hours
                    speech_for_ai = (
                        parent_speech +
                        " [Note: Parent has confirmed they have nothing more to discuss."
                        " Give ONE warm closing sentence in English only."
                        " Mention the school is available Monday to Friday, 9 AM to 4 PM,"
                        " and they can call at any time. Use [END_CALL].]"
                    )
                else:
                    # First closing signal — move to farewell, ask if anything else
                    state.stage = STAGE_FAREWELL
                    speech_for_ai = (
                        parent_speech +
                        " [Note: Before ending, warmly ask if the parent has any other"
                        " questions or concerns they'd like to discuss. Do NOT use"
                        " [END_CALL] yet. Use [CONTINUE].]"
                    )
            else:
                # Too early — treat as normal reply, don't close
                speech_for_ai = parent_speech
        else:
            speech_for_ai = parent_speech

        ai_text = self._ask_groq(state, speech_for_ai)
        spoken, end_call, check_availability = self._parse_reply(ai_text)

        if check_availability:
            spoken = self._offer_available_slots(state, spoken)
            end_call = False

        if end_call:
            state.ended = True

        state.last_agent_message = spoken

        print(f"  [P] Parent : \"{parent_speech}\"")
        print(f"  [AI] Priya  : \"{spoken[:90]}...\"")
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
        print(f"[OK] Ready: {payload.student_name} [{payload.dimension.upper()} / {payload.risk_level}]")

        twiml = self._opening_twiml(payload)

        try:
            print(f"\n[CALL] Calling {payload.parent_name} ({payload.to_number})...")
            call_options = dict(
                to=payload.to_number,
                from_=self._from_number,
                twiml=twiml,
            )
            if self._ngrok_url:
                call_options.update(
                    status_callback=f"{self._ngrok_url}/twilio/call-status",
                    status_callback_event=["initiated", "ringing", "answered", "completed"],
                    status_callback_method="POST",
                )
            call = self._client.calls.create(**call_options)
            # CallSid is unique even when the same student has overlapping calls.
            state.call_sid = call.sid
            self._conversations[call.sid] = state
            repository = getattr(self, "_call_history", None)
            if repository and repository.enabled:
                opening_message = (
                    f"Hello. This is Priya calling from {self._school}. "
                    f"Am I speaking with {payload.parent_name}?"
                )
                repository.create_call(call.sid, payload, opening_message)
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
