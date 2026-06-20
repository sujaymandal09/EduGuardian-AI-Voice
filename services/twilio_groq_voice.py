"""
services/twilio_groq_voice.py
──────────────────────────────
2-Way AI Voice — Groq (llama-3.3-70b) + Twilio
"""
import html
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
    return bool(re.search(
        r"\b(?:make|do)\s+(?:the\s+meeting\s+|it\s+|that\s+)?(?:for\s+|at\s+)?"
        r"(?:\d{1,2}(?::\d{2})?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        text,
    ))


def _wants_to_cancel(speech: str) -> bool:
    text = speech.lower()
    return any(phrase in text for phrase in (
        "cancel the meeting", "cancel my meeting", "cancel our meeting",
        "delete the meeting", "remove the meeting", "can't attend the meeting",
        "cannot attend the meeting",
    ))


@dataclass(frozen=True)
class ParsedMeetingRequest:
    date: calendar_date | None = None
    time: clock_time | None = None


def _parse_meeting_request(speech: str, timezone, now: datetime | None = None) -> ParsedMeetingRequest:
    text = speech.lower().strip()
    today = (now or datetime.now(timezone)).astimezone(timezone).date()
    requested_date = None

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
        elif "day after tomorrow" in text:
            requested_date = today + timedelta(days=2)
        elif "tomorrow" in text:
            requested_date = today + timedelta(days=1)
        elif "today" in text:
            requested_date = today
        else:
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for weekday, name in enumerate(weekdays):
                if re.search(rf"\b{name}\b", text):
                    distance = (weekday - today.weekday()) % 7
                    if "next " + name in text and distance == 0:
                        distance = 7
                    requested_date = today + timedelta(days=distance)
                    break
    except ValueError:
        requested_date = None

    requested_time = None
    time_match = (
        re.search(r"\b(?:at|around|from|by)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", text)
        or re.search(r"\b(\d{1,2})[:.](\d{2})\s*(a\.?m\.?|p\.?m\.?)?\b", text)
        or re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", text)
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
            r"\b(?:at|around|from|by)\s+(" + "|".join(word_hours) + r")\b", text
        )
        if word_match:
            hour = word_hours[word_match.group(1)]
            if "pm" in text and hour < 12:
                hour += 12
            requested_time = clock_time(hour, 30 if "thirty" in text else 0)

    return ParsedMeetingRequest(requested_date, requested_time)


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

    def __init__(self, calendar_service: CalendarService | None = None):
        self._twilio_sid   = os.getenv("TWILIO_ACCOUNT_SID")
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self._from_number  = os.getenv("TWILIO_FROM_NUMBER")
        self._groq_key     = os.getenv("GROQ_API_KEY")
        self._ngrok_url    = os.getenv("NGROK_URL", "")
        self._school       = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone        = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self._calendar     = calendar_service or create_calendar_service()

        self._twilio_ready = all([self._twilio_sid, self._twilio_token, self._from_number])
        self._ai_ready     = bool(self._groq_key and Groq)

        if self._twilio_ready:
            from twilio.rest import Client
            self._client = Client(self._twilio_sid, self._twilio_token)
            print("[OK] Twilio Connected")

        if self._ai_ready:
            self._groq = Groq(api_key=self._groq_key)
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
        state.advance_stage()

        try:
            # Cap message history to last 12 messages to limit input tokens
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

    def _offer_available_slots(
        self, state: ConversationState, prefix: str = "", *, for_reschedule: bool = False
    ) -> str:
        try:
            slots = self._calendar.find_available_slots(state.payload.teacher_id, limit=3)
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
            return (
                "I checked the teacher's calendar, but there aren't any openings in the next week. "
                f"Please call the school at {self._phone} and we'll help arrange another time."
            )

        state.stage = STAGE_RESCHEDULE if for_reschedule else STAGE_MEETING
        choices = _join_spoken_slots(slots)
        lead = f"{prefix.strip()} " if prefix.strip() else ""
        return f"{lead}I checked the teacher's calendar. The available times are {choices}. Which works best for you?"

    def _handle_meeting_choice(self, state: ConversationState, parent_speech: str) -> str:
        if _declines_meeting(parent_speech):
            state.stage = STAGE_FAREWELL
            state.offered_slots = []
            self._clear_pending_meeting_request(state)
            return "Of course, I understand. Is there anything else you'd like to discuss?"

        slot = self._requested_slot_with_context(state, parent_speech)
        if not slot:
            return self._clarify_meeting_request(state, parent_speech)

        policy_error = self._slot_policy_error(slot)
        if policy_error:
            choices = _join_spoken_slots(state.offered_slots[:2])
            return f"{policy_error} The next available options are {choices}. Which would you prefer?"

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
        state.offered_slots = []
        self._clear_pending_meeting_request(state)
        state.stage = STAGE_FAREWELL
        return f"Your meeting is booked for {slot.spoken()}. Is there anything else I can help with?"

    def _handle_reschedule(self, state: ConversationState, parent_speech: str) -> str:
        slot = self._requested_slot_with_context(state, parent_speech)
        if not slot:
            state.stage = STAGE_RESCHEDULE
            return self._clarify_meeting_request(state, parent_speech, action="move the meeting")

        policy_error = self._slot_policy_error(slot)
        if policy_error:
            return self._offer_available_slots(
                state, policy_error, for_reschedule=True
            )

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

        state.offered_slots = []
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
        state.offered_slots = []
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
        if slot.start <= datetime.now(self._calendar.timezone):
            return "That requested time has already passed."
        if slot.start.weekday() not in self._calendar.working_days:
            return f"Meetings are only available {_working_days_label(self._calendar.working_days)}."
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
        <Say voice="Polly.Aditi" language="en-IN">Hello! Am I speaking with {safe_parent}? This is Priya calling from {safe_school}, regarding your child {safe_student}.</Say>
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

        if state.booking and _wants_to_cancel(parent_speech):
            spoken = self._handle_cancellation(state)
            return self._twiml(spoken, False)

        if state.stage == STAGE_RESCHEDULE or (
            state.booking and _wants_to_reschedule(parent_speech)
        ):
            spoken = self._handle_reschedule(state, parent_speech)
            return self._twiml(spoken, False)

        if state.stage == STAGE_MEETING:
            spoken = self._handle_meeting_choice(state, parent_speech)
            print(f"  [P] Parent : \"{parent_speech}\"")
            print(f"  [CAL] Priya: \"{spoken[:90]}...\"")
            return self._twiml(spoken, False)

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
