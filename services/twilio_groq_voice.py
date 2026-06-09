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
{student_name}'s attendance is low. Details: {details}
Ask an open question. Listen. Then follow RESOLUTION.

{risk_resolution}
""",
        "performance": f"""
CONCERN: Academic Performance
{student_name} is struggling with grades. Details: {details}
Ask an open question. Listen. Then follow RESOLUTION.

{risk_resolution}
""",
        "behavior": f"""
CONCERN: Behaviour
{student_name} has had behavioural incidents. Details: {details}
Ask an open question. Listen. Then follow RESOLUTION.

{risk_resolution}
""",
    }.get(dim_key, f"""
CONCERN: {dimension.upper()}
Details: {details}
Ask an open question. Listen. Then follow RESOLUTION.

{risk_resolution}
""")

    meeting_rules = {
        "HIGH": "Suggest meeting ASAP. If parent unavailable, persuade once warmly then accept their time.",
        "MEDIUM": "Default is monitoring. Say we'll watch closely, escalate if needed. Only offer meeting if parent asks.",
        "LOW": "No meeting needed. End on encouraging note.",
    }.get(risk_level.upper(), "Suggest meeting if helpful. Accept whatever time parent proposes.")

    return f"""You are Priya — a warm, experienced school counselor calling from {school}.
You are speaking with {parent_name}, parent of {student_name}.
School phone: {phone}

YOUR ROLE:
- You are a school counselor, NOT a parent.
- Listen. Empathize. Ask clarifying questions.
- Never tell the parent what to do. Never give parenting advice.
- Your role is to UNDERSTAND the issue, then suggest NEXT STEPS (meeting, monitoring, resources).

YOUR PERSONALITY:
- You sound like a real human. Warm, calm, genuinely caring.
- Speak in clear, natural, grammatically correct English only. No Hindi or mixed-language words whatsoever.
- Do NOT use words such as: ji, haan, accha, acha, bilkul, theek hai, nahi, or any other Hindi or regional language term — not even as filler or courtesy words.
- If the parent speaks in Hindi or mixed language, always reply in English only.
- You LISTEN first. You acknowledge before moving forward.
- Never robotic. Never scripted. Never repeat yourself.
- 2 to 3 sentences per reply. Phone calls need space.
- Use contractions: I'll, we'll, that's, it's, don't.
- Vary your sentence starters.

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
        if self.stage in (STAGE_FAREWELL, STAGE_CLOSING):
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
    MAX_HISTORY = 12   # keep last 12 messages (~6 exchanges) to limit input tokens
    # Hint words boost ASR probability for these terms over similar-sounding proper nouns
    # (e.g. "sick" over "Sikh", "absent" over ambiguous homophones in en-IN)
    ASR_HINTS = (
        "sick,fever,cold,flu,ill,unwell,doctor,hospital,medicine,better,well,fine,"
        "absent,absence,attendance,homework,exam,test,class,grade,marks,result,subject,"
        "teacher,principal,school,college,classroom,studying,study,tuition,"
        "yes,no,okay,sure,busy,free,available,sorry,understood,question,concern,"
        "meeting,tomorrow,today,this week,next week,morning,afternoon,evening,"
        "son,daughter,child,children,home,family,work,office,worried,stress,pressure,"
        "behaviour,behavior,incident,fight,argument,trouble,issue,problem,"
        "improving,improved,trying,effort,support,help,advice,counsellor,counselor"
    )

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
            # Pre-warm the API to reduce cold-start latency on first call
            try:
                self._groq.chat.completions.create(
                    model=self.MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                )
                print("[OK] Groq Pre-warmed")
            except Exception as e:
                print(f"[WARN] Groq warmup failed: {str(e)[:60]}")
        else:
            print("[WARN] GROQ_API_KEY not set - Demo mode active")

        self._conversations: dict[str, ConversationState] = {}
        self.calls_made = []

    # ── Ask Groq ──────────────────────────────────────────────────
    def _ask_groq(self, state: ConversationState, parent_speech: str) -> str:
        state.messages.append({"role": "user", "content": parent_speech})
        state.turn_count += 1
        state.advance_stage()

        try:
            # Only send the last MAX_HISTORY messages to keep input tokens low
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
    def _parse_reply(self, ai_text: str) -> tuple[str, bool]:
        end_call = "[END_CALL]" in ai_text
        spoken = (ai_text
                  .replace("[END_CALL]", "")
                  .replace("[CONTINUE]", "")
                  .strip())
        return spoken, end_call

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split into individual sentences for per-sentence barge-in Gathers."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    # ── Build TwiML — one Gather per sentence for true barge-in ───
    # Each non-final sentence uses timeout="0": if the parent speaks
    # during it, that speech is captured immediately; if not, Twilio
    # falls through to the next sentence with zero delay.
    # Only the last sentence Gather waits for the parent's reply.
    def _twiml(self, spoken: str, end_call: bool) -> str:
        safe_ph = html.escape(self._phone)
        webhook = f"{self._ngrok_url}/handle-parent-response"

        if end_call:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">{html.escape(spoken)}</Say>
    <Pause length="1"/>
</Response>"""

        sentences = self._split_sentences(spoken) or [spoken]
        gather_blocks = []
        for i, sentence in enumerate(sentences):
            is_last = (i == len(sentences) - 1)
            t = "4" if is_last else "0"
            gather_blocks.append(
                f'    <Gather input="speech" language="en-IN"\n'
                f'            action="{webhook}"\n'
                f'            method="POST"\n'
                f'            timeout="{t}"\n'
                f'            speechTimeout="auto"\n'
                f'            hints="{self.ASR_HINTS}">\n'
                f'        <Say voice="Polly.Aditi" language="en-IN">{html.escape(sentence)}</Say>\n'
                f'    </Gather>'
            )

        fallback = (f'    <Say voice="Polly.Aditi" language="en-IN">'
                    f"I didn't catch that. Please call us at {safe_ph}. Goodbye.</Say>")
        body = "\n".join(gather_blocks) + "\n" + fallback
        return f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n{body}\n</Response>'

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
            timeout="5"
            speechTimeout="auto"
            hints="yes,speaking,hello,this is,yes speaking,yes this is,who is this,wrong number,busy,call back">
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
        spoken, end_call = self._parse_reply(ai_text)

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
        print(f"\n{'='*56}")
        print(f"  DEMO MODE - {payload.dimension.upper()}")
        print(f"  Student: {payload.student_name}   Parent: {payload.parent_name}")
        print(f"  (Set GROQ_API_KEY for real AI calls)")
        print(f"{'='*56}\n")
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
