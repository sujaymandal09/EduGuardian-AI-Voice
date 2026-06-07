"""
services/twilio_gemini_voice.py
────────────────────────────────
Main Orchestrator - English Version
Uses all team modules (Greeting, Conversation, Closing, Error)
"""
import html
import logging
import os
import time
from groq import Groq
from core.models import CallPayload, NotificationResult

# Import team modules
from call_flow.greeting_handler import GreetingHandler
from call_flow.conversation_handler import ConversationHandler
from call_flow.closing_handler import ClosingHandler
from services.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class TwoWayAIVoiceService:
    """2-Way AI Voice - English - Orchestrates all team modules"""

    def __init__(self):
        self._twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self._from_number = os.getenv("TWILIO_FROM_NUMBER")
        self._groq_key = os.getenv("GROQ_API_KEY")
        self._ngrok_url = os.getenv("NGROK_URL", "")
        self._school = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone = os.getenv("SCHOOL_PHONE", "033-4805-1910")

        self._twilio_ready = all([self._twilio_sid, self._twilio_token, self._from_number])
        self._ai_ready = bool(self._groq_key)

        if self._twilio_ready:
            from twilio.rest import Client
            self._client = Client(self._twilio_sid, self._twilio_token)
            print("Twilio Connected")

        if self._ai_ready:
            self._groq_client = Groq(api_key=self._groq_key)
            print("Groq AI Connected")

        # Initialize team modules
        self.greeting = GreetingHandler(self._school, self._phone)
        self.conversation = ConversationHandler(self._school, self._phone)
        self.closing = ClosingHandler(self._school, self._phone)
        self.error_handler = ErrorHandler(self._phone)

        self._conversations = {}
        self._student_data = {}
        self.calls_made = []

    def _get_chat(self, student_data: dict) -> dict:
        """Create Groq chat context with student information."""
        system_prompt = f"""You are Priya, a professional and caring teacher from {self._school}.

CRITICAL RULES:
- Speak ONLY in English
- Be warm, professional, and helpful
- Keep responses SHORT (2-3 sentences for voice)
- Address parents as "Mr./Mrs." respectfully
- Answer questions directly using the student data provided

STUDENT INFORMATION:
{student_data}

SCHOOL INFORMATION:
- Phone: {self._phone}
- Hours: Monday to Friday, 9 AM to 4 PM
- Parent meetings: 9 AM to 12 PM
- Minimum attendance: 75%
- Below 75% may affect examination eligibility

KEY PHRASES:
- "Good morning/afternoon/evening"
- "Thank you for taking the time to speak with me"
- "I completely understand your concern"
- "Please don't hesitate to contact us"
- "Have a wonderful day!"
"""
        return {
            "client": self._groq_client,
            "messages": [{"role": "system", "content": system_prompt}]
        }

    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Start call using greeting module."""
        if not self._twilio_ready:
            return NotificationResult(success=False, error_message="Twilio not configured")

        # Store student data for conversation
        student_data = {
            'name': payload.student_name,
            'parent_name': payload.parent_name,
            'details': payload.details,
            'dimension': payload.dimension,
            'risk_level': payload.risk_level,
            'attendance_pct': payload.attendance_pct,
            'attendance_total': payload.attendance_total,
            'attendance_attended': payload.attendance_attended,
            'consecutive_absences': payload.consecutive_absences,
            'performance_grade': payload.performance_grade,
            'performance_remarks': payload.performance_remarks,
            'behavior_incidents': payload.behavior_incidents,
            'behavior_status': payload.behavior_status,
        }
        self._student_data[payload.registration] = student_data

        if self._ai_ready:
            chat = self._get_chat(student_data)
            self._conversations[payload.registration] = chat

        # USE PERSON 1'S GREETING MODULE
        message = self.greeting.get_greeting(payload)

        webhook_url = f"{self._ngrok_url}/handle-parent-response"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech"
            language="en-IN"
            action="{webhook_url}"
            method="POST"
            timeout="5"
            speechTimeout="auto"
            bargein="true">
        <Say voice="alice" language="en-IN">{message}</Say>
    </Gather>
    <Say voice="alice" language="en-IN">{self.closing.get_silent_ending_message()}</Say>
</Response>"""

        try:
            print(f"\nCalling {payload.parent_name}...")
            print(f"   Student: {payload.student_name}")
            print(f"   Language: English")

            call = self._client.calls.create(
                to=payload.to_number,
                from_=self._from_number,
                twiml=twiml
            )

            self.calls_made.append(payload)
            print(f"   SID: {call.sid}")
            print(f"   Parent can interrupt anytime!\n")
            return NotificationResult(success=True, sid=call.sid, student_id=payload.registration, channel="ai_2way")

        except Exception as e:
            print(f"   {str(e)[:100]}")
            return NotificationResult(success=False, error_message=str(e))

    def handle_response(self, registration: str, parent_speech: str) -> str:
        """Handle parent speech using conversation + closing modules."""
        chat = self._conversations.get(registration)
        student_data = self._student_data.get(registration, {})

        if not chat:
            return self.error_handler.get_misunderstanding_message()

        # Check if parent wants to end
        if self.conversation.should_end_conversation(parent_speech):
            return self.closing.get_closing_message(
                parent_name=student_data.get('parent_name', ''),
                student_name=student_data.get('name', ''),
                meeting_booked=True
            )

        # Normal conversation - USE PERSON 2'S MODULE
        response = self.conversation.handle_parent_speech(chat, parent_speech, student_data)

        # If AI didn't understand
        if not response or len(response) < 5:
            return self.error_handler.get_misunderstanding_message(registration)

        # Reset error count on successful understanding
        self.error_handler.reset_count(registration)

        return response

    def generate_followup_twiml(self, registration: str, parent_speech: str) -> str:
        """Generate next conversation turn."""
        ai_response = self.handle_response(registration, parent_speech)
        safe = html.escape(ai_response)
        webhook_url = f"{self._ngrok_url}/handle-parent-response"

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech"
            language="en-IN"
            action="{webhook_url}"
            method="POST"
            timeout="3"
            speechTimeout="auto"
            bargein="true">
        <Say voice="alice" language="en-IN">{safe}</Say>
    </Gather>
    <Say voice="alice" language="en-IN">{self.closing.get_silent_ending_message()}</Say>
</Response>"""

    def make_batch_calls(self, payloads: list) -> dict:
        """Make multiple AI calls."""
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        print(f"\n2-Way AI Calls (English): {len(payloads)}\n")
        for i, p in enumerate(payloads, 1):
            print(f"[{i}/{len(payloads)}] ", end="")
            r = self.make_call(p)
            results["details"].append({
                "student": p.student_name,
                "parent": p.parent_name,
                "success": r.success,
                "channel": "ai_2way",
                "language": "English"
            })
            if r.success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            time.sleep(2)
        print(f"\n{results['successful']}/{results['total']} calls completed\n")
        return results


class TwoWayDemoService:
    """Demo mode - English."""

    def __init__(self):
        self._school = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self.calls_made = []

    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Demonstrate English AI call."""
        self.calls_made.append(payload)
        p = payload.parent_name
        s = payload.student_name
        d = payload.details
        att = getattr(payload, 'attendance_pct', 'N/A')
        grade = getattr(payload, 'performance_grade', 'N/A')

        print(f"\n{'='*55}")
        print(f"  PRIYA - ENGLISH MODE")
        print(f"  {self._school}")
        print(f"  Data: Attendance={att}%, Grade={grade}")
        print(f"  Parent can interrupt anytime!")
        print(f"{'='*55}\n")
        print(f"  Calling {p}...\n")
        time.sleep(0.5)

        print(f"  Priya: \"Good morning, Mr./Mrs. {p}!\"")
        print(f"  Priya: \"This is Priya from {self._school}.\"")
        print(f"  Priya: \"I'm calling about your child, {s}.\"")
        print(f"  Priya: \"{d}\"")
        print(f"  Priya: \"Do you have a moment to discuss this?\"\n")
        time.sleep(1)

        print(f"  Parent interrupts: \"Yes, I'm listening!\"\n")
        time.sleep(0.5)

        print(f"  {p}: \"What's the minimum attendance required?\"\n")
        time.sleep(0.5)

        print(f"  Priya: \"The minimum attendance is 75%.\"")
        print(f"  Priya: \"Your child currently has {att}%.\"")
        print(f"  Priya: \"Would you like me to schedule a meeting?\"\n")
        time.sleep(1)

        print(f"  {p}: \"Yes please, tomorrow if possible.\"\n")
        time.sleep(0.5)

        print(f"  Priya: \"Absolutely! Tomorrow at 10:30 AM.\"")
        print(f"  Priya: \"Is there anything else I can help with?\"\n")
        time.sleep(0.5)

        print(f"  {p}: \"No, that's all. Thank you!\"\n")
        time.sleep(0.5)

        print(f"  Priya: \"Have a wonderful day! Goodbye!\"\n")

        print(f"  {'='*55}")
        print(f"  Call Complete | English")
        print(f"  {'='*55}\n")

        return NotificationResult(
            success=True,
            sid=f"ENG_DEMO_{len(self.calls_made):04d}",
            student_id=payload.registration,
            channel="ai_2way"
        )

    def make_batch_calls(self, payloads: list) -> dict:
        """Multiple demo calls."""
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        print(f"\nDemo Calls (English): {len(payloads)}\n")
        for p in payloads:
            r = self.make_call(p)
            results["successful"] += 1
            results["details"].append({
                "student": p.student_name,
                "parent": p.parent_name,
                "success": True,
                "channel": "ai_2way"
            })
        return results
