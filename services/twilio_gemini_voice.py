"""
services/twilio_gemini_voice.py
────────────────────────────────
2-Way AI Voice - Gemini 2.0 Working
"""
import html
import logging
import os
import time
from google import genai
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)


class TwoWayAIVoiceService:
    """2-Way AI Voice with Gemini 2.0"""
    
    def __init__(self):
        self._twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self._twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self._from_number = os.getenv("TWILIO_FROM_NUMBER")
        self._gemini_key = os.getenv("GEMINI_API_KEY")
        self._ngrok_url = os.getenv("NGROK_URL", "")
        self._school = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        
        self._twilio_ready = all([self._twilio_sid, self._twilio_token, self._from_number])
        self._ai_ready = bool(self._gemini_key)
        
        if self._twilio_ready:
            from twilio.rest import Client
            self._client = Client(self._twilio_sid, self._twilio_token)
            print("✅ Twilio Connected")
        
        if self._ai_ready:
            self._genai_client = genai.Client(api_key=self._gemini_key)
            print("✅ Gemini 2.0 Connected")
        
        self._conversations = {}
        self.calls_made = []
    
    def _get_chat(self, student_name, parent_name):
        """Create a new chat with teacher personality."""
        return self._genai_client.chats.create(
            model="gemini-2.5-flash",
            config={
                "system_instruction": f"""
                You are Priya, a friendly teacher from {self._school}.
                You are talking to {parent_name} about their child {student_name}.
                Speak in natural Hinglish (Hindi + English).
                Be warm, helpful, and answer ANY question naturally.
                School phone: {self._phone}
                """
            }
        )
    
    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Start AI call."""
        if not self._twilio_ready:
            return NotificationResult(success=False, error_message="Twilio not configured")
        
        if self._ai_ready:
            chat = self._get_chat(payload.student_name, payload.parent_name)
            self._conversations[payload.registration] = chat
            print(f"✅ Chat: {payload.student_name}")
        
        safe_parent = html.escape(payload.parent_name)
        safe_student = html.escape(payload.student_name)
        webhook_url = f"{self._ngrok_url}/handle-parent-response"
        
        message = (
            f"Namaste {safe_parent} ji! "
            f"Main {self._school} se Priya bol rahi hoon. "
            f"Aapke bete {safe_student} ke baare mein baat karni thi. "
            f"{payload.details} "
            f"Aapko koi sawal hai toh poochiye."
        )
        
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">{message}</Say>
    <Gather input="speech" language="en-IN" action="{webhook_url}" method="POST" timeout="5" speechTimeout="auto">
    </Gather>
    <Say voice="alice" language="en-IN">Contact college: {self._phone}. Goodbye.</Say>
</Response>"""
        
        try:
            print(f"\n📞 Calling {payload.parent_name}...")
            call = self._client.calls.create(to=payload.to_number, from_=self._from_number, twiml=twiml)
            self.calls_made.append(payload)
            print(f"   ✅ SID: {call.sid}\n")
            return NotificationResult(success=True, sid=call.sid, student_id=payload.registration, channel="ai_2way")
        except Exception as e:
            print(f"   ❌ {str(e)[:100]}")
            return NotificationResult(success=False, error_message=str(e))
    
    def handle_response(self, registration: str, parent_speech: str) -> str:
        """Gemini responds to parent."""
        chat = self._conversations.get(registration)
        if not chat:
            return f"Contact college: {self._phone}"
        
        try:
            print(f"  🧠 Parent: \"{parent_speech[:60]}...\"")
            response = chat.send_message(parent_speech)
            ai_reply = response.text.strip()
            print(f"  ✅ AI: \"{ai_reply[:60]}...\"")
            return ai_reply
        except Exception as e:
            print(f"  ❌ {str(e)[:100]}")
            return f"Contact college: {self._phone}"
    
    def generate_followup_twiml(self, registration: str, parent_speech: str) -> str:
        """Generate next conversation turn."""
        ai_response = self.handle_response(registration, parent_speech)
        safe = html.escape(ai_response)
        webhook_url = f"{self._ngrok_url}/handle-parent-response"
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">{safe}</Say>
    <Gather input="speech" language="en-IN" action="{webhook_url}" method="POST" timeout="3" speechTimeout="auto">
    </Gather>
    <Say voice="alice" language="en-IN">Contact: {self._phone}. Goodbye.</Say>
</Response>"""
    
    def make_batch_calls(self, payloads: list) -> dict:
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        print(f"\n🤖 2-Way AI Calls: {len(payloads)}\n")
        for i, p in enumerate(payloads, 1):
            print(f"[{i}/{len(payloads)}] ", end="")
            r = self.make_call(p)
            results["details"].append({"student": p.student_name, "parent": p.parent_name, "success": r.success})
            if r.success: results["successful"] += 1
            else: results["failed"] += 1
            time.sleep(2)
        print(f"\n✅ {results['successful']}/{results['total']}\n")
        return results


class TwoWayDemoService:
    """Demo mode."""
    def __init__(self):
        self._school = os.getenv("SCHOOL_NAME", "Siliguri College")
        self._phone = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self.calls_made = []
    
    def make_call(self, payload: CallPayload) -> NotificationResult:
        self.calls_made.append(payload)
        p, s = payload.parent_name, payload.student_name
        print(f"\n{'═'*50}")
        print(f"  🤖 AI TEACHER DEMO")
        print(f"{'═'*50}")
        print(f"  📞 {p}: \"Namaste! Main {self._school} se Priya.\"")
        print(f"  📞 {p}: \"Aapke bete {s} ke baare mein...\"")
        print(f"  👤 Parent: \"Attendance kitna chahiye?\"")
        print(f"  🤖 AI: \"75% compulsory hai. Improve karne ka time hai.\"")
        print(f"  👤 Parent: \"Thank you!\"")
        print(f"  🤖 AI: \"Dhanyavad! Goodbye!\"")
        print(f"  ✅ Done\n")
        return NotificationResult(success=True, sid=f"DEMO_{len(self.calls_made)}", student_id=payload.registration, channel="ai_2way")
    
    def make_batch_calls(self, payloads: list) -> dict:
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        for p in payloads:
            r = self.make_call(p)
            results["successful"] += 1
            results["details"].append({"student": p.student_name, "parent": p.parent_name, "success": True})
        return results