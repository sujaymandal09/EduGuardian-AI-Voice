"""
services/twilio_voice_service.py
────────────────────────────────
Twilio Voice Call Service with proper error handling.
"""
import html
import logging
import os
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)


class TwilioVoiceService:
    """
    Production Twilio voice call service.
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER env vars.
    """
    
    def __init__(self):
        self._account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self._auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self._from_number = os.getenv("TWILIO_FROM_NUMBER")
        self._enabled = all([self._account_sid, self._auth_token, self._from_number])
        
        if self._enabled:
            try:
                from twilio.rest import Client
                self._client = Client(self._account_sid, self._auth_token)
                logger.info("✅ Twilio Voice Service initialized")
            except ImportError:
                logger.warning("⚠️ Install twilio: pip install twilio")
                self._enabled = False
            except Exception as e:
                logger.error(f"❌ Twilio init failed: {e}")
                self._enabled = False
    
    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Make a real voice call to a parent using Twilio."""
        if not self._enabled:
            return NotificationResult(
                success=False,
                error_message="Twilio not configured",
                student_id=payload.registration,
                channel="voice"
            )
        
        # Escape XML special characters
        safe_parent = html.escape(payload.parent_name)
        safe_student = html.escape(payload.student_name)
        safe_registration = html.escape(payload.registration)
        safe_details = html.escape(payload.details)
        safe_action = html.escape(getattr(payload, 'recommended_action', 'Please contact the school'))
        risk_level = payload.risk_level
        
        # Voice settings
        voice = "alice"
        lang_map = {"en": "en-IN", "hi": "hi-IN", "bn": "bn-IN"}
        language = lang_map.get(payload.language, "en-IN")
        
        # Build detailed message based on dimension and risk level
        if payload.dimension == "attendance":
            if risk_level == "HIGH":
                urgency = "This is an URGENT attendance alert. "
                consequence = "If attendance does not improve immediately, your child may be disqualified from examinations. "
            elif risk_level == "MEDIUM":
                urgency = "This is an important attendance notification. "
                consequence = "Continued absences may affect your child's academic progress. "
            else:
                urgency = ""
                consequence = ""
            
            message = (
                f"Hello {safe_parent}. "
                f"This is an automated call from Sunrise School. "
                f"{urgency}"
                f"We are calling regarding your child, {safe_student}, "
                f"registration number {safe_registration}. "
                f"{safe_details} "
                f"{consequence}"
                f"You need to take the following action. {safe_action} "
                f"Please contact the school immediately. Thank you."
            )
        
        elif payload.dimension == "performance":
            if risk_level == "HIGH":
                urgency = "This is an URGENT academic performance alert. "
                consequence = "Without immediate intervention, your child may fail this academic year. "
            elif risk_level == "MEDIUM":
                urgency = "This is an important academic notification. "
                consequence = "Your child may need additional academic support. "
            else:
                urgency = ""
                consequence = ""
            
            message = (
                f"Hello {safe_parent}. "
                f"This is an automated call from Sunrise School. "
                f"{urgency}"
                f"We are calling regarding your child, {safe_student}, "
                f"registration number {safe_registration}. "
                f"{safe_details} "
                f"{consequence}"
                f"Recommended action. {safe_action} "
                f"Please visit the school to discuss your child's progress. Thank you."
            )
        
        elif payload.dimension == "behavior":
            if risk_level == "HIGH":
                urgency = "This is an URGENT behavior alert. "
                consequence = "Immediate parent involvement is required to prevent further disciplinary action. "
            elif risk_level == "MEDIUM":
                urgency = "This is an important behavior notification. "
                consequence = "Early intervention can help your child improve their conduct. "
            else:
                urgency = ""
                consequence = ""
            
            message = (
                f"Hello {safe_parent}. "
                f"This is an automated call from Sunrise School. "
                f"{urgency}"
                f"We are calling regarding your child, {safe_student}, "
                f"registration number {safe_registration}. "
                f"{safe_details} "
                f"{consequence}"
                f"Required action. {safe_action} "
                f"Your immediate attention is required. Thank you."
            )
        
        else:
            message = (
                f"Hello {safe_parent}. "
                f"This is regarding your child {safe_student}, "
                f"registration number {safe_registration}. "
                f"{safe_details} "
                f"Please contact the school. Thank you."
            )
        
        # Build TwiML
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="{voice}" language="{language}">{message}</Say>
    <Pause length="2"/>
    <Say voice="{voice}" language="{language}">
        To repeat. This call is regarding your child {safe_student}, 
        registration number {safe_registration}. 
        Please contact Sunrise School immediately. Thank you.
    </Say>
    <Pause length="1"/>
    <Say voice="{voice}" language="{language}">Goodbye.</Say>
</Response>"""
        
        try:
            masked_phone = self._mask_number(payload.to_number)
            logger.info(f"📞 Calling {payload.parent_name} at {masked_phone}")
            
            call = self._client.calls.create(
                to=payload.to_number,
                from_=self._from_number,
                twiml=twiml
            )
            
            logger.info(f"✅ Call initiated: SID={call.sid}")
            
            return NotificationResult(
                success=True,
                sid=call.sid,
                student_id=payload.registration,
                channel="voice"
            )
            
        except Exception as e:
            logger.error(f"❌ Call failed: {e}")
            return NotificationResult(
                success=False,
                error_message=str(e),
                student_id=payload.registration,
                channel="voice"
            )
    
    def make_batch_calls(self, payloads: list) -> dict:
        """Make calls to multiple parents."""
        results = {
            "total": len(payloads),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for i, payload in enumerate(payloads, 1):
            logger.info(f"Call {i}/{len(payloads)}: {payload.student_name}")
            result = self.make_call(payload)
            
            results["details"].append({
                "registration": payload.registration,
                "student": payload.student_name,
                "parent": payload.parent_name,
                "success": result.success,
                "sid": result.sid,
            })
            
            if result.success:
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    @staticmethod
    def _mask_number(phone: str) -> str:
        """Mask phone number for privacy."""
        if len(phone) < 5:
            return "****"
        return phone[:3] + "*" * (len(phone) - 7) + phone[-4:]


class MockVoiceService:
    """
    Mock voice service for testing without real Twilio calls.
    Simulates calls and prints what would have been sent.
    """
    
    def __init__(self):
        self.calls_made = []
    
    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Simulate a voice call."""
        self.calls_made.append(payload)
        
        action = getattr(payload, 'recommended_action', 'Contact school')
        
        print(f"\n{'='*60}")
        print(f"  📞 MOCK VOICE CALL #{len(self.calls_made)}")
        print(f"{'='*60}")
        print(f"  To:        {payload.parent_name}")
        print(f"  Phone:     {payload.to_number}")
        print(f"  Student:   {payload.student_name}")
        print(f"  Reg No:    {payload.registration}")
        print(f"  Alert:     {payload.dimension.upper()}")
        print(f"  Risk:      {payload.risk_level}")
        print(f"  Details:   {payload.details[:80]}...")
        print(f"  Action:    {action[:80]}...")
        print(f"{'='*60}\n")
        
        return NotificationResult(
            success=True,
            sid=f"MOCK_CALL_{len(self.calls_made):04d}",
            student_id=payload.registration,
            channel="voice"
        )
    
    def make_batch_calls(self, payloads: list) -> dict:
        """Simulate batch calls."""
        results = {
            "total": len(payloads),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        print(f"\n📞 Making {len(payloads)} mock calls...")
        
        for payload in payloads:
            result = self.make_call(payload)
            results["successful"] += 1
            results["details"].append({
                "registration": payload.registration,
                "student": payload.student_name,
                "parent": payload.parent_name,
                "success": True,
                "sid": result.sid
            })
        
        print(f"✅ All {results['total']} mock calls completed\n")
        return results