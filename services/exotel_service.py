"""
services/exotel_service.py
───────────────────────────
Exotel Voice Call Service for Indian Numbers
"""
import logging
import os
import time
import requests
from requests.auth import HTTPBasicAuth
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)


class ExotelVoiceService:
    """
    Exotel voice call service for Indian parents.
    """
    
    def __init__(self):
        self._account_sid = os.getenv("EXOTEL_ACCOUNT_SID", "siliguricollege1")
        self._api_key = os.getenv("EXOTEL_API_KEY")
        self._api_token = os.getenv("EXOTEL_API_TOKEN")
        self._from_number = os.getenv("EXOTEL_FROM_NUMBER", "03348051910")
        self._school_name = os.getenv("SCHOOL_NAME", "Siliguri College")
        
        self._enabled = all([self._api_key, self._api_token, self._from_number])
        
        if self._enabled:
            # Exotel API URL
            self._base_url = f"https://api.exotel.com/v1/Accounts/{self._account_sid}"
            self._auth = HTTPBasicAuth(self._api_key, self._api_token)
            logger.info("✅ Exotel Voice Service Ready")
            print("✅ Exotel Voice Service Ready")
        else:
            logger.warning("⚠️ Exotel: Missing API Key or Token")
    
    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Make voice call to parent."""
        if not self._enabled:
            return NotificationResult(
                success=False,
                error_message="Exotel not configured. Check .env file",
                student_id=payload.registration,
                channel="voice"
            )
        
        # Clean phone number to 10 digits
        to_number = self._clean_number(payload.to_number)
        
        # Build short CallerId (max 11 chars)
        caller_id = "SiliguriCol"  # Exactly 11 characters
        
        # Build message text (will be spoken by Exotel's TTS)
        message = (
            f"Hello {payload.parent_name}. "
            f"This is Siliguri College. "
            f"We are calling about your child {payload.student_name}, "
            f"registration number {payload.registration}. "
            f"{payload.details} "
            f"Action required: {payload.recommended_action} "
            f"Please contact the college immediately. Thank you."
        )
        
        # Exotel API call data
        call_data = {
            "From": self._from_number,
            "To": to_number,
            "CallerId": caller_id,
            "CallType": "trans",
            "Url": "http://my.exotel.in/exoml/start_voice/your_template",
            "CustomField": payload.student_name
        }
        
        try:
            print(f"\n📞 Calling {payload.parent_name}...")
            print(f"   Number: {self._mask_number(to_number)}")
            print(f"   Student: {payload.student_name}")
            
            # Make API call
            response = requests.post(
                f"{self._base_url}/Calls/connect.json",
                data=call_data,
                auth=self._auth,
                timeout=30
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                call_sid = result.get('Call', {}).get('Sid', 'Unknown')
                
                logger.info(f"✅ Call initiated: {call_sid}")
                print(f"   ✅ SID: {call_sid}")
                
                return NotificationResult(
                    success=True,
                    sid=call_sid,
                    student_id=payload.registration,
                    channel="voice"
                )
            elif response.status_code == 403:
                print("   ❌ 403 Forbidden - Check API Key and Token")
                return NotificationResult(
                    success=False,
                    error_message="Authentication failed. Check credentials.",
                    student_id=payload.registration
                )
            elif response.status_code == 401:
                print("   ❌ 401 Unauthorized - Account not active")
                return NotificationResult(
                    success=False,
                    error_message="Account not active. Contact Exotel support.",
                    student_id=payload.registration
                )
            else:
                print(f"   ❌ Failed: {response.text[:200]}")
                return NotificationResult(
                    success=False,
                    error_message=f"Error {response.status_code}",
                    student_id=payload.registration
                )
                
        except requests.exceptions.Timeout:
            print("   ❌ Timeout")
            return NotificationResult(
                success=False,
                error_message="Request timeout",
                student_id=payload.registration
            )
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return NotificationResult(
                success=False,
                error_message=str(e),
                student_id=payload.registration
            )
    
    def make_batch_calls(self, payloads: list) -> dict:
        """Make calls to multiple parents."""
        results = {
            "total": len(payloads),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        print(f"\n📞 Exotel: Calling {len(payloads)} parents...\n")
        
        for i, payload in enumerate(payloads, 1):
            print(f"[{i}/{len(payloads)}] ", end="")
            
            result = self.make_call(payload)
            
            results["details"].append({
                "registration": payload.registration,
                "student": payload.student_name,
                "parent": payload.parent_name,
                "success": result.success,
                "sid": result.sid,
                "channel": "voice"
            })
            
            if result.success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            if i < len(payloads):
                time.sleep(2)  # Delay between calls
        
        print(f"\n✅ Done: {results['successful']}/{results['total']} successful")
        return results
    
    def _clean_number(self, number: str) -> str:
        """Clean to 10-digit format."""
        number = str(number).strip()
        number = number.replace(" ", "").replace("-", "").replace("+91", "")
        if number.startswith("91"):
            number = number[2:]
        if number.startswith("0"):
            number = number[1:]
        return number[-10:]
    
    @staticmethod
    def _mask_number(phone: str) -> str:
        """Mask number for privacy."""
        if len(phone) < 5:
            return "****"
        return phone[:2] + "****" + phone[-4:]


class MockVoiceService:
    """Mock service for testing without real calls."""
    
    def __init__(self):
        self.calls_made = []
    
    def make_call(self, payload: CallPayload) -> NotificationResult:
        """Simulate a voice call."""
        self.calls_made.append(payload)
        
        action = getattr(payload, 'recommended_action', 'Contact college')
        
        print(f"\n{'='*60}")
        print(f"  📞 MOCK CALL #{len(self.calls_made)}")
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
            sid=f"MOCK_{len(self.calls_made):04d}",
            student_id=payload.registration,
            channel="voice"
        )
    
    def make_batch_calls(self, payloads: list) -> dict:
        """Simulate batch calls."""
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        
        print(f"\n📞 Mock: Calling {len(payloads)} parents...\n")
        
        for payload in payloads:
            result = self.make_call(payload)
            results["successful"] += 1
            results["details"].append({
                "registration": payload.registration,
                "student": payload.student_name,
                "parent": payload.parent_name,
                "success": True,
                "sid": result.sid,
                "channel": "voice"
            })
        
        print(f"✅ All {results['total']} mock calls done\n")
        return results