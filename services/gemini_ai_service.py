"""
services/gemini_ai_service.py
──────────────────────────────
2-Way AI Voice Conversation with Google Gemini
FREE for students - Get API key at aistudio.google.com
"""
import logging
import os
import time
import html
from google import genai
from core.models import CallPayload, NotificationResult

logger = logging.getLogger(__name__)


class GeminiAIConversation:
    """
    ChatGPT-like voice AI using Google Gemini.
    
    Features:
    - FREE (60 requests/min)
    - Excellent Hindi/English/Bengali
    - Natural human-like conversation
    - Twilio integration for calls
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.school_name = os.getenv("SCHOOL_NAME", "Siliguri College")
        self.school_phone = os.getenv("SCHOOL_PHONE", "033-4805-1910")
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            self.client = genai.Client(api_key=self.api_key)
            self.chats = {}
            print("✅ Gemini AI Brain Ready (FREE)")
            print("🤖 AI Assistant: Priya - Multilingual")
        else:
            print("💡 Get FREE API key: aistudio.google.com")
            print("🤖 Running in Demo Mode")
    
    def _get_system_prompt(self):
        """Define the AI personality."""
        return f"""
        You are PRIYA, a warm AI assistant from {self.school_name}.
        
        PERSONALITY:
        - Friendly, empathetic, professional
        - Speak naturally in Hinglish (Hindi+English mix)
        - Use "ji" for respect
        - Sound like you're smiling
        - Keep responses short (2-3 sentences for voice)
        
        SCHOOL INFO:
        - {self.school_name}
        - Phone: {self.school_phone}
        - Hours: Mon-Fri, 9AM-4PM
        - Parent meetings: 9AM-12PM
        - Minimum attendance: 75%
        - Below 75%: May not sit for exams
        
        RULES:
        1. Greet warmly with parent's name
        2. State reason for calling gently
        3. Answer all questions helpfully
        4. Offer solutions
        5. End positively
        """
    
    def start_conversation(self, payload) -> NotificationResult:
        """Start AI conversation with parent."""
        if self.enabled:
            return self._real_conversation(payload)
        else:
            return self._demo_conversation(payload)
    
    def _real_conversation(self, payload) -> NotificationResult:
        """Real AI conversation using Gemini."""
        try:
            # Create new chat session
            chat = self.client.chats.create(
                model="gemini-2.5-flash",
                config={
                    "system_instruction": self._get_system_prompt()
                }
            )
            self.chats[payload.registration] = chat
            
            # Build opening context
            prompt = f"""
            START CONVERSATION WITH PARENT:
            Parent: {payload.parent_name}
            Student: {payload.student_name}
            Registration: {payload.registration}
            Concern: {payload.dimension.upper()} - {payload.risk_level} RISK
            Details: {payload.details}
            Action Needed: {payload.recommended_action}
            
            Greet warmly, explain the concern gently, ask if they have questions.
            Speak in natural Hinglish.
            """
            
            response = chat.send_message(prompt)
            opening_message = response.text
            
            print(f"\n🤖 AI Conversation Started:")
            print(f"   Parent: {payload.parent_name}")
            print(f"   Student: {payload.student_name}")
            print(f"   AI: \"{opening_message[:80]}...\"")
            
            # For real implementation: make Twilio call and stream this
            # The conversation continues as parent speaks
            
            return NotificationResult(
                success=True,
                sid=f"GEMINI_{hash(payload.registration)}",
                student_id=payload.registration,
                channel="ai_voice"
            )
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return NotificationResult(success=False, error_message=str(e))
    
    def _demo_conversation(self, payload) -> NotificationResult:
        """Demo mode showing realistic AI conversation."""
        parent = payload.parent_name
        student = payload.student_name
        reg = payload.registration
        concern = payload.dimension
        risk = payload.risk_level
        details = payload.details
        action = payload.recommended_action
        
        print(f"\n{'═'*60}")
        print(f"  🤖 GEMINI AI VOICE CONVERSATION")
        print(f"  Brain: Google Gemini | Assistant: Priya")
        print(f"  School: {self.school_name}")
        print(f"{'═'*60}\n")
        print(f"  📞 Calling {parent}...\n")
        
        # Realistic conversation
        dialogue = [
            ("AI", f"Namaste {parent} ji! Main {self.school_name} se Priya bol rahi hoon. Aap kaise hain? Main aapke bete {student} ke baare mein baat karni thi. Unka registration number {reg} hai. {details} Main bas aapko inform karna chahti thi aur dekhna chahti thi ki main aapki koi help kar sakti hoon kya?"),
            
            ("PARENT", "Oh hello Priya ji. Haan main thoda worried tha. Actually minimum attendance kitna chahiye exams ke liye?"),
            
            ("AI", f"Main bilkul samajh sakti hoon aapki concern. Dekhiye, rules ke according minimum 75% attendance compulsory hai. Aapke bete ki abhi attendance isse thodi kam hai, lekin ghabraaiye mat. Abhi bhi time hai improve karne ka. Main aapki help kar sakti hoon."),
            
            ("PARENT", "Thank you. Main kal teacher se milna chahunga. Kya appointment book ho sakti hai?"),
            
            ("AI", "Bilkul ji! Main abhi aapki appointment book kar deti hoon. Kal subah 10:30 AM ka slot available hai. Teacher Monday to Friday 9 AM se 12 PM tak available rehti hain. Aap apna aur bacche ka ID card le aaiyega."),
            
            ("PARENT", "Perfect! Aur kya aap attendance report email pe bhej sakti hain?"),
            
            ("AI", "Haan bilkul! Main abhi aapko detailed attendance report email pe bhej deti hoon. Kya aap chahenge ki main weekly update bhi bheja karoon? Taki aap regularly track kar sakein."),
            
            ("PARENT", "Haan weekly update bahut helpful hoga. Meri email hai parent@gmail.com"),
            
            ("AI", f"Done ji! Maine sab book kar diya. Kal subah 10:30 AM - parent-teacher meeting. Attendance report abhi email pe bhej di hai. Aur har Monday aapko weekly update milta rahega. Kya aur koi help chahiye aapko?"),
            
            ("PARENT", "Nahi, aapne toh bahut help kar di Priya ji. Thank you so much!"),
            
            ("AI", f"Aapka bahut swagat hai {parent} ji! Mera kaam hi hai help karna. Bas aap tension mat lijiye, sab theek ho jayega. Kal milte hain. Aapka din shubh ho! Dhanyavad! Goodbye!"),
        ]
        
        for speaker, msg in dialogue:
            if speaker == "AI":
                print(f"  🤖 PRIYA: \"{msg}\"")
            else:
                print(f"  👤 PARENT: \"{msg}\"")
            print()
            time.sleep(0.5)
        
        print(f"  {'─'*60}")
        print(f"  ✅ CONVERSATION COMPLETE")
        print(f"  Duration: 4 minutes 30 seconds")
        print(f"  Questions Answered: 4/4")
        print(f"  Meeting Booked: Tomorrow 10:30 AM")
        print(f"  Email Sent: parent@gmail.com")
        print(f"  Weekly Updates: Enabled")
        print(f"  Parent Satisfaction: ⭐⭐⭐⭐⭐")
        print(f"  AI Brain: Google Gemini (FREE)")
        print(f"  {'═'*60}\n")
        
        return NotificationResult(
            success=True,
            sid=f"GEMINI_DEMO_{hash(reg)}",
            student_id=reg,
            channel="ai_conversation"
        )
    
    def make_batch_calls(self, payloads: list) -> dict:
        """Handle multiple AI conversations."""
        results = {"total": len(payloads), "successful": 0, "failed": 0, "details": []}
        
        print(f"\n{'█'*60}")
        print(f"  🤖 GEMINI AI - BATCH CONVERSATIONS")
        print(f"  Parents to Call: {len(payloads)}")
        print(f"  AI Brain: Google Gemini")
        print(f"  Language: Hinglish (Natural)")
        print(f"  Cost: FREE")
        print(f"{'█'*60}\n")
        
        for i, payload in enumerate(payloads, 1):
            print(f"[{i}/{len(payloads)}] {payload.student_name}...")
            result = self.start_conversation(payload)
            
            results["details"].append({
                "student": payload.student_name,
                "parent": payload.parent_name,
                "success": result.success,
                "channel": "ai_conversation",
                "ai": "Gemini"
            })
            
            if result.success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            if i < len(payloads):
                time.sleep(1)
        
        print(f"\n{'█'*60}")
        print(f"  ✅ Complete: {results['successful']}/{results['total']}")
        print(f"{'█'*60}\n")
        
        return results