"""
call_flow/conversation_handler.py
──────────────────────────────────
PERSON 2: CONVERSATION MODULE - ENGLISH VERSION
"""
import logging
import random

logger = logging.getLogger(__name__)


class ConversationHandler:
    """Handles the conversation portion of the call in English."""
    
    def __init__(self, school_name, school_phone):
        self.school = school_name
        self.phone = school_phone
    
    def handle_parent_speech(self, chat, parent_speech: str, student_data: dict = None) -> str:
        """Process parent speech and generate AI response in English."""
        if not chat:
            return self._get_fallback_message()
        
        try:
            context = ""
            if student_data:
                context = self._build_context(student_data)
            
            prompt = f"{context}\nParent said: \"{parent_speech}\"\nRespond naturally in English. Keep it short (2-3 sentences)."
            
            response = chat.send_message(prompt)
            ai_reply = response.text.strip()
            
            logger.info(f"Parent: {parent_speech[:50]}... → AI: {ai_reply[:50]}...")
            return ai_reply
            
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            return self._get_error_message()
    
    def _build_context(self, student_data: dict) -> str:
        """Build English context for AI."""
        context = "STUDENT INFORMATION:\n"
        context += f"- Name: {student_data.get('name', 'N/A')}\n"
        if student_data.get('attendance_pct'):
            context += f"- Attendance: {student_data['attendance_pct']}%\n"
        if student_data.get('attendance_attended') and student_data.get('attendance_total'):
            context += f"- Classes: {student_data['attendance_attended']}/{student_data['attendance_total']}\n"
        if student_data.get('consecutive_absences'):
            context += f"- Consecutive absences: {student_data['consecutive_absences']} days\n"
        if student_data.get('performance_grade'):
            context += f"- Grade: {student_data['performance_grade']}\n"
        if student_data.get('behavior_incidents'):
            context += f"- Behavior incidents: {student_data['behavior_incidents']}\n"
        return context
    
    def should_end_conversation(self, parent_speech: str) -> bool:
        """Check if parent wants to end the conversation."""
        ending_phrases = [
            "that's all", "nothing else", "thank you", "thanks",
            "i'm done", "that's it", "no more", "bye", "goodbye",
            "okay", "alright", "got it", "understood", "that helps",
            "no thank you", "i'm good", "that's enough"
        ]
        speech_lower = parent_speech.lower()
        return any(phrase in speech_lower for phrase in ending_phrases)
    
    def get_followup_question(self) -> str:
        """Generate English follow-up question."""
        questions = [
            "Is there anything else I can help you with?",
            "Would you like to schedule a meeting with the teacher?",
            "Do you have any other questions?",
            "Would you like me to send the report via email?",
        ]
        return random.choice(questions)
    
    def _get_fallback_message(self) -> str:
        return (
            "I apologize, but I'm experiencing a technical issue. "
            f"Please contact the college at {self.phone}. Thank you."
        )
    
    def _get_error_message(self) -> str:
        return (
            "I heard what you said. "
            f"If you need further assistance, please contact the college at {self.phone}."
        )