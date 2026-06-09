"""
call_flow/conversation_handler.py
──────────────────────────────────
PERSON 2: CONVERSATION MODULE - ENGLISH ONLY VERSION
"""
import logging
import random
import time

logger = logging.getLogger(__name__)


class ConversationHandler:
    """Handles the conversation portion of the call in English only."""
    
    def __init__(self, school_name, school_phone):
        self.school = school_name
        self.phone = school_phone
    
    def handle_parent_speech(self, chat, parent_speech: str, student_data: dict = None) -> str:
        """Process parent speech and generate AI response in English only."""
        if not chat:
            return self._get_fallback_message()

        context = self._build_context(student_data) if student_data else ""
        user_message = (
            f"{context}\n"
            f"Parent said: \"{parent_speech}\"\n"
            f"Respond naturally in clear, professional English only. "
            f"Do not use any Hindi or regional language words (no ji, haan, accha, bilkul, theek hai, etc.). "
            f"Keep it short (2-3 sentences)."
        )

        # Append user turn to history
        chat["messages"].append({"role": "user", "content": user_message})

        for attempt in range(3):
            try:
                response = chat["client"].chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=chat["messages"]
                )
                ai_reply = response.choices[0].message.content.strip()
                # Append assistant reply so future turns have context
                chat["messages"].append({"role": "assistant", "content": ai_reply})
                logger.info(f"Parent: {parent_speech[:50]}... -> AI: {ai_reply[:50]}...")
                return ai_reply
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"Groq attempt {attempt + 1} failed ({e}), retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Conversation error after 3 attempts: {e}")
                    # Remove the unanswered user message so history stays clean
                    chat["messages"].pop()
                    return self._get_error_message()
    
    def _build_context(self, student_data: dict) -> str:
        """Build a concise student context block for the AI prompt."""
        lines = ["STUDENT INFORMATION:"]
        lines.append(f"- Name: {student_data.get('name', 'N/A')}")
        if student_data.get('details'):
            lines.append(f"- Summary: {student_data['details']}")
        if student_data.get('attendance_pct') is not None:
            lines.append(f"- Attendance: {student_data['attendance_pct']}%")
        if student_data.get('attendance_attended') and student_data.get('attendance_total'):
            lines.append(f"- Classes attended: {student_data['attendance_attended']}/{student_data['attendance_total']}")
        if student_data.get('consecutive_absences'):
            lines.append(f"- Consecutive absences: {student_data['consecutive_absences']} days")
        if student_data.get('performance_grade'):
            lines.append(f"- Grade: {student_data['performance_grade']}")
        if student_data.get('performance_remarks'):
            lines.append(f"- Remarks: {student_data['performance_remarks']}")
        if student_data.get('behavior_incidents'):
            lines.append(f"- Behaviour incidents: {student_data['behavior_incidents']}")
        if student_data.get('behavior_status'):
            lines.append(f"- Behaviour status: {student_data['behavior_status']}")
        return "\n".join(lines)

    def should_end_conversation(self, parent_speech: str) -> bool:
        """
        Return True only when the parent clearly signals they are done.

        Deliberately avoids short words like 'okay', 'thanks', 'got it'
        because parents use those mid-conversation all the time.
        """
        explicit_endings = [
            "that's all",
            "that is all",
            "nothing else",
            "no more questions",
            "no other questions",
            "i'm done",
            "i am done",
            "no, goodbye",
            "no, bye",
            "goodbye",
            "bye bye",
            "that's it, bye",
            "no thank you, goodbye",
            "no thank you, bye",
            "i have no more questions",
        ]
        speech_lower = parent_speech.lower().strip()
        # Exact short phrases only (avoid partial matches on "bye" inside longer words)
        if speech_lower in ("bye", "goodbye"):
            return True
        return any(phrase in speech_lower for phrase in explicit_endings)
    
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