"""
call_flow/closing_handler.py
─────────────────────────────
PERSON 3: CLOSING MODULE - ENGLISH VERSION
"""
import html


class ClosingHandler:
    """Handles the closing portion of the call in English."""
    
    def __init__(self, school_name, school_phone):
        self.school = school_name
        self.phone = school_phone
    
    def get_closing_message(self, parent_name: str, student_name: str = None, 
                           meeting_booked: bool = False, email_sent: bool = False) -> str:
        """Generate English closing message."""
        parent = html.escape(parent_name)
        closing = self._build_warm_closing(parent, student_name)
        
        if meeting_booked or email_sent:
            closing += self._get_summary(meeting_booked, email_sent)
        
        closing += self._get_final_goodbye()
        return closing
    
    def _build_warm_closing(self, parent, student):
        """Build warm English closing."""
        messages = [
            f"It was wonderful speaking with you, Mr./Mrs. {parent}!",
            f"I'm glad I could assist you today.",
        ]
        return " ".join(messages) + " "
    
    def _get_summary(self, meeting_booked, email_sent):
        """Get summary of actions taken."""
        summary = ""
        if meeting_booked:
            summary += "Your meeting has been scheduled. Please visit us tomorrow. "
        if email_sent:
            summary += "The report has been sent to your email. "
        return summary
    
    def _get_final_goodbye(self):
        """Get final English goodbye."""
        return (
            f"If you have any questions, please don't hesitate to call us at {self.phone}. "
            f"Our office is open Monday through Friday, 9 AM to 4 PM. "
            f"Have a wonderful day! Thank you! Goodbye!"
        )
    
    def get_silent_ending_message(self) -> str:
        """Message when parent doesn't respond."""
        return (
            f"I wasn't able to hear your response. "
            f"Please feel free to contact the college at {self.phone}. "
            f"Our office hours are Monday to Friday, 9 AM to 4 PM. "
            f"Thank you for your time. Have a wonderful day. Goodbye."
        )
    
    def get_repeat_message(self) -> str:
        """Message asking parent to repeat."""
        return "I didn't catch that clearly. Could you please repeat?"