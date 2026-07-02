"""
call_flow/closing_handler.py
─────────────────────────────
PERSON 3: CLOSING MODULE - ENGLISH VERSION

Handles the end of the call:
- Warm goodbye messages
- Summary of actions taken (meeting booked, email sent)
- Silent ending when parent doesn't respond

Modify this file independently without affecting other modules.
Must keep: get_closing_message(parent_name, ...) -> str
           get_silent_ending_message() -> str
"""
import html
import random


class ClosingHandler:
    """
    Handles the closing portion of the call in English.

    PERSON 3 - Your Responsibilities:
    - Design the goodbye message
    - Summarise what was agreed (meeting, email, etc.)
    - Handle silent / no-response endings

    You can modify ANYTHING in this class.
    Just keep these method signatures unchanged:
        get_closing_message(parent_name, student_name, meeting_booked, email_sent) -> str
        get_silent_ending_message() -> str
    """

    def __init__(self, school_name: str, school_phone: str):
        self.school = school_name
        self.phone = school_phone

    def get_closing_message(self, parent_name: str, student_name: str = None,
                            meeting_booked: bool = False, email_sent: bool = False) -> str:
        """
        Generate the complete closing / goodbye message.

        Args:
            parent_name:    Parent's name
            student_name:   Student's name (optional)
            meeting_booked: True if a meeting was scheduled during the call
            email_sent:     True if a report was sent via email

        Returns:
            str: Complete closing message to be spoken by AI
        """
        parent = html.escape(parent_name)
        parts = [self._build_warm_closing(parent, student_name)]

        if meeting_booked or email_sent:
            parts.append(self._get_summary(meeting_booked, email_sent))

        parts.append(self._get_final_goodbye())
        return " ".join(parts)

    # ── private helpers ───────────────────────────────────────────

    def _build_warm_closing(self, parent: str, student: str = None) -> str:
        messages = [
            f"It was wonderful speaking with you, Mr./Mrs. {parent}!",
            f"Thank you so much for taking the time to speak with me, Mr./Mrs. {parent}.",
            f"I really appreciate your time and concern, Mr./Mrs. {parent}.",
        ]
        return random.choice(messages) + " I'm glad I could assist you today."

    def _get_summary(self, meeting_booked: bool, email_sent: bool) -> str:
        summary = ""
        if meeting_booked:
            summary += "Your meeting has been scheduled. Please visit us tomorrow. "
        if email_sent:
            summary += "The report has been sent to your email. "
        return summary.strip()

    def _get_final_goodbye(self) -> str:
        return (
            f"If you have any questions, please don't hesitate to call us at {self.phone}. "
            f"Our office is open Monday through Friday, 9 AM to 4 PM. "
            f"Have a wonderful day! Thank you! Goodbye!"
        )

    def get_silent_ending_message(self) -> str:
        """Message spoken when the parent doesn't respond (timeout)."""
        return (
            f"I wasn't able to hear your response. "
            f"Please feel free to contact the college at {self.phone}. "
            f"Our office hours are Monday to Friday, 9 AM to 4 PM. "
            f"Thank you for your time. Have a wonderful day. Goodbye."
        )

    def get_repeat_message(self) -> str:
        """Message asking parent to repeat themselves."""
        return "I didn't catch that clearly. Could you please repeat?"


# ── TESTING ──────────────────────────────────────────────────────

if __name__ == "__main__":
    handler = ClosingHandler("Sunrise College", "033-4805-1910")

    print("CLOSING (no actions):")
    print(handler.get_closing_message("Sharma"))
    print()

    print("CLOSING (meeting booked):")
    print(handler.get_closing_message("Sharma", "Aarav", meeting_booked=True))
    print()

    print("SILENT ENDING:")
    print(handler.get_silent_ending_message())
