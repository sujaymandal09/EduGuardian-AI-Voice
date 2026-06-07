"""
call_flow/greeting_handler.py
──────────────────────────────
PERSON 1: GREETING MODULE - ENGLISH VERSION

Handles the opening of the call:
- Warm, professional greeting
- Time-appropriate salutation
- Clear introduction
- Purpose of call
- Invitation to respond

Modify this file independently without affecting other modules.
"""
import datetime
import html
import random
import logging

logger = logging.getLogger(__name__)


class GreetingHandler:
    """
    Handles the greeting portion of every call.
    
    PERSON 1 - Your Responsibilities:
    - Design the opening message
    - Make it warm and professional
    - Handle different times of day
    - Handle different urgency levels
    - Handle different call reasons (attendance/performance/behavior)
    - Make the parent feel comfortable
    
    You can modify ANYTHING in this class.
    Just keep these method signatures unchanged:
        get_greeting(payload) -> str
        get_welcome_back_message() -> str
    """
    
    def __init__(self, school_name: str, school_phone: str):
        """
        Initialize the greeting handler.
        
        Args:
            school_name: Name of the school/college
            school_phone: Contact phone number
        """
        self.school = school_name
        self.phone = school_phone
        
        # Greeting variations for natural feel
        self._opening_phrases = [
            "I hope I'm not disturbing you.",
            "I hope this is a convenient time to talk.",
            "I hope you're doing well today.",
            "Thank you for taking my call.",
        ]
        
        logger.info(f"GreetingHandler initialized for {school_name}")
    
    def get_greeting(self, payload) -> str:
        """
        Generate the complete opening greeting message.
        
        This is the MAIN method called by the orchestrator.
        It builds the entire greeting that the AI speaks when the call connects.
        
        Args:
            payload: CallPayload object containing:
                - parent_name: Parent's name
                - student_name: Student's name
                - details: The concern details
                - risk_level: HIGH, MEDIUM, or LOW
                - dimension: attendance, performance, or behavior
                
        Returns:
            str: Complete greeting message to be spoken by AI
        """
        parent = html.escape(payload.parent_name)
        student = html.escape(payload.student_name)
        
        # Build greeting in parts
        parts = []
        
        # Part 1: Time greeting + Introduction
        parts.append(self._get_opening(parent))
        
        # Part 2: Reason for calling
        parts.append(self._get_reason(payload, student))
        
        # Part 3: Invitation to respond
        parts.append(self._get_invitation())
        
        # Combine all parts
        greeting = " ".join(parts)
        
        logger.info(f"Greeting generated for {parent} ({len(greeting)} chars)")
        return greeting
    
    def _get_opening(self, parent: str) -> str:
        """
        Build the opening: time greeting + introduction.
        
        Example: "Good morning, Mr./Mrs. Sharma. This is Priya calling from Sunrise College."
        """
        time_greet = self._get_time_greeting()
        
        # Random opening phrase for natural feel
        opening_phrase = random.choice(self._opening_phrases)
        
        return (
            f"{time_greet}, Mr./Mrs. {parent}. "
            f"This is Priya calling from {self.school}. "
            f"{opening_phrase}"
        )
    
    def _get_reason(self, payload, student: str) -> str:
        """
        Build the reason for calling.
        
        Different messages based on:
        - Urgency level (HIGH/MEDIUM/LOW)
        - Concern type (attendance/performance/behavior)
        """
        risk = payload.risk_level
        dimension = payload.dimension
        details = payload.details
        
        # Choose urgency phrase
        if risk == "HIGH":
            urgency = self._get_high_urgency_phrase(dimension)
        elif risk == "MEDIUM":
            urgency = self._get_medium_urgency_phrase(dimension)
        else:
            urgency = self._get_low_urgency_phrase(dimension)
        
        return (
            f"{urgency} your child, {student}. "
            f"{details} "
            f"I wanted to personally reach out and discuss this with you."
        )
    
    def _get_high_urgency_phrase(self, dimension: str) -> str:
        """Phrases for HIGH risk situations."""
        phrases = {
            "attendance": [
                "I need to bring to your attention an urgent matter regarding",
                "This is an important alert concerning",
                "I'm calling about a serious concern with",
            ],
            "performance": [
                "I need to discuss an urgent academic concern regarding",
                "This is an important academic alert about",
                "I'm calling about a serious academic matter with",
            ],
            "behavior": [
                "I need to bring to your attention an urgent behavioral concern regarding",
                "This is an important alert about",
                "I'm calling about a serious behavioral matter with",
            ],
        }
        return random.choice(phrases.get(dimension, phrases["attendance"]))
    
    def _get_medium_urgency_phrase(self, dimension: str) -> str:
        """Phrases for MEDIUM risk situations."""
        phrases = {
            "attendance": [
                "I wanted to discuss an important matter concerning",
                "I'm calling to talk about",
                "I wanted to bring something to your attention regarding",
            ],
            "performance": [
                "I wanted to discuss your child's academic progress with",
                "I'm calling about",
                "I wanted to talk about the academic performance of",
            ],
            "behavior": [
                "I wanted to discuss a behavioral concern with",
                "I'm calling to talk about",
                "I wanted to bring something to your attention regarding",
            ],
        }
        return random.choice(phrases.get(dimension, phrases["attendance"]))
    
    def _get_low_urgency_phrase(self, dimension: str) -> str:
        """Phrases for LOW risk situations."""
        phrases = {
            "attendance": [
                "I'm calling with an update about",
                "I wanted to share some information regarding",
                "I'm reaching out about",
            ],
            "performance": [
                "I'm calling with good news about",
                "I wanted to share positive feedback regarding",
                "I'm reaching out to discuss",
            ],
            "behavior": [
                "I'm calling with an update about",
                "I wanted to share some feedback regarding",
                "I'm reaching out about",
            ],
        }
        return random.choice(phrases.get(dimension, phrases["attendance"]))
    
    def _get_invitation(self) -> str:
        """
        Build the invitation for parent to respond.
        
        This ends the greeting and invites the parent to speak.
        """
        invitations = [
            "Do you have a moment to talk about this? I'm here to help and answer any questions you may have.",
            "Would you like to discuss this now? I'm here to assist you.",
            "Do you have time to talk? I'm ready to help with any questions.",
            "Can we discuss this briefly? I'm here to provide any information you need.",
        ]
        return random.choice(invitations)
    
    def _get_time_greeting(self) -> str:
        """
        Get time-appropriate greeting.
        
        Returns:
            - "Good morning" (before 12 PM)
            - "Good afternoon" (12 PM - 5 PM)
            - "Good evening" (after 5 PM)
        """
        hour = datetime.datetime.now().hour
        
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"
    
    def get_welcome_back_message(self) -> str:
        """
        Message when parent returns after being on hold or after a pause.
        
        Example: "Welcome back. I'm listening, please go ahead."
        """
        messages = [
            "Welcome back. I'm still here. Please go ahead.",
            "Thank you for holding. I'm listening.",
            "I'm still here. What would you like to know?",
            "Welcome back. How can I help you?",
        ]
        return random.choice(messages)
    
    def get_initial_greeting_without_details(self, parent: str) -> str:
        """
        Simple greeting without details - used for quick check-ins.
        
        Args:
            parent: Parent's name
            
        Returns:
            Short greeting message
        """
        time_greet = self._get_time_greeting()
        return (
            f"{time_greet}, Mr./Mrs. {html.escape(parent)}. "
            f"This is Priya from {self.school}. "
            f"How are you today?"
        )
    
    def get_follow_up_greeting(self, parent: str) -> str:
        """
        Greeting for follow-up calls (not first call).
        
        Args:
            parent: Parent's name
            
        Returns:
            Follow-up greeting
        """
        messages = [
            f"Hello again, Mr./Mrs. {html.escape(parent)}. This is Priya from {self.school}.",
            f"Good to speak with you again, Mr./Mrs. {html.escape(parent)}. Priya here from {self.school}.",
            f"Hi Mr./Mrs. {html.escape(parent)}, Priya from {self.school} calling again.",
        ]
        return random.choice(messages)
    
    def get_apology_for_calling_early(self) -> str:
        """Apology if calling very early or very late."""
        return (
            "I apologize if this is an unusual time to call. "
            "This is important, but I can call back if it's inconvenient."
        )
    
    def get_apology_for_calling_late(self) -> str:
        """Apology if calling late in the evening."""
        return (
            "I apologize for calling at this hour. "
            "If this isn't a good time, I can call back tomorrow."
        )


# ── TESTING ──────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test the greeting handler independently."""
    
    # Mock payload for testing
    class MockPayload:
        parent_name = "Sharma"
        student_name = "Aarav"
        risk_level = "HIGH"
        dimension = "attendance"
        details = "Your child has 65% attendance, which is below the required 75%."
    
    handler = GreetingHandler("Sunrise College", "033-4805-1910")
    
    # Test greeting
    greeting = handler.get_greeting(MockPayload())
    print("GREETING:")
    print(greeting)
    print()
    
    # Test time greetings
    print(f"Morning greeting: {handler._get_time_greeting()}")
    print(f"Welcome back: {handler.get_welcome_back_message()}")
    print(f"Follow-up: {handler.get_follow_up_greeting('Sharma')}")