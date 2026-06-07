"""
services/error_handler.py
──────────────────────────
ERROR HANDLER MODULE - ENGLISH VERSION
"""
import random


class ErrorHandler:
    """Handles misunderstandings during conversation in English."""
    
    def __init__(self, school_phone):
        self.phone = school_phone
        self.misunderstand_count = {}
    
    def get_misunderstanding_message(self, call_sid: str = None) -> str:
        """Get English message when AI didn't understand."""
        if call_sid:
            self.misunderstand_count[call_sid] = self.misunderstand_count.get(call_sid, 0) + 1
            count = self.misunderstand_count[call_sid]
        else:
            count = 1
        
        if count == 1:
            return "I didn't quite catch that. Could you please repeat?"
        elif count == 2:
            return "I'm sorry, I still didn't understand. Could you say that a bit more clearly?"
        elif count == 3:
            return (
                "It seems there might be a connection issue. "
                f"You can reach us at {self.phone}. "
                "I'd love to help, but I'm having trouble hearing you."
            )
        else:
            return (
                f"I apologize, but I'm unable to understand clearly. "
                f"Please visit or call the college at {self.phone}. Thank you."
            )
    
    def should_escalate(self, call_sid: str) -> bool:
        """Check if we should escalate."""
        return self.misunderstand_count.get(call_sid, 0) >= 3
    
    def get_no_speech_message(self) -> str:
        """Message when no speech detected."""
        messages = [
            "I didn't hear anything. Are you still there?",
            "I can't hear you. Could you please speak a bit louder?",
            "Are you there? I didn't catch what you said.",
        ]
        return random.choice(messages)
    
    def reset_count(self, call_sid: str):
        """Reset misunderstanding count."""
        if call_sid in self.misunderstand_count:
            self.misunderstand_count[call_sid] = 0