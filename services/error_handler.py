"""
services/error_handler.py
──────────────────────────
ERROR HANDLER MODULE - ENGLISH ONLY VERSION

One clarification attempt maximum.
If still unclear, transitions to a polite closing — no repeat loops.
"""


class ErrorHandler:
    """Handles misunderstandings during conversation in English only."""

    def __init__(self, school_phone):
        self.phone = school_phone
        self.misunderstand_count = {}

    def get_misunderstanding_message(self, call_sid: str = None) -> str:
        """
        Return an appropriate message based on how many times in a row
        the AI has failed to understand the parent.

        Count 1 → single short clarification (no question, just a nudge).
        Count 2+ → graceful closing statement. No further retry.
        """
        if call_sid:
            self.misunderstand_count[call_sid] = (
                self.misunderstand_count.get(call_sid, 0) + 1
            )
            count = self.misunderstand_count[call_sid]
        else:
            count = 1

        if count == 1:
            # One attempt — brief, not a repeated question
            return (
                "I'm sorry, I couldn't catch that clearly. "
                "Please go ahead whenever you're ready."
            )
        else:
            # Second failure — close gracefully, no more retries
            self.misunderstand_count.pop(call_sid, None)   # clean up
            return (
                "I wasn't able to hear the response clearly, "
                "so I'll conclude here for now. "
                f"Please feel free to contact us at {self.phone}. "
                "Thank you for your time. Have a good day."
            )

    def should_escalate(self, call_sid: str) -> bool:
        """
        Return True after just ONE failed attempt — the next message
        from get_misunderstanding_message will already be a closing,
        so the caller can use this flag to skip re-gathering.
        """
        return self.misunderstand_count.get(call_sid, 0) >= 2

    def get_no_speech_message(self) -> str:
        """Message when no speech is detected (first occurrence only)."""
        return "I'm sorry, I couldn't catch that. Please go ahead whenever you're ready."

    def get_graceful_close(self) -> str:
        """Standalone closing used when the system decides to end the call."""
        return (
            "I wasn't able to hear the response clearly, so I'll conclude here for now. "
            f"Please feel free to contact us at {self.phone}. "
            "Thank you for your time. Have a good day."
        )

    def reset_count(self, call_sid: str):
        """Reset misunderstanding count after a successful exchange."""
        self.misunderstand_count.pop(call_sid, None)
