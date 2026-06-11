"""
services/summary_agent.py
──────────────────────────
Summary Agent for EduGuardian.

When a call ends:
1. Reads the full conversation from SQLite
2. Uses state.last_booked_slot as ground truth for meeting info (never guesses)
3. Asks Groq for a tight 2-3 sentence summary
4. Saves to call_summaries table
"""

import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


def _format_time_spoken(t: str) -> str:
    """Convert 24-hour 'HH:MM' to natural form: '10:30 AM', '2 PM'."""
    try:
        h, m = map(int, t.split(":"))
        period = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12} {period}" if m == 0 else f"{h12}:{m:02d} {period}"
    except Exception:
        return t


class SummaryAgent:

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, groq_client):
        self._groq = groq_client

    def generate_and_save(
        self,
        call_id: int,
        payload,
        last_booked_slot: dict = None,
    ) -> bool:
        """
        Full pipeline: read conversation → summarise → save.
        last_booked_slot is the authoritative source for meeting info.
        """
        from services.database import get_conversation, save_summary, mark_call_ended

        try:
            turns = get_conversation(call_id)
            if not turns:
                logger.warning("[SummaryAgent] No turns found for call_id=%d", call_id)
                return False

            # Ground truth — never let Groq guess this
            meeting = None
            if last_booked_slot:
                day  = last_booked_slot.get("day", "")
                time = _format_time_spoken(last_booked_slot.get("start_time", ""))
                meeting = f"{day} at {time}"

            summary_text = self._summarise(turns, payload, meeting)

            save_summary(
                call_id        = call_id,
                registration   = payload.registration,
                student_name   = payload.student_name,
                parent_name    = payload.parent_name,
                dimension      = payload.dimension,
                risk_level     = payload.risk_level,
                summary        = summary_text,
                meeting_booked = meeting,
                call_date      = datetime.now().isoformat(),
            )
            mark_call_ended(call_id)
            logger.info("[SummaryAgent] Done: call_id=%d meeting=%s", call_id, meeting)
            return True

        except Exception as e:
            logger.error("[SummaryAgent] Failed call_id=%d: %s", call_id, e)
            return False

    def _summarise(self, turns: list, payload, meeting: str) -> str:
        conversation_text = self._format_conversation(turns)

        meeting_fact = (
            f"A meeting WAS booked: {meeting}."
            if meeting else
            "No meeting was booked."
        )

        prompt = f"""You are writing a brief call log entry for a school teacher.

Student : {payload.student_name} (Reg: {payload.registration})
Parent  : {payload.parent_name}
Category: {payload.dimension.upper()} | Risk: {payload.risk_level}
Meeting : {meeting_fact}

TRANSCRIPT:
{conversation_text}

Write EXACTLY 2-3 sentences covering:
1. The concern raised and the parent's response/reason
2. What was decided or agreed
3. The meeting outcome (use the confirmed fact above — do not contradict it)

Rules: No bullet points. No headings. No filler phrases. Be direct and factual."""

        response = self._groq.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()

    def _format_conversation(self, turns: list) -> str:
        lines = []
        for t in turns:
            speaker = "Priya" if t["role"] == "assistant" else "Parent"
            lines.append(f"{speaker}: {t['content']}")
        return "\n".join(lines)