"""Exercise the complete call-history repository without placing a real call."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    load_dotenv(ROOT / ".env")
    if os.getenv("CALL_HISTORY_ENABLED", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("CALL_HISTORY_ENABLED is not true in .env")

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    from sqlalchemy import delete

    from services.call_history import SqlCallHistoryRepository

    repository = SqlCallHistoryRepository(database_url)
    call_sid = f"VERIFY-{uuid4().hex}"
    payload = SimpleNamespace(
        registration="VERIFY-ONLY",
        student_name="Persistence Check",
        parent_name="Test Parent",
        to_number="+910000000000",
        dimension="PERFORMANCE",
        risk_level="MEDIUM",
    )

    try:
        repository.create_call(call_sid, payload, "This is a persistence check.")
        repository.update_status(call_sid, "answered")
        repository.append_turn(
            call_sid,
            "parent",
            "Yes, I can hear you.",
            intent="available_to_talk",
            stage="availability",
        )
        repository.append_turn(
            call_sid,
            "agent",
            "Thank you. This test is complete.",
            stage="conversation",
        )
        repository.update_status(call_sid, "completed", duration=5)
        if not repository.claim_summary(call_sid):
            raise RuntimeError("Could not claim the synthetic call summary")
        repository.save_summary(
            call_sid,
            {
                "brief_summary": "Synthetic persistence check completed.",
                "parent_concerns": [],
                "school_observations": [],
                "agreed_actions": [],
                "unresolved_questions": [],
                "follow_up_required": False,
                "parent_sentiment": "neutral",
            },
        )

        stored_call = repository.get_call(call_sid)
        stored_turns = repository.get_turns(call_sid)
        if not stored_call:
            raise RuntimeError("The synthetic call could not be read back")
        if stored_call["call_status"] != "completed":
            raise RuntimeError("Call status was not persisted")
        if stored_call["summary_status"] != "completed":
            raise RuntimeError("Call summary was not persisted")
        if len(stored_turns) != 3:
            raise RuntimeError(
                f"Expected 3 transcript turns but read back {len(stored_turns)}"
            )

        print("PASS: connection, call, transcript, status, and summary persistence work.")
    finally:
        with repository.engine.begin() as connection:
            connection.execute(
                delete(repository.turns).where(repository.turns.c.call_sid == call_sid)
            )
            connection.execute(
                delete(repository.calls).where(repository.calls.c.call_sid == call_sid)
            )
        print("Cleanup complete: the synthetic call was removed.")


if __name__ == "__main__":
    main()
