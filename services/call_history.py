"""Persistent call transcripts and summaries for PostgreSQL or local SQLite."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _mask_phone(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    return f"******{digits[-4:]}" if digits else ""


class NullCallHistoryRepository:
    """No-op storage used when CALL_HISTORY_ENABLED is false."""

    enabled = False

    def create_call(self, *args, **kwargs): return None
    def append_turn(self, *args, **kwargs): return None
    def update_status(self, *args, **kwargs): return None
    def update_meeting(self, *args, **kwargs): return None
    def claim_summary(self, *args, **kwargs): return False
    def save_summary(self, *args, **kwargs): return None
    def fail_summary(self, *args, **kwargs): return None
    def list_calls(self, *args, **kwargs): return []
    def get_call(self, *args, **kwargs): return None
    def get_turns(self, *args, **kwargs): return []
    def delete_calls(self, *args, **kwargs): return 0
    def reset_summary(self, *args, **kwargs): return False


class SqlCallHistoryRepository:
    """SQLAlchemy Core repository compatible with PostgreSQL and SQLite."""

    enabled = True

    def __init__(self, database_url: str):
        from sqlalchemy import (
            JSON, BigInteger, Boolean, Column, DateTime, ForeignKey, Integer,
            MetaData, String, Table, Text, UniqueConstraint, create_engine,
        )

        if database_url.startswith("postgres://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
        elif database_url.startswith("postgresql://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]

        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        if self.engine.url.get_backend_name() == "sqlite":
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        self.metadata = MetaData()
        self.calls = Table(
            "calls", self.metadata,
            Column("call_sid", String(64), primary_key=True),
            Column("registration", String(100), nullable=False, index=True),
            Column("student_name", String(200), nullable=False),
            Column("parent_name", String(200), nullable=False),
            Column("parent_phone_masked", String(32), nullable=False, default=""),
            Column("dimension", String(40), nullable=False, index=True),
            Column("risk_level", String(20), nullable=False, index=True),
            Column("call_status", String(30), nullable=False, default="initiated"),
            Column("started_at", DateTime(timezone=True), nullable=False),
            Column("answered_at", DateTime(timezone=True)),
            Column("ended_at", DateTime(timezone=True)),
            Column("duration_seconds", Integer),
            Column("summary_status", String(30), nullable=False, default="pending"),
            Column("brief_summary", Text),
            Column("parent_concerns", JSON, nullable=False, default=list),
            Column("school_observations", JSON, nullable=False, default=list),
            Column("agreed_actions", JSON, nullable=False, default=list),
            Column("unresolved_questions", JSON, nullable=False, default=list),
            Column("follow_up_required", Boolean, nullable=False, default=False),
            Column("parent_sentiment", String(40)),
            Column("meeting_status", String(30), nullable=False, default="none"),
            Column("meeting_event_id", String(255)),
            Column("meeting_start", DateTime(timezone=True)),
            Column("meeting_end", DateTime(timezone=True)),
            Column("summary_error", Text),
        )
        self.turns = Table(
            "conversation_turns", self.metadata,
            Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
            Column("call_sid", String(64), ForeignKey("calls.call_sid", ondelete="CASCADE"), nullable=False, index=True),
            Column("turn_number", Integer, nullable=False),
            Column("speaker", String(20), nullable=False),
            Column("message", Text, nullable=False),
            Column("interpreted_intent", String(60)),
            Column("conversation_stage", String(40)),
            Column("created_at", DateTime(timezone=True), nullable=False),
            UniqueConstraint("call_sid", "turn_number", name="uq_call_turn_number"),
        )
        self.metadata.create_all(self.engine)
        self._turn_lock = Lock()

    def create_call(self, call_sid: str, payload, opening_message: str) -> None:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy import insert

        values = {
            "call_sid": call_sid,
            "registration": payload.registration,
            "student_name": payload.student_name,
            "parent_name": payload.parent_name,
            "parent_phone_masked": _mask_phone(payload.to_number),
            "dimension": payload.dimension,
            "risk_level": payload.risk_level,
            "call_status": "initiated",
            "started_at": _utcnow(),
            "summary_status": "pending",
            "parent_concerns": [],
            "school_observations": [],
            "agreed_actions": [],
            "unresolved_questions": [],
            "follow_up_required": False,
            "meeting_status": "none",
        }
        with self.engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                statement = pg_insert(self.calls).values(**values).on_conflict_do_nothing(index_elements=["call_sid"])
            else:
                statement = insert(self.calls).values(**values).prefix_with("OR IGNORE")
            connection.execute(statement)
        if not self.get_turns(call_sid):
            self.append_turn(call_sid, "agent", opening_message, stage="intro")

    def append_turn(
        self, call_sid: str, speaker: str, message: str, *,
        intent: str | None = None, stage: str | None = None,
    ) -> None:
        from sqlalchemy import func, insert, select

        if not call_sid or not message:
            return
        with self._turn_lock, self.engine.begin() as connection:
            next_turn = connection.execute(
                select(func.coalesce(func.max(self.turns.c.turn_number), 0) + 1)
                .where(self.turns.c.call_sid == call_sid)
            ).scalar_one()
            connection.execute(insert(self.turns).values(
                call_sid=call_sid,
                turn_number=next_turn,
                speaker=speaker,
                message=message,
                interpreted_intent=intent,
                conversation_stage=stage,
                created_at=_utcnow(),
            ))

    def update_status(self, call_sid: str, status: str, duration: int | None = None) -> None:
        from sqlalchemy import update

        values: dict[str, Any] = {"call_status": status}
        if status == "answered":
            values["answered_at"] = _utcnow()
        if status in {"completed", "busy", "failed", "canceled", "no-answer"}:
            values["ended_at"] = _utcnow()
        if status in {"busy", "failed", "canceled", "no-answer"}:
            values["summary_status"] = "not_applicable"
        if duration is not None:
            values["duration_seconds"] = duration
        with self.engine.begin() as connection:
            connection.execute(update(self.calls).where(self.calls.c.call_sid == call_sid).values(**values))

    def update_meeting(self, call_sid: str, booking, status: str) -> None:
        from sqlalchemy import update

        values = {"meeting_status": status}
        if booking:
            values.update(
                meeting_event_id=booking.event_id,
                meeting_start=booking.start,
                meeting_end=booking.end,
            )
        elif status == "cancelled":
            values.update(meeting_event_id=None, meeting_start=None, meeting_end=None)
        with self.engine.begin() as connection:
            connection.execute(update(self.calls).where(self.calls.c.call_sid == call_sid).values(**values))

    def claim_summary(self, call_sid: str) -> bool:
        from sqlalchemy import update

        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.calls)
                .where(self.calls.c.call_sid == call_sid)
                .where(self.calls.c.summary_status.in_(["pending", "failed"]))
                .values(summary_status="generating", summary_error=None)
            )
            return result.rowcount == 1

    def save_summary(self, call_sid: str, summary: dict[str, Any]) -> None:
        from sqlalchemy import update

        values = {
            "summary_status": "completed",
            "brief_summary": summary.get("brief_summary", ""),
            "parent_concerns": summary.get("parent_concerns", []),
            "school_observations": summary.get("school_observations", []),
            "agreed_actions": summary.get("agreed_actions", []),
            "unresolved_questions": summary.get("unresolved_questions", []),
            "follow_up_required": bool(summary.get("follow_up_required", False)),
            "parent_sentiment": summary.get("parent_sentiment", "neutral"),
            "summary_error": None,
        }
        with self.engine.begin() as connection:
            connection.execute(update(self.calls).where(self.calls.c.call_sid == call_sid).values(**values))

    def fail_summary(self, call_sid: str, error: str) -> None:
        from sqlalchemy import update

        with self.engine.begin() as connection:
            connection.execute(
                update(self.calls).where(self.calls.c.call_sid == call_sid)
                .values(summary_status="failed", summary_error=error[:500])
            )

    def reset_summary(self, call_sid: str) -> bool:
        """Return an incomplete summary to pending so it can be regenerated."""
        from sqlalchemy import update

        with self.engine.begin() as connection:
            result = connection.execute(
                update(self.calls)
                .where(self.calls.c.call_sid == call_sid)
                .where(self.calls.c.summary_status.in_(["pending", "failed", "generating", "completed"]))
                .values(
                    summary_status="pending",
                    brief_summary=None,
                    parent_concerns=[],
                    school_observations=[],
                    agreed_actions=[],
                    unresolved_questions=[],
                    follow_up_required=False,
                    parent_sentiment=None,
                    summary_error=None,
                )
            )
            return result.rowcount == 1

    def delete_calls(self, call_sids: list[str]) -> int:
        """Delete selected calls; related transcript turns cascade in the database."""
        from sqlalchemy import delete

        unique_sids = list(dict.fromkeys(sid for sid in call_sids if sid))
        if not unique_sids:
            return 0
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(self.calls).where(self.calls.c.call_sid.in_(unique_sids))
            )
            return result.rowcount or 0

    def list_calls(self, limit: int = 100) -> list[dict[str, Any]]:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(self.calls).order_by(self.calls.c.started_at.desc()).limit(limit)
            )
            return [dict(row._mapping) for row in rows]

    def get_call(self, call_sid: str) -> dict[str, Any] | None:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            row = connection.execute(select(self.calls).where(self.calls.c.call_sid == call_sid)).first()
            return dict(row._mapping) if row else None

    def get_turns(self, call_sid: str) -> list[dict[str, Any]]:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(self.turns)
                .where(self.turns.c.call_sid == call_sid)
                .order_by(self.turns.c.turn_number)
            )
            return [dict(row._mapping) for row in rows]


_repository = None
_repository_lock = Lock()


def get_call_history_repository():
    global _repository
    with _repository_lock:
        if _repository is not None:
            return _repository
        enabled = os.getenv("CALL_HISTORY_ENABLED", "false").lower() in {"1", "true", "yes"}
        database_url = os.getenv("DATABASE_URL", "")
        if not enabled or not database_url:
            _repository = NullCallHistoryRepository()
            return _repository
        try:
            _repository = SqlCallHistoryRepository(database_url)
        except Exception:
            logger.exception("Call-history database initialization failed")
            raise
        return _repository
