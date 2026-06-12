"""
services/database.py
─────────────────────
PostgreSQL (Amazon RDS) database service for EduGuardian.

Replaces the SQLite version. All function signatures are identical —
nothing else in the codebase needs to change.

Connection is configured via .env:
    DB_HOST     — RDS endpoint
    DB_PORT     — 5432
    DB_NAME     — database name
    DB_USER     — master username
    DB_PASSWORD — master password

Tables are created automatically on first run via init_db().
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              SERIAL PRIMARY KEY,
    registration    TEXT        NOT NULL,
    student_name    TEXT        NOT NULL,
    parent_name     TEXT        NOT NULL,
    dimension       TEXT        NOT NULL,
    risk_level      TEXT        NOT NULL,
    call_date       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended           BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id          SERIAL PRIMARY KEY,
    call_id     INTEGER     NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    turn_order  INTEGER     NOT NULL
);

CREATE TABLE IF NOT EXISTS call_summaries (
    id              SERIAL PRIMARY KEY,
    call_id         INTEGER     UNIQUE NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    registration    TEXT        NOT NULL,
    student_name    TEXT        NOT NULL,
    parent_name     TEXT        NOT NULL,
    dimension       TEXT        NOT NULL,
    risk_level      TEXT        NOT NULL,
    summary         TEXT        NOT NULL,
    meeting_booked  TEXT,
    call_date       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# ── Connection ────────────────────────────────────────────────────

def _get_dsn() -> str:
    """Build the PostgreSQL connection string from environment variables."""
    return (
        f"host={os.getenv('DB_HOST')} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"dbname={os.getenv('DB_NAME')} "
        f"user={os.getenv('DB_USER')} "
        f"password={os.getenv('DB_PASSWORD')} "
        f"sslmode=require"
    )


@contextmanager
def _get_conn():
    """Yields an open psycopg2 connection with RealDictCursor."""
    conn = psycopg2.connect(_get_dsn())
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialise ────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create tables if they don't exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA)
        logger.info("[DB] PostgreSQL tables initialised (RDS: %s)", os.getenv("DB_HOST"))
        print(f"[DB] Connected to RDS: {os.getenv('DB_HOST')}")
    except Exception as e:
        logger.error("[DB] init_db failed: %s", e)
        print(f"[DB ERROR] Could not connect to RDS: {e}")
        raise


# ── Call lifecycle ─────────────────────────────────────────────────

def create_call_record(
    registration: str,
    student_name: str,
    parent_name: str,
    dimension: str,
    risk_level: str,
) -> int:
    """
    Insert a new call row when a call starts.
    Returns the new call_id (integer primary key).
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO calls
                   (registration, student_name, parent_name, dimension, risk_level)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id""",
                (registration, student_name, parent_name, dimension, risk_level),
            )
            call_id = cur.fetchone()[0]
    logger.info("[DB] Call record created: call_id=%d reg=%s", call_id, registration)
    return call_id


def mark_call_ended(call_id: int) -> None:
    """Mark the call as ended."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE calls SET ended=TRUE WHERE id=%s",
                (call_id,),
            )


# ── Conversation turns ────────────────────────────────────────────

def save_turn(call_id: int, role: str, content: str, turn_order: int) -> None:
    """
    Save one conversation turn.
    role: 'assistant' (Priya) or 'user' (parent)
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversation_turns
                   (call_id, role, content, turn_order)
                   VALUES (%s, %s, %s, %s)""",
                (call_id, role, content, turn_order),
            )


def get_conversation(call_id: int) -> list:
    """
    Return all turns for a call in order.
    [{"role": "assistant", "content": "..."}, ...]
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT role, content
                   FROM conversation_turns
                   WHERE call_id=%s
                   ORDER BY turn_order""",
                (call_id,),
            )
            rows = cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ── Summary ───────────────────────────────────────────────────────

def save_summary(
    call_id: int,
    registration: str,
    student_name: str,
    parent_name: str,
    dimension: str,
    risk_level: str,
    summary: str,
    meeting_booked: Optional[str],
    call_date: str,
) -> None:
    """
    Upsert the summary for a call.
    Uses ON CONFLICT to handle duplicate call_ids safely.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO call_summaries
                   (call_id, registration, student_name, parent_name,
                    dimension, risk_level, summary, meeting_booked)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (call_id) DO UPDATE SET
                       summary        = EXCLUDED.summary,
                       meeting_booked = EXCLUDED.meeting_booked""",
                (call_id, registration, student_name, parent_name,
                 dimension, risk_level, summary, meeting_booked),
            )
    logger.info("[DB] Summary saved: call_id=%d", call_id)


# ── Dashboard queries ─────────────────────────────────────────────

def get_all_summaries() -> list:
    """
    Return all summary rows newest first.
    Each row is a plain dict for the dashboard template.
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT id, call_id, registration, student_name, parent_name,
                          dimension, risk_level, summary, meeting_booked,
                          TO_CHAR(call_date AT TIME ZONE 'Asia/Kolkata',
                                  'YYYY-MM-DD"T"HH24:MI:SS') AS call_date
                   FROM call_summaries
                   ORDER BY call_date DESC"""
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def delete_summary(summary_id: int) -> bool:
    """
    Delete a summary and its parent call (CASCADE removes turns too).
    Returns True if a row was deleted.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM calls
                   WHERE id = (
                       SELECT call_id FROM call_summaries WHERE id = %s
                   )""",
                (summary_id,),
            )
            affected = cur.rowcount
    logger.info("[DB] Deleted summary_id=%d rows_affected=%d", summary_id, affected)
    return affected > 0