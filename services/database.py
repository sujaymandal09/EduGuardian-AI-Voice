"""
services/database.py
─────────────────────
SQLite database service for EduGuardian.

Responsibilities:
- Create and maintain the database schema on first run
- Save a new call record when a call starts
- Save each conversation turn as it happens
- Save the generated summary when the call ends
- Provide data for the dashboard (all summaries, delete)

Database file: data/eduguardian.db
Future: swap SQLite for Amazon RDS/Aurora by replacing this file only.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "eduguardian.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    registration    TEXT    NOT NULL,
    student_name    TEXT    NOT NULL,
    parent_name     TEXT    NOT NULL,
    dimension       TEXT    NOT NULL,
    risk_level      TEXT    NOT NULL,
    call_date       TEXT    NOT NULL,
    ended           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id     INTEGER NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    turn_order  INTEGER NOT NULL,
    FOREIGN KEY(call_id) REFERENCES calls(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS call_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id         INTEGER UNIQUE NOT NULL,
    registration    TEXT    NOT NULL,
    student_name    TEXT    NOT NULL,
    parent_name     TEXT    NOT NULL,
    dimension       TEXT    NOT NULL,
    risk_level      TEXT    NOT NULL,
    summary         TEXT    NOT NULL,
    meeting_booked  TEXT,
    call_date       TEXT    NOT NULL,
    FOREIGN KEY(call_id) REFERENCES calls(id) ON DELETE CASCADE
);
"""


@contextmanager
def _get_conn():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _get_conn() as conn:
        conn.executescript(_SCHEMA)
    logger.info("[DB] Initialised: %s", DB_PATH)


def create_call_record(registration, student_name, parent_name, dimension, risk_level) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO calls (registration, student_name, parent_name, dimension, risk_level, call_date, ended) VALUES (?,?,?,?,?,?,0)",
            (registration, student_name, parent_name, dimension, risk_level, datetime.now().isoformat()),
        )
        call_id = cur.lastrowid
    logger.info("[DB] Call record created: call_id=%d reg=%s", call_id, registration)
    return call_id


def mark_call_ended(call_id: int) -> None:
    with _get_conn() as conn:
        conn.execute("UPDATE calls SET ended=1 WHERE id=?", (call_id,))


def save_turn(call_id: int, role: str, content: str, turn_order: int) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO conversation_turns (call_id, role, content, turn_order) VALUES (?,?,?,?)",
            (call_id, role, content, turn_order),
        )


def get_conversation(call_id: int) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversation_turns WHERE call_id=? ORDER BY turn_order",
            (call_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_summary(call_id, registration, student_name, parent_name,
                 dimension, risk_level, summary, meeting_booked, call_date) -> None:
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO call_summaries
               (call_id, registration, student_name, parent_name, dimension, risk_level,
                summary, meeting_booked, call_date)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(call_id) DO UPDATE SET
                   summary=excluded.summary, meeting_booked=excluded.meeting_booked""",
            (call_id, registration, student_name, parent_name, dimension, risk_level,
             summary, meeting_booked, call_date),
        )
    logger.info("[DB] Summary saved: call_id=%d", call_id)


def get_all_summaries() -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT id, call_id, registration, student_name, parent_name,
                      dimension, risk_level, summary, meeting_booked, call_date
               FROM call_summaries ORDER BY call_date DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def delete_summary(summary_id: int) -> bool:
    with _get_conn() as conn:
        affected = conn.execute(
            "DELETE FROM calls WHERE id = (SELECT call_id FROM call_summaries WHERE id=?)",
            (summary_id,),
        ).rowcount
    return affected > 0