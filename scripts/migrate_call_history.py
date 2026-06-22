"""Safely install or repair the call-history PostgreSQL schema."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations" / "001_call_history.sql"


def _database_url() -> str:
    load_dotenv(ROOT / ".env")
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is missing from .env")
    if value.startswith("postgresql+psycopg://"):
        return "postgresql://" + value.removeprefix("postgresql+psycopg://")
    if value.startswith("postgres://"):
        return "postgresql://" + value.removeprefix("postgres://")
    return value


def _table_exists(cursor: psycopg.Cursor, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
        """,
        (table_name,),
    )
    return bool(cursor.fetchone()[0])


def _columns(cursor: psycopg.Cursor, table_name: str) -> set[str]:
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table_name,),
    )
    return {row[0] for row in cursor.fetchall()}


def _rename(cursor: psycopg.Cursor, old_name: str, new_name: str) -> None:
    cursor.execute(
        sql.SQL("ALTER TABLE {} RENAME TO {}").format(
            sql.Identifier(old_name), sql.Identifier(new_name)
        )
    )
    print(f"Preserved old table: {old_name} -> {new_name}")


def main() -> None:
    migration_sql = MIGRATION.read_text(encoding="utf-8")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with psycopg.connect(_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_schema()")
            database, schema = cursor.fetchone()
            print(f"Connected to database '{database}', schema '{schema}'.")

            calls_is_legacy = _table_exists(cursor, "calls") and "call_sid" not in _columns(
                cursor, "calls"
            )
            if calls_is_legacy:
                _rename(cursor, "calls", f"calls_legacy_{suffix}")
                if _table_exists(cursor, "conversation_turns"):
                    _rename(
                        cursor,
                        "conversation_turns",
                        f"conversation_turns_legacy_{suffix}",
                    )
            elif _table_exists(cursor, "conversation_turns"):
                expected = {"call_sid", "turn_number"}
                if not expected.issubset(_columns(cursor, "conversation_turns")):
                    _rename(
                        cursor,
                        "conversation_turns",
                        f"conversation_turns_legacy_{suffix}",
                    )

            cursor.execute(migration_sql)

            required_calls = {
                "call_sid",
                "student_name",
                "risk_level",
                "call_status",
            }
            required_turns = {"call_sid", "turn_number", "speaker", "message"}
            if not required_calls.issubset(_columns(cursor, "calls")):
                raise RuntimeError("The calls table failed schema verification")
            if not required_turns.issubset(_columns(cursor, "conversation_turns")):
                raise RuntimeError("The conversation_turns table failed schema verification")

        connection.commit()

    print("Call-history schema is ready.")


if __name__ == "__main__":
    main()
