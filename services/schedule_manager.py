"""
services/schedule_manager.py
─────────────────────────────
Manages teacher meeting slots for EduGuardian.

Responsibilities:
- Weekly rollover: on app startup, rolls next_week_bookings → current_week_bookings
  if the stored Monday date doesn't match this week's Monday.
- Slot lookup: find the nearest available slot from tomorrow, or for a named day.
- Booking: marks a slot booked and writes back to CSV immediately.

This module is NEVER imported during normal call flow.
It is only loaded when a parent agrees to a meeting during a Priya call.
"""

import csv
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CSV_PATH = os.path.join("data", "meeting_slots.csv")

# Day name → integer (Monday=0 … Saturday=5, Sunday=6)
DAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}


def _this_weeks_monday(ref: date = None) -> date:
    """Return the Monday of the ISO week that contains *ref* (default: today)."""
    ref = ref or date.today()
    return ref - timedelta(days=ref.weekday())


def _next_weeks_monday(ref: date = None) -> date:
    return _this_weeks_monday(ref) + timedelta(weeks=1)


class ScheduleManager:
    """Thread-safe-ish schedule manager (single-process Flask is fine)."""

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path

    # ── Public API ────────────────────────────────────────────────

    def reset_week_if_needed(self) -> None:
        """
        Called once at app startup.

        If the stored week_start_date in the CSV differs from this week's Monday:
          - current_week_bookings ← next_week_bookings
          - next_week_bookings    ← 0
          - week_start_date       ← this week's Monday
        Writes back immediately.
        """
        rows = self._load()
        if not rows:
            return

        stored_monday_str = rows[0].get("week_start_date", "")
        this_monday = _this_weeks_monday()

        try:
            stored_monday = date.fromisoformat(stored_monday_str)
        except ValueError:
            stored_monday = None

        if stored_monday == this_monday:
            return  # Nothing to do

        logger.info(
            f"[ScheduleManager] New week detected. Rolling over: "
            f"{stored_monday_str} → {this_monday.isoformat()}"
        )

        for row in rows:
            row["current_week_bookings"] = row["next_week_bookings"]
            row["next_week_bookings"] = "0"
            row["week_start_date"] = this_monday.isoformat()

        self._save(rows)

    def get_next_available_slot(
        self, prefer_next_week: bool = False
    ) -> Optional[dict]:
        """
        Return the nearest free slot starting from tomorrow.

        If *prefer_next_week* is True, only look at next_week_bookings counters.
        Skips Sunday entirely.
        If today is Saturday and prefer_next_week is False, skips Sunday and
        starts from next Monday (using next_week_bookings).
        """
        rows = self._load()
        today = date.today()
        tomorrow = today + timedelta(days=1)

        # If today is Saturday, "tomorrow" is Sunday — skip to Monday next week
        if tomorrow.weekday() == 6:  # Sunday
            tomorrow = tomorrow + timedelta(days=1)
            prefer_next_week = True

        # Try up to 7 days forward (skipping Sunday)
        for offset in range(7):
            candidate = tomorrow + timedelta(days=offset)
            if candidate.weekday() == 6:  # skip Sunday
                continue

            day_name = candidate.strftime("%A")  # e.g. "Monday"
            use_next = prefer_next_week or (candidate >= _next_weeks_monday())
            booking_col = "next_week_bookings" if use_next else "current_week_bookings"

            for row in rows:
                if row["day"].strip().lower() == day_name.lower():
                    booked = int(row[booking_col])
                    capacity = int(row["total_capacity"])
                    if booked < capacity:
                        return {
                            "day": day_name,
                            "start_time": row["start_time"],
                            "end_time": row["end_time"],
                            "use_next_week": use_next,
                            "date": candidate.isoformat(),
                        }
        return None  # No slots found in the next 7 days

    def get_available_slots_for_day(
        self, day_name: str, next_week: bool = False
    ) -> list[dict]:
        """
        Return all free slots for a named day (e.g. "Thursday").

        *next_week* controls which booking counter to check.
        """
        rows = self._load()
        booking_col = "next_week_bookings" if next_week else "current_week_bookings"
        slots = []

        for row in rows:
            if row["day"].strip().lower() == day_name.strip().lower():
                booked = int(row[booking_col])
                capacity = int(row["total_capacity"])
                if booked < capacity:
                    slots.append({
                        "day": row["day"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "use_next_week": next_week,
                    })
        return slots

    def book_slot(
        self, day: str, start_time: str, next_week: bool = False
    ) -> bool:
        """
        Mark a specific slot as booked and write back to CSV immediately.

        Returns True on success, False if slot not found or already full.
        """
        rows = self._load()
        booking_col = "next_week_bookings" if next_week else "current_week_bookings"

        for row in rows:
            if (
                row["day"].strip().lower() == day.strip().lower()
                and row["start_time"].strip() == start_time.strip()
            ):
                booked = int(row[booking_col])
                capacity = int(row["total_capacity"])
                if booked >= capacity:
                    logger.warning(
                        f"[ScheduleManager] Slot {day} {start_time} already full."
                    )
                    return False

                row[booking_col] = str(booked + 1)
                self._save(rows)
                logger.info(
                    f"[ScheduleManager] Booked: {day} {start_time} "
                    f"({'next' if next_week else 'current'} week)"
                )
                return True

        logger.warning(f"[ScheduleManager] Slot not found: {day} {start_time}")
        return False

    # ── Internal helpers ──────────────────────────────────────────

    def _load(self) -> list[dict]:
        """Load CSV rows as list of dicts."""
        if not os.path.exists(self.csv_path):
            logger.error(f"[ScheduleManager] CSV not found: {self.csv_path}")
            return []
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        except Exception as e:
            logger.error(f"[ScheduleManager] Load error: {e}")
            return []

    def _save(self, rows: list[dict]) -> None:
        """Write rows back to CSV, preserving column order."""
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        try:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            logger.error(f"[ScheduleManager] Save error: {e}")