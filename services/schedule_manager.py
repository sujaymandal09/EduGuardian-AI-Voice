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
WEEKLY_CSV_PATH = os.path.join("data", "Weekly_Schedule.csv")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

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


def _normalise_time(t: str) -> str:
    """
    Convert informal 12-h time string to zero-padded 24-h HH:MM.
    Hours 1-7 are treated as PM (13:xx-19:xx); 8-12 are kept as-is.
    '9:00' → '09:00', '1:15' → '13:15', '2:30' → '14:30'.
    """
    parts = t.strip().split(":")
    if len(parts) != 2:
        return t.strip()
    hour = int(parts[0])
    minute = parts[1].strip().zfill(2)
    if 1 <= hour <= 7:
        hour += 12
    return f"{hour:02d}:{minute}"


def sync_from_weekly_csv(
    weekly_path: str = WEEKLY_CSV_PATH,
    slots_path: str = CSV_PATH,
) -> None:
    """
    Reads Weekly_Schedule.csv (FREE cells = bookable parent-meeting slots) and
    rebuilds meeting_slots.csv, preserving existing current/next booking counts.

    Call once at startup, before reset_week_if_needed().
    """
    if not os.path.exists(weekly_path):
        logger.warning(f"[ScheduleManager] Weekly_Schedule.csv not found: {weekly_path}")
        return

    # Load existing rows so we can carry over booking counts
    existing: dict[tuple, dict] = {}
    if os.path.exists(slots_path):
        try:
            with open(slots_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    key = (row["day"].strip(), row["start_time"].strip())
                    existing[key] = row
        except Exception as e:
            logger.error(f"[ScheduleManager] Could not read existing slots: {e}")

    today_monday = _this_weeks_monday().isoformat()
    new_rows: list[dict] = []

    try:
        with open(weekly_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                time_range = raw_row.get("Time", "").strip()
                if not time_range:
                    continue
                parts = [p.strip() for p in time_range.split(" - ")]
                if len(parts) != 2:
                    continue
                start_time = _normalise_time(parts[0])
                end_time   = _normalise_time(parts[1])

                for day in DAYS:
                    cell = raw_row.get(day, "").strip().upper()
                    if cell != "FREE":
                        continue
                    key = (day, start_time)
                    old = existing.get(key, {})
                    new_rows.append({
                        "day":                   day,
                        "start_time":            start_time,
                        "end_time":              end_time,
                        "total_capacity":        old.get("total_capacity",        "1"),
                        "current_week_bookings": old.get("current_week_bookings", "0"),
                        "next_week_bookings":    old.get("next_week_bookings",    "0"),
                        "week_start_date":       old.get("week_start_date",       today_monday),
                    })
    except Exception as e:
        logger.error(f"[ScheduleManager] sync_from_weekly_csv read error: {e}")
        return

    day_order = {d: i for i, d in enumerate(DAYS)}
    new_rows.sort(key=lambda r: (day_order.get(r["day"], 99), r["start_time"]))

    if not new_rows:
        logger.warning("[ScheduleManager] No FREE slots found in Weekly_Schedule.csv — meeting_slots.csv unchanged.")
        return

    fieldnames = ["day", "start_time", "end_time", "total_capacity",
                  "current_week_bookings", "next_week_bookings", "week_start_date"]
    try:
        with open(slots_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        logger.info(f"[ScheduleManager] Synced {len(new_rows)} FREE slots from Weekly_Schedule.csv")
    except Exception as e:
        logger.error(f"[ScheduleManager] sync_from_weekly_csv write error: {e}")


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
        Return the nearest free slot starting from today.

        For today's slots, only slots whose start_time is still in the future
        are considered (already-started slots are skipped).
        If today is Saturday, jumps to Monday next week (skipping Sunday).
        If *prefer_next_week* is True, only look at next_week_bookings counters.
        """
        rows = self._load()
        today = date.today()
        now_str = datetime.now().strftime("%H:%M")

        # If today is Saturday, tomorrow is Sunday — jump straight to Monday
        start = today
        if today.weekday() == 5:  # Saturday
            start = today + timedelta(days=2)
            prefer_next_week = True

        # Scan up to 8 days (including today), skipping Sunday
        for offset in range(8):
            candidate = start + timedelta(days=offset)
            if candidate.weekday() == 6:  # skip Sunday
                continue

            day_name = candidate.strftime("%A")
            is_today = (candidate == today)
            use_next = prefer_next_week or (candidate >= _next_weeks_monday())
            booking_col = "next_week_bookings" if use_next else "current_week_bookings"

            for row in rows:
                if row["day"].strip().lower() != day_name.lower():
                    continue
                # For today: skip slots that have already started
                if is_today and row["start_time"].strip() <= now_str:
                    continue
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
        return None  # No slots found in the next 8 days

    def get_today_available_slot(self) -> Optional[dict]:
        """
        Return the first free slot for today whose start_time is still in the future.
        Uses current_week_bookings (today is always the current week).
        Returns None if today has no upcoming free slots.
        """
        rows = self._load()
        today = date.today()
        day_name = today.strftime("%A")
        now_str = datetime.now().strftime("%H:%M")

        for row in rows:
            if row["day"].strip().lower() != day_name.lower():
                continue
            if row["start_time"].strip() <= now_str:
                continue  # slot has already started or passed
            booked = int(row["current_week_bookings"])
            capacity = int(row["total_capacity"])
            if booked < capacity:
                return {
                    "day": day_name,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "use_next_week": False,
                    "date": today.isoformat(),
                }
        return None

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

    def cancel_slot(
        self, day: str, start_time: str, next_week: bool = False
    ) -> bool:
        """
        Decrement the booking counter for a slot (reverses a prior book_slot call).
        Returns True on success, False if slot not found or already at zero.
        """
        rows = self._load()
        booking_col = "next_week_bookings" if next_week else "current_week_bookings"

        for row in rows:
            if (
                row["day"].strip().lower() == day.strip().lower()
                and row["start_time"].strip() == start_time.strip()
            ):
                booked = int(row[booking_col])
                if booked <= 0:
                    logger.warning(
                        f"[ScheduleManager] Cannot cancel {day} {start_time} — already at 0."
                    )
                    return False
                row[booking_col] = str(booked - 1)
                self._save(rows)
                logger.info(
                    f"[ScheduleManager] Cancelled: {day} {start_time} "
                    f"({'next' if next_week else 'current'} week)"
                )
                return True

        logger.warning(f"[ScheduleManager] Cancel: slot not found: {day} {start_time}")
        return False

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