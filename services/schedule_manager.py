"""
services/schedule_manager.py
─────────────────────────────
Manages teacher meeting slots for EduGuardian.
"""

import csv
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CSV_PATH        = os.path.join("data", "meeting_slots.csv")
WEEKLY_CSV_PATH = os.path.join("data", "Weekly_Schedule.csv")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

DAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}


def _this_weeks_monday(ref: date = None) -> date:
    ref = ref or date.today()
    return ref - timedelta(days=ref.weekday())


def _next_weeks_monday(ref: date = None) -> date:
    return _this_weeks_monday(ref) + timedelta(weeks=1)


def _normalise_time(t: str) -> str:
    parts = t.strip().split(":")
    if len(parts) != 2:
        return t.strip()
    hour   = int(parts[0])
    minute = parts[1].strip().zfill(2)
    if 1 <= hour <= 7:
        hour += 12
    return f"{hour:02d}:{minute}"


def sync_from_weekly_csv(
    weekly_path: str = WEEKLY_CSV_PATH,
    slots_path:  str = CSV_PATH,
) -> None:
    if not os.path.exists(weekly_path):
        logger.warning(f"[ScheduleManager] Weekly_Schedule.csv not found: {weekly_path}")
        return

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
        logger.warning("[ScheduleManager] No FREE slots found in Weekly_Schedule.csv")
        return

    fieldnames = ["day", "start_time", "end_time", "total_capacity",
                  "current_week_bookings", "next_week_bookings", "week_start_date"]
    try:
        with open(slots_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        logger.info(f"[ScheduleManager] Synced {len(new_rows)} FREE slots")
    except Exception as e:
        logger.error(f"[ScheduleManager] sync_from_weekly_csv write error: {e}")


class ScheduleManager:

    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path

    # ── Public API ─────────────────────────────────────────────────

    def reset_week_if_needed(self) -> None:
        rows = self._load()
        if not rows:
            return

        stored_monday_str = rows[0].get("week_start_date", "")
        this_monday       = _this_weeks_monday()

        try:
            stored_monday = date.fromisoformat(stored_monday_str)
        except ValueError:
            stored_monday = None

        if stored_monday == this_monday:
            return

        logger.info(f"[ScheduleManager] New week — rolling over: {stored_monday_str} → {this_monday}")
        for row in rows:
            row["current_week_bookings"] = row["next_week_bookings"]
            row["next_week_bookings"]    = "0"
            row["week_start_date"]       = this_monday.isoformat()
        self._save(rows)

    def get_next_available_slot(self, prefer_next_week: bool = False) -> Optional[dict]:
        """
        Return the nearest free slot.

        prefer_next_week=True  → start scanning from next Monday (not today).
                                  Used when parent says 'next week'.
        prefer_next_week=False → start from today/tomorrow as normal.

        FIX: previously, prefer_next_week=True still started from today,
        which meant it could return a slot from the current week while
        incorrectly incrementing the next_week_bookings counter.
        """
        rows  = self._load()
        today = date.today()
        now_str = datetime.now().strftime("%H:%M")

        if prefer_next_week:
            # Always start from next Monday — never return a slot from this week
            start            = _next_weeks_monday()
            prefer_next_week = True   # keep True so booking_col stays next_week_bookings
        elif today.weekday() == 5:    # Today is Saturday → next slot is Monday next week
            start            = today + timedelta(days=2)
            prefer_next_week = True
        else:
            start = today

        for offset in range(14):     # scan up to 2 weeks ahead
            candidate = start + timedelta(days=offset)
            if candidate.weekday() == 6:   # skip Sunday
                continue

            day_name    = candidate.strftime("%A")
            is_today    = (candidate == today)
            use_next    = prefer_next_week or (candidate >= _next_weeks_monday())
            booking_col = "next_week_bookings" if use_next else "current_week_bookings"

            for row in rows:
                if row["day"].strip().lower() != day_name.lower():
                    continue
                if is_today and row["start_time"].strip() <= now_str:
                    continue
                booked   = int(row[booking_col])
                capacity = int(row["total_capacity"])
                if booked < capacity:
                    return {
                        "day":          day_name,
                        "start_time":   row["start_time"],
                        "end_time":     row["end_time"],
                        "use_next_week": use_next,
                        "date":         candidate.isoformat(),
                    }
        return None

    def get_today_available_slot(self) -> Optional[dict]:
        rows    = self._load()
        today   = date.today()
        day_name = today.strftime("%A")
        now_str  = datetime.now().strftime("%H:%M")

        for row in rows:
            if row["day"].strip().lower() != day_name.lower():
                continue
            if row["start_time"].strip() <= now_str:
                continue
            booked   = int(row["current_week_bookings"])
            capacity = int(row["total_capacity"])
            if booked < capacity:
                return {
                    "day":          day_name,
                    "start_time":   row["start_time"],
                    "end_time":     row["end_time"],
                    "use_next_week": False,
                    "date":         today.isoformat(),
                }
        return None

    def get_available_slots_for_day(self, day_name: str, next_week: bool = False) -> list[dict]:
        rows        = self._load()
        booking_col = "next_week_bookings" if next_week else "current_week_bookings"
        slots       = []
        for row in rows:
            if row["day"].strip().lower() == day_name.strip().lower():
                booked   = int(row[booking_col])
                capacity = int(row["total_capacity"])
                if booked < capacity:
                    slots.append({
                        "day":          row["day"],
                        "start_time":   row["start_time"],
                        "end_time":     row["end_time"],
                        "use_next_week": next_week,
                    })
        return slots

    def book_slot(self, day: str, start_time: str, next_week: bool = False) -> bool:
        rows        = self._load()
        booking_col = "next_week_bookings" if next_week else "current_week_bookings"
        for row in rows:
            if (row["day"].strip().lower()  == day.strip().lower()
                    and row["start_time"].strip() == start_time.strip()):
                booked   = int(row[booking_col])
                capacity = int(row["total_capacity"])
                if booked >= capacity:
                    logger.warning(f"[ScheduleManager] Slot {day} {start_time} already full.")
                    return False
                row[booking_col] = str(booked + 1)
                self._save(rows)
                logger.info(f"[ScheduleManager] Booked: {day} {start_time} "
                            f"({'next' if next_week else 'current'} week)")
                return True
        logger.warning(f"[ScheduleManager] Slot not found: {day} {start_time}")
        return False

    def cancel_slot(self, day: str, start_time: str, next_week: bool = False) -> bool:
        rows        = self._load()
        booking_col = "next_week_bookings" if next_week else "current_week_bookings"
        for row in rows:
            if (row["day"].strip().lower()  == day.strip().lower()
                    and row["start_time"].strip() == start_time.strip()):
                booked = int(row[booking_col])
                if booked <= 0:
                    logger.warning(f"[ScheduleManager] Cannot cancel {day} {start_time} — already at 0.")
                    return False
                row[booking_col] = str(booked - 1)
                self._save(rows)
                logger.info(f"[ScheduleManager] Cancelled: {day} {start_time} "
                            f"({'next' if next_week else 'current'} week)")
                return True
        logger.warning(f"[ScheduleManager] Cancel: slot not found: {day} {start_time}")
        return False

    # ── Internal ───────────────────────────────────────────────────

    def _load(self) -> list[dict]:
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