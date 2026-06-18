"""
services/schedule_manager.py
─────────────────────────────
Production PostgreSQL Database Scheduling Engine.
Handles atomic locks, real dates, and risk-based load balancing.
(Updated with explicit Postgres Type Casting)
"""

import csv
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional
import psycopg2.extras
from services.database import _get_conn

logger = logging.getLogger(__name__)

WEEKLY_CSV_PATH = os.path.join("data", "Weekly_Schedule.csv")

# Minimum lead time: a slot must be at least this many minutes in the future
# to be offered to a parent. 90 minutes gives time to travel.
MINIMUM_LEAD_MINUTES = 90
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
DAY_INDEX = {d.lower(): i for i, d in enumerate(DAYS)}
DAY_INDEX["sunday"] = 6

def _this_weeks_monday(ref: date = None) -> date:
    ref = ref or date.today()
    return ref - timedelta(days=ref.weekday())

def _next_weeks_monday(ref: date = None) -> date:
    return _this_weeks_monday(ref) + timedelta(weeks=1)

def _normalise_time(t: str) -> str:
    parts = t.strip().split(":")
    if len(parts) != 2: return t.strip()
    hour = int(parts[0])
    minute = parts[1].strip().zfill(2)
    if 1 <= hour <= 7: hour += 12
    return f"{hour:02d}:{minute}"

def sync_from_weekly_csv(weekly_path: str = WEEKLY_CSV_PATH) -> None:
    """Reads the CSV template and projects it 30 days into the future in the DB."""
    if not os.path.exists(weekly_path):
        logger.warning(f"[ScheduleManager] {weekly_path} not found.")
        return

    template = {d: [] for d in DAYS}
    with open(weekly_path, "r", encoding="utf-8-sig", newline="") as f:
        for raw_row in csv.DictReader(f):
            time_range = raw_row.get("Time", "").strip()
            if not time_range or " - " not in time_range: continue
            start, end = [p.strip() for p in time_range.split(" - ")]
            start_t, end_t = _normalise_time(start), _normalise_time(end)

            for day in DAYS:
                if raw_row.get(day, "").strip().upper() == "FREE":
                    template[day].append({"start": start_t, "end": end_t, "cap": 1})

    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                for offset in range(30):
                    target = date.today() + timedelta(days=offset)
                    day_name = target.strftime("%A")
                    for slot in template.get(day_name, []):
                        # Explicitly cast %s::date and %s::time to prevent Postgres Operator errors
                        cur.execute("""
                            INSERT INTO schedule_availability (slot_date, day_of_week, start_time, end_time, capacity)
                            VALUES (%s::date, %s, %s::time, %s::time, %s)
                            ON CONFLICT (slot_date, start_time) DO NOTHING
                        """, (target, day_name, slot["start"], slot["end"], slot["cap"]))
        logger.info("[ScheduleManager] Synced 30-day rolling schedule to DB.")
    except Exception as e:
        logger.error(f"[ScheduleManager] DB sync failed: {e}")

class ScheduleManager:
    def reset_week_if_needed(self) -> None:
        pass 

    def _resolve_date(self, day_name: str, date_iso: str = None, next_week: bool = False) -> date:
        if date_iso: return date.fromisoformat(date_iso)
        today = date.today()
        days_ahead = (DAY_INDEX.get(day_name.lower(), 0) - today.weekday()) % 7
        if days_ahead == 0 and next_week: days_ahead = 7
        elif days_ahead == 0 and not next_week: days_ahead = 0
        target = today + timedelta(days=days_ahead)
        if next_week and target < _next_weeks_monday():
            target += timedelta(weeks=1)
        return target

    def get_available_slots_for_day(self, day_name: str, next_week: bool = False, risk_level: str = "MEDIUM", date_iso: str = None) -> list[dict]:
        target_date = self._resolve_date(day_name, date_iso, next_week)
        
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, day_of_week as day, 
                           TO_CHAR(start_time, 'HH24:MI') as start_time,
                           TO_CHAR(end_time, 'HH24:MI') as end_time,
                           capacity,
                           (SELECT count(*) FROM schedule_bookings b 
                            WHERE b.availability_id = schedule_availability.id 
                            AND b.status = 'CONFIRMED') as booked
                    FROM schedule_availability
                    WHERE slot_date = %s::date
                """, (target_date,))
                rows = cur.fetchall()

        slots = []
        for r in rows:
            if r['booked'] < r['capacity']:
                slots.append({
                    "day": r['day'], "start_time": r['start_time'], "end_time": r['end_time'],
                    "use_next_week": next_week, "date": target_date.isoformat(),
                    "available_spots": r['capacity'] - r['booked']
                })

        if risk_level.upper() == "HIGH":
            slots.sort(key=lambda x: x["start_time"])
        else:
            slots.sort(key=lambda x: (-x["available_spots"], x["start_time"]))

        return slots

    def get_today_available_slot(self, risk_level: str = "MEDIUM") -> Optional[dict]:
        today_str = date.today().isoformat()
        # Only offer slots at least MINIMUM_LEAD_MINUTES away — parents need travel time
        cutoff = (datetime.now() + timedelta(minutes=MINIMUM_LEAD_MINUTES)).strftime("%H:%M")
        slots = self.get_available_slots_for_day(date.today().strftime("%A"), date_iso=today_str, risk_level=risk_level)
        slots = [s for s in slots if s["start_time"] >= cutoff]
        return slots[0] if slots else None

    def get_next_available_slot(self, prefer_next_week: bool = False, risk_level: str = "MEDIUM") -> Optional[dict]:
        start = _next_weeks_monday() if prefer_next_week else date.today()
        if date.today().weekday() == 5 and not prefer_next_week: start += timedelta(days=2) 
        
        # Only offer slots at least MINIMUM_LEAD_MINUTES away when the slot is today
        cutoff = (datetime.now() + timedelta(minutes=MINIMUM_LEAD_MINUTES)).strftime("%H:%M")

        for offset in range(14):
            candidate = start + timedelta(days=offset)
            if candidate.weekday() == 6: continue 
            
            slots = self.get_available_slots_for_day(candidate.strftime("%A"), date_iso=candidate.isoformat(), risk_level=risk_level)
            if candidate == date.today():
                slots = [s for s in slots if s["start_time"] >= cutoff]
            if slots:
                return slots[0] 
        return None

    def book_slot(self, day: str, start_time: str, registration: str, date_iso: str = None, next_week: bool = False) -> bool:
        target_date = self._resolve_date(day, date_iso, next_week)

        with _get_conn() as conn:
            with conn.cursor() as cur:
                # Step 1: Lock the availability row so no other request can
                # read or write it until this transaction commits or rolls back.
                # This is the key fix — without FOR UPDATE, two simultaneous
                # webhooks can both pass the capacity check and both insert.
                cur.execute("""
                    SELECT id, capacity
                    FROM schedule_availability
                    WHERE slot_date = %s::date AND start_time = %s::time
                    FOR UPDATE
                """, (target_date, start_time))
                row = cur.fetchone()
                if not row:
                    logger.warning(f"[ScheduleManager] Slot not found: {day} {start_time} on {target_date}")
                    return False

                availability_id, capacity = row

                # Step 2: Count confirmed bookings while we hold the lock
                cur.execute("""
                    SELECT count(*) FROM schedule_bookings
                    WHERE availability_id = %s AND status = 'CONFIRMED'
                """, (availability_id,))
                booked = cur.fetchone()[0]

                if booked >= capacity:
                    logger.warning(f"[ScheduleManager] Slot full: {day} {start_time} on {target_date} ({booked}/{capacity})")
                    return False

                # Step 3: Safe to insert — we hold the lock
                cur.execute("""
                    INSERT INTO schedule_bookings (availability_id, registration, status)
                    VALUES (%s, %s, 'CONFIRMED')
                    RETURNING id
                """, (availability_id, registration))
                if cur.fetchone():
                    logger.info(f"[ScheduleManager] Confirmed DB booking for {registration} on {target_date} at {start_time}")
                    return True
                return False

    def cancel_slot(self, day: str, start_time: str, registration: str, date_iso: str = None, next_week: bool = False) -> bool:
        target_date = self._resolve_date(day, date_iso, next_week)
        with _get_conn() as conn:
            with conn.cursor() as cur:
                # Explicitly cast %s::date and %s::time
                cur.execute("""
                    UPDATE schedule_bookings SET status = 'CANCELLED'
                    WHERE registration = %s AND status = 'CONFIRMED'
                    AND availability_id = (
                        SELECT id FROM schedule_availability 
                        WHERE slot_date = %s::date AND start_time = %s::time 
                        LIMIT 1
                    )
                """, (registration, target_date, start_time))
                return cur.rowcount > 0