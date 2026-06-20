"""Teacher availability and meeting booking services."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone as fixed_timezone, tzinfo
from pathlib import Path
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CalendarSlot:
    start: datetime
    end: datetime

    def spoken(self) -> str:
        day = self.start.strftime("%A, %B %d")
        clock = self.start.strftime("%I:%M %p").lstrip("0")
        return f"{day} at {clock}"


@dataclass(frozen=True)
class MeetingBooking:
    event_id: str
    teacher_id: str
    start: datetime
    end: datetime
    student_name: str
    parent_name: str
    reason: str


class CalendarService(Protocol):
    timezone: tzinfo
    duration: timedelta
    working_days: frozenset[int]
    meeting_start: time
    meeting_end: time

    def find_available_slots(
        self, teacher_id: str, *, start: datetime | None = None,
        days: int = 7, limit: int = 3
    ) -> list[CalendarSlot]: ...

    def is_available(self, teacher_id: str, slot: CalendarSlot) -> bool: ...

    def book_meeting(
        self, teacher_id: str, slot: CalendarSlot, *, student_name: str,
        parent_name: str, reason: str
    ) -> MeetingBooking: ...

    def reschedule_meeting(
        self, booking: MeetingBooking, new_slot: CalendarSlot
    ) -> MeetingBooking: ...

    def cancel_meeting(self, booking: MeetingBooking) -> None: ...


class LocalCalendarService:
    """A small persistent calendar suitable for development and single-process use."""

    def __init__(
        self,
        path: str | Path = "data/teacher_calendar.json",
        timezone: str = "Asia/Kolkata",
        meeting_start: time = time(9, 0),
        meeting_end: time = time(12, 0),
        duration_minutes: int = 30,
        working_days: frozenset[int] | None = None,
        now_fn=None,
    ):
        self.path = Path(path)
        self.timezone = _load_timezone(timezone)
        self.meeting_start = meeting_start
        self.meeting_end = meeting_end
        self.duration = timedelta(minutes=duration_minutes)
        self.working_days = working_days or frozenset(range(5))
        self._now_fn = now_fn or (lambda: datetime.now(self.timezone))
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "LocalCalendarService":
        return cls(
            path=os.getenv("TEACHER_CALENDAR_PATH", "data/teacher_calendar.json"),
            timezone=os.getenv("SCHOOL_TIMEZONE", "Asia/Kolkata"),
            meeting_start=_parse_clock(os.getenv("MEETING_HOURS_START", "09:00")),
            meeting_end=_parse_clock(os.getenv("MEETING_HOURS_END", "12:00")),
            duration_minutes=int(os.getenv("MEETING_DURATION_MINUTES", "30")),
            working_days=_parse_working_days(os.getenv("MEETING_WORKING_DAYS", "mon,tue,wed,thu,fri")),
        )

    def find_available_slots(
        self, teacher_id: str, *, start: datetime | None = None,
        days: int = 7, limit: int = 3
    ) -> list[CalendarSlot]:
        cursor = (start or self._now_fn()).astimezone(self.timezone)
        results: list[CalendarSlot] = []
        for offset in range(days + 1):
            date = cursor.date() + timedelta(days=offset)
            if date.weekday() not in self.working_days:
                continue
            slot_start = datetime.combine(date, self.meeting_start, self.timezone)
            day_end = datetime.combine(date, self.meeting_end, self.timezone)
            while slot_start + self.duration <= day_end:
                slot = CalendarSlot(slot_start, slot_start + self.duration)
                if slot.start > cursor and self.is_available(teacher_id, slot):
                    results.append(slot)
                    if len(results) == limit:
                        return results
                slot_start += self.duration
        return results

    def is_available(self, teacher_id: str, slot: CalendarSlot) -> bool:
        if slot.start <= self._now_fn().astimezone(self.timezone):
            return False
        if slot.start.weekday() not in self.working_days or slot.end - slot.start != self.duration:
            return False
        if slot.start.timetz().replace(tzinfo=None) < self.meeting_start:
            return False
        if slot.end.timetz().replace(tzinfo=None) > self.meeting_end:
            return False
        return not any(
            event["teacher_id"] == teacher_id
            and slot.start < datetime.fromisoformat(event["end"])
            and slot.end > datetime.fromisoformat(event["start"])
            for event in self._read_events()
        )

    def book_meeting(
        self, teacher_id: str, slot: CalendarSlot, *, student_name: str,
        parent_name: str, reason: str
    ) -> MeetingBooking:
        with self._lock:
            if not self.is_available(teacher_id, slot):
                raise ValueError("The selected time is no longer available.")
            booking = MeetingBooking(
                event_id=str(uuid4()), teacher_id=teacher_id,
                start=slot.start, end=slot.end, student_name=student_name,
                parent_name=parent_name, reason=reason,
            )
            events = self._read_events()
            event = asdict(booking)
            event["start"] = booking.start.isoformat()
            event["end"] = booking.end.isoformat()
            events.append(event)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(events, indent=2), encoding="utf-8")
            return booking

    def reschedule_meeting(
        self, booking: MeetingBooking, new_slot: CalendarSlot
    ) -> MeetingBooking:
        with self._lock:
            events = self._read_events()
            index = next((i for i, event in enumerate(events) if event.get("event_id") == booking.event_id), None)
            if index is None:
                raise ValueError("The original meeting could not be found.")
            if not self._slot_is_valid(new_slot) or self._has_conflict(
                booking.teacher_id, new_slot, events, ignore_event_id=booking.event_id
            ):
                raise ValueError("The requested time is not available.")
            updated = MeetingBooking(
                event_id=booking.event_id, teacher_id=booking.teacher_id,
                start=new_slot.start, end=new_slot.end,
                student_name=booking.student_name, parent_name=booking.parent_name,
                reason=booking.reason,
            )
            event = asdict(updated)
            event["start"] = updated.start.isoformat()
            event["end"] = updated.end.isoformat()
            events[index] = event
            self.path.write_text(json.dumps(events, indent=2), encoding="utf-8")
            return updated

    def cancel_meeting(self, booking: MeetingBooking) -> None:
        with self._lock:
            events = self._read_events()
            remaining = [event for event in events if event.get("event_id") != booking.event_id]
            if len(remaining) == len(events):
                raise ValueError("The meeting could not be found.")
            self.path.write_text(json.dumps(remaining, indent=2), encoding="utf-8")

    def _slot_is_valid(self, slot: CalendarSlot) -> bool:
        return (
            slot.start > self._now_fn().astimezone(self.timezone)
            and slot.start.weekday() in self.working_days
            and slot.end - slot.start == self.duration
            and slot.start.timetz().replace(tzinfo=None) >= self.meeting_start
            and slot.end.timetz().replace(tzinfo=None) <= self.meeting_end
        )

    def _has_conflict(
        self, teacher_id: str, slot: CalendarSlot, events: list[dict],
        ignore_event_id: str | None = None,
    ) -> bool:
        return any(
            event.get("teacher_id") == teacher_id
            and event.get("event_id") != ignore_event_id
            and slot.start < datetime.fromisoformat(event["end"])
            and slot.end > datetime.fromisoformat(event["start"])
            for event in events
        )

    def _read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []


class GoogleCalendarService(LocalCalendarService):
    """Google Calendar implementation using teacher email addresses as calendar IDs."""

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self, credentials_file: str, **kwargs):
        super().__init__(path=Path(os.devnull), **kwargs)
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "Google Calendar dependencies are not installed. Run pip install -r requirements.txt."
            ) from exc
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=self.SCOPES
        )
        self._service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

    @classmethod
    def from_environment(cls) -> "GoogleCalendarService":
        credentials_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        if not credentials_file:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_FILE is required for Google Calendar.")
        return cls(
            credentials_file=credentials_file,
            timezone=os.getenv("SCHOOL_TIMEZONE", "Asia/Kolkata"),
            meeting_start=_parse_clock(os.getenv("MEETING_HOURS_START", "09:00")),
            meeting_end=_parse_clock(os.getenv("MEETING_HOURS_END", "12:00")),
            duration_minutes=int(os.getenv("MEETING_DURATION_MINUTES", "30")),
            working_days=_parse_working_days(os.getenv("MEETING_WORKING_DAYS", "mon,tue,wed,thu,fri")),
        )

    def find_available_slots(
        self, teacher_id: str, *, start: datetime | None = None,
        days: int = 7, limit: int = 3
    ) -> list[CalendarSlot]:
        cursor = (start or self._now_fn()).astimezone(self.timezone)
        range_end = cursor + timedelta(days=days + 1)
        busy = self._busy_periods(teacher_id, cursor, range_end)
        results: list[CalendarSlot] = []
        for offset in range(days + 1):
            date = cursor.date() + timedelta(days=offset)
            if date.weekday() not in self.working_days:
                continue
            slot_start = datetime.combine(date, self.meeting_start, self.timezone)
            day_end = datetime.combine(date, self.meeting_end, self.timezone)
            while slot_start + self.duration <= day_end:
                slot = CalendarSlot(slot_start, slot_start + self.duration)
                if slot.start > cursor and not _overlaps_any(slot, busy):
                    results.append(slot)
                    if len(results) == limit:
                        return results
                slot_start += self.duration
        return results

    def is_available(self, teacher_id: str, slot: CalendarSlot) -> bool:
        if not self._within_meeting_hours(slot):
            return False
        return not self._busy_periods(teacher_id, slot.start, slot.end)

    def book_meeting(
        self, teacher_id: str, slot: CalendarSlot, *, student_name: str,
        parent_name: str, reason: str
    ) -> MeetingBooking:
        with self._lock:
            if not self.is_available(teacher_id, slot):
                raise ValueError("The selected time is no longer available.")
            body = {
                "summary": f"Parent meeting: {student_name}",
                "description": f"Parent: {parent_name}\nReason: {reason}",
                "start": {"dateTime": slot.start.isoformat(), "timeZone": str(self.timezone)},
                "end": {"dateTime": slot.end.isoformat(), "timeZone": str(self.timezone)},
            }
            event = self._service.events().insert(calendarId=teacher_id, body=body).execute()
            return MeetingBooking(
                event_id=event["id"], teacher_id=teacher_id, start=slot.start,
                end=slot.end, student_name=student_name, parent_name=parent_name,
                reason=reason,
            )

    def reschedule_meeting(
        self, booking: MeetingBooking, new_slot: CalendarSlot
    ) -> MeetingBooking:
        with self._lock:
            if new_slot != CalendarSlot(booking.start, booking.end) and not self.is_available(
                booking.teacher_id, new_slot
            ):
                raise ValueError("The requested time is not available.")
            body = {
                "start": {"dateTime": new_slot.start.isoformat(), "timeZone": str(self.timezone)},
                "end": {"dateTime": new_slot.end.isoformat(), "timeZone": str(self.timezone)},
            }
            self._service.events().patch(
                calendarId=booking.teacher_id, eventId=booking.event_id, body=body
            ).execute()
            return MeetingBooking(
                event_id=booking.event_id, teacher_id=booking.teacher_id,
                start=new_slot.start, end=new_slot.end,
                student_name=booking.student_name, parent_name=booking.parent_name,
                reason=booking.reason,
            )

    def cancel_meeting(self, booking: MeetingBooking) -> None:
        with self._lock:
            self._service.events().delete(
                calendarId=booking.teacher_id, eventId=booking.event_id
            ).execute()

    def _busy_periods(self, teacher_id: str, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        response = self._service.freebusy().query(body={
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": str(self.timezone),
            "items": [{"id": teacher_id}],
        }).execute()
        periods = response.get("calendars", {}).get(teacher_id, {}).get("busy", [])
        return [(datetime.fromisoformat(item["start"]), datetime.fromisoformat(item["end"])) for item in periods]

    def _within_meeting_hours(self, slot: CalendarSlot) -> bool:
        return (
            slot.start > self._now_fn().astimezone(self.timezone)
            and slot.start.weekday() in self.working_days
            and slot.end - slot.start == self.duration
            and slot.start.timetz().replace(tzinfo=None) >= self.meeting_start
            and slot.end.timetz().replace(tzinfo=None) <= self.meeting_end
        )


def create_calendar_service() -> CalendarService:
    provider = os.getenv("CALENDAR_PROVIDER", "local").lower()
    if provider == "google":
        return GoogleCalendarService.from_environment()
    if provider == "local":
        return LocalCalendarService.from_environment()
    raise ValueError(f"Unsupported CALENDAR_PROVIDER: {provider}")


def _overlaps_any(slot: CalendarSlot, periods: list[tuple[datetime, datetime]]) -> bool:
    return any(slot.start < end and slot.end > start for start, end in periods)


def _parse_clock(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _parse_working_days(value: str) -> frozenset[int]:
    names = {
        "mon": 0, "monday": 0, "tue": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2, "thu": 3, "thursday": 3,
        "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    days = frozenset(names[item.strip().lower()] for item in value.split(",") if item.strip().lower() in names)
    if not days:
        raise ValueError("MEETING_WORKING_DAYS must contain at least one valid weekday.")
    return days


def _load_timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except Exception:
        if name == "Asia/Kolkata":
            return fixed_timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
        raise
