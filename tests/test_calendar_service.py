import tempfile
import threading
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.models import CallPayload
from services.calendar_service import CalendarSlot, GoogleCalendarService, LocalCalendarService
from services.twilio_groq_voice import (
    ConversationState,
    STAGE_FAREWELL,
    STAGE_MEETING,
    TwoWayAIVoiceService,
    _parse_meeting_request,
    _wants_to_cancel,
)


class CalendarServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.calendar = LocalCalendarService(
            path=Path(self.temp_dir.name) / "calendar.json",
        )
        self.zone = self.calendar.timezone
        self.now = datetime(2026, 6, 22, 8, 0, tzinfo=self.zone)
        self.calendar._now_fn = lambda: self.now

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_booking_removes_slot_and_persists(self):
        slots = self.calendar.find_available_slots("teacher@example.com")
        self.assertEqual([slot.start.hour for slot in slots], [9, 9, 10])

        booking = self.calendar.book_meeting(
            "teacher@example.com", slots[0], student_name="Aarav",
            parent_name="Mrs Sharma", reason="attendance",
        )

        self.assertTrue(booking.event_id)
        refreshed = self.calendar.find_available_slots("teacher@example.com")
        self.assertNotIn(slots[0], refreshed)
        self.assertTrue(self.calendar.path.exists())

    def test_double_booking_is_rejected(self):
        slot = self.calendar.find_available_slots("teacher@example.com")[0]
        self.calendar.book_meeting(
            "teacher@example.com", slot, student_name="Aarav",
            parent_name="Mrs Sharma", reason="attendance",
        )
        with self.assertRaises(ValueError):
            self.calendar.book_meeting(
                "teacher@example.com", slot, student_name="Neel",
                parent_name="Mr Banerjee", reason="performance",
            )

    def test_weekends_are_skipped(self):
        self.calendar._now_fn = lambda: datetime(2026, 6, 20, 8, 0, tzinfo=self.zone)
        slot = self.calendar.find_available_slots("teacher@example.com", limit=1)[0]
        self.assertEqual(slot.start.weekday(), 0)

    def test_natural_date_corrections_and_ranges_are_parsed(self):
        corrected = _parse_meeting_request(
            "Not Tuesday, Wednesday instead", self.zone, self.now
        )
        next_week = _parse_meeting_request("Sometime next week", self.zone, self.now)
        later_week = _parse_meeting_request("The week after next", self.zone, self.now)
        next_month = _parse_meeting_request("Try next month", self.zone, self.now)

        self.assertEqual(corrected.date, date(2026, 6, 24))
        self.assertEqual((next_week.range_start, next_week.range_end), (
            date(2026, 6, 29), date(2026, 7, 5)
        ))
        self.assertEqual((later_week.range_start, later_week.range_end), (
            date(2026, 7, 6), date(2026, 7, 12)
        ))
        self.assertEqual((next_month.range_start, next_month.range_end), (
            date(2026, 7, 1), date(2026, 7, 31)
        ))
        self.assertTrue(_wants_to_cancel("Please cancel it"))

    def test_make_call_isolates_conversations_by_call_sid(self):
        service = object.__new__(TwoWayAIVoiceService)
        service._twilio_ready = True
        service._school = "Test School"
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._from_number = "+910000000099"
        service._conversations = {}
        service.calls_made = []
        service._client = MagicMock()
        service._client.calls.create.side_effect = [
            SimpleNamespace(sid="CA-DUMMY-1"),
            SimpleNamespace(sid="CA-DUMMY-2"),
        ]
        payload = CallPayload(
            to_number="+910000000000", registration="REG-SAME",
            student_name="Aarav", parent_name="Mrs Sharma",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
        )

        service.make_call(payload)
        service.make_call(payload)

        self.assertEqual(set(service._conversations), {"CA-DUMMY-1", "CA-DUMMY-2"})
        self.assertIsNot(
            service._conversations["CA-DUMMY-1"],
            service._conversations["CA-DUMMY-2"],
        )

    def test_time_outside_meeting_hours_is_rejected(self):
        start = datetime(2026, 6, 22, 14, 0, tzinfo=self.zone)
        slot = CalendarSlot(start, start + self.calendar.duration)
        self.assertFalse(self.calendar.is_available("teacher@example.com", slot))

    def test_voice_flow_books_only_an_offered_choice(self):
        payload = CallPayload(
            to_number="+910000000000", registration="REG-1",
            student_name="Aarav", parent_name="Mrs Sharma",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.offered_slots = self.calendar.find_available_slots(payload.teacher_id)
        state.stage = STAGE_MEETING

        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        reply = service._handle_meeting_choice(state, "The second option works for me")

        self.assertIn("booked for", reply)
        self.assertEqual(state.booking.start.hour, 9)
        self.assertEqual(state.booking.start.minute, 30)
        self.assertEqual(state.stage, STAGE_FAREWELL)

    def test_reschedule_updates_existing_event(self):
        original = self.calendar.find_available_slots("teacher@example.com")[0]
        booking = self.calendar.book_meeting(
            "teacher@example.com", original, student_name="Neel",
            parent_name="Mr Banerjee", reason="attendance",
        )
        new_start = datetime(2026, 6, 22, 11, 0, tzinfo=self.zone)
        replacement = CalendarSlot(new_start, new_start + self.calendar.duration)

        updated = self.calendar.reschedule_meeting(booking, replacement)

        self.assertEqual(updated.event_id, booking.event_id)
        self.assertTrue(self.calendar.is_available(booking.teacher_id, original))
        self.assertFalse(self.calendar.is_available(booking.teacher_id, replacement))

    def test_cancellation_removes_event(self):
        slot = self.calendar.find_available_slots("teacher@example.com")[0]
        booking = self.calendar.book_meeting(
            "teacher@example.com", slot, student_name="Neel",
            parent_name="Mr Banerjee", reason="attendance",
        )
        self.calendar.cancel_meeting(booking)
        self.assertTrue(self.calendar.is_available(booking.teacher_id, slot))

    def test_voice_reschedule_changes_booking_not_just_reply(self):
        payload = CallPayload(
            to_number="+910000000000", registration="REG-2",
            student_name="Neel", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"

        reply = service._handle_reschedule(
            state, "Move the meeting to June 22, 2026 at 11:00 AM"
        )

        self.assertIn("moved to", reply)
        self.assertEqual(state.booking.start.hour, 11)
        self.assertTrue(self.calendar.is_available(payload.teacher_id, original))

    def test_sunday_and_ambiguous_requests_get_specific_clarification(self):
        payload = CallPayload(
            to_number="+910000000000", registration="REG-3",
            student_name="Neel", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.offered_slots = self.calendar.find_available_slots(payload.teacher_id)
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"

        sunday = service._clarify_meeting_request(state, "Sunday")
        ambiguous = service._clarify_meeting_request(state, "What is the date and time?")

        self.assertIn("outside the teacher's working days", sunday)
        self.assertIn("Please choose one of these times", ambiguous)

        state.stage = STAGE_MEETING
        sunday_with_time = service._handle_meeting_choice(state, "Sunday at 10 AM")
        self.assertIn("Monday through Friday", sunday_with_time)

    def test_partial_reschedule_remembers_date_for_next_turn(self):
        payload = CallPayload(
            to_number="+910000000000", registration="REG-4",
            student_name="Neel", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"

        clarification = service._handle_reschedule(state, "Move it to Monday")
        confirmation = service._handle_reschedule(state, "At 11 AM")

        self.assertIn("What time on Monday", clarification)
        self.assertIn("moved to", confirmation)
        self.assertEqual(state.booking.start.hour, 11)

    def test_transcribed_shift_request_updates_calendar_before_reply(self):
        payload = CallPayload(
            to_number="+910000000000", registration="REG-TRANSCRIPT",
            student_name="Saanvi Joshi", parent_name="Mrs Joshi",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[1]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        state.stage = STAGE_FAREWELL

        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration,
            "Also, sorry, can you shifted to Monday from 11:00 a.m. as I may be called in for work.",
        )

        self.assertIn("has been moved to Monday, June 22 at 11:00 AM", twiml)
        self.assertEqual((state.booking.start.hour, state.booking.start.minute), (11, 0))
        self.assertTrue(self.calendar.is_available(payload.teacher_id, original))
        events = self.calendar._read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(datetime.fromisoformat(events[0]["start"]).hour, 11)

    def test_fallback_reschedule_uses_destination_time_and_existing_date(self):
        payload = CallPayload(
            to_number="+910000000000", registration="REG-FALLBACK",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[1]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"

        reply = service._handle_reschedule(
            state, "I wanted to schedule my 9:30 AM meeting to 11:30 AM."
        )

        self.assertIn("moved to", reply)
        self.assertEqual(state.booking.start.date(), original.start.date())
        self.assertEqual((state.booking.start.hour, state.booking.start.minute), (11, 30))

    def test_google_reschedule_patches_original_event_without_inserting(self):
        google = object.__new__(GoogleCalendarService)
        google.timezone = self.zone
        google.duration = self.calendar.duration
        google.working_days = self.calendar.working_days
        google.meeting_start = self.calendar.meeting_start
        google.meeting_end = self.calendar.meeting_end
        google._now_fn = self.calendar._now_fn
        google._lock = threading.Lock()
        google._service = MagicMock()
        google._service.freebusy.return_value.query.return_value.execute.return_value = {
            "calendars": {"teacher@example.com": {"busy": []}}
        }
        google._service.events.return_value.patch.return_value.execute.return_value = {}

        original = CalendarSlot(
            datetime(2026, 6, 22, 9, 30, tzinfo=self.zone),
            datetime(2026, 6, 22, 10, 0, tzinfo=self.zone),
        )
        booking = self.calendar.book_meeting(
            "teacher@example.com", original, student_name="Saanvi Joshi",
            parent_name="Mrs Joshi", reason="attendance",
        )
        replacement = CalendarSlot(
            datetime(2026, 6, 22, 11, 0, tzinfo=self.zone),
            datetime(2026, 6, 22, 11, 30, tzinfo=self.zone),
        )

        updated = google.reschedule_meeting(booking, replacement)

        google._service.events.return_value.patch.assert_called_once()
        patch_kwargs = google._service.events.return_value.patch.call_args.kwargs
        self.assertEqual(patch_kwargs["eventId"], booking.event_id)
        self.assertEqual(patch_kwargs["body"]["start"]["dateTime"], replacement.start.isoformat())
        google._service.events.return_value.insert.assert_not_called()
        self.assertEqual(updated.start, replacement.start)

    def test_parser_understands_relative_dates_formats_and_spoken_time(self):
        now = datetime(2026, 6, 20, 8, 0, tzinfo=self.zone)
        tomorrow = _parse_meeting_request("tomorrow at ten", self.zone, now)
        named = _parse_meeting_request("next Monday at 9:30 AM", self.zone, now)
        formatted = _parse_meeting_request("22 June 2026 from 11.00 AM", self.zone, now)

        self.assertEqual(tomorrow.date.isoformat(), "2026-06-21")
        self.assertEqual((tomorrow.time.hour, tomorrow.time.minute), (10, 0))
        self.assertEqual(named.date.isoformat(), "2026-06-22")
        self.assertEqual((named.time.hour, named.time.minute), (9, 30))
        self.assertEqual(formatted.date.isoformat(), "2026-06-22")
        self.assertEqual((formatted.time.hour, formatted.time.minute), (11, 0))


if __name__ == "__main__":
    unittest.main()
