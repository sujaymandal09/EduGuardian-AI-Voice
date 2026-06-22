import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.models import CallPayload
from services.calendar_service import LocalCalendarService
from services.intent_interpreter import (
    ContextualIntentInterpreter,
    ParentIntent,
    TurnUnderstanding,
)
from services.twilio_groq_voice import (
    ConversationState,
    STAGE_AVAILABILITY,
    STAGE_CONVERSATION,
    STAGE_FAREWELL,
    STAGE_INTRO,
    STAGE_MEETING,
    STAGE_RESCHEDULE,
    STAGE_SOLUTION,
    TwoWayAIVoiceService,
)


class FakeInterpreter:
    def __init__(self, understanding):
        self.understanding = understanding
        self.context = None

    def interpret(self, parent_speech, context):
        self.context = context
        return self.understanding


class SequenceInterpreter:
    def __init__(self, understandings):
        self.understandings = iter(understandings)

    def interpret(self, parent_speech, context):
        return next(self.understandings)


class ContextualIntentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.calendar = LocalCalendarService(
            path=Path(self.temp_dir.name) / "calendar.json"
        )
        self.zone = self.calendar.timezone
        self.now = datetime(2026, 6, 22, 8, 0, tzinfo=self.zone)
        self.calendar._now_fn = lambda: self.now

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_identity_answer_advances_to_availability_then_conversation(self):
        payload = CallPayload(
            to_number="+910000000000", registration="IDENTITY-1",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM",
            details="Saanvi's Science performance has declined.",
        )
        state = ConversationState(payload, "Test School", "12345")
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}
        service._intent_interpreter = SequenceInterpreter([
            TurnUnderstanding(
                intent=ParentIntent.CONFIRM_IDENTITY, confidence=1.0
            ),
            TurnUnderstanding(
                intent=ParentIntent.CONFIRM_IDENTITY, confidence=1.0
            ),
        ])

        identity_reply = service.generate_followup_twiml(payload.registration, "Yes.")
        self.assertIn("Do you have a few minutes", identity_reply)
        self.assertEqual(state.stage, STAGE_AVAILABILITY)

        availability_reply = service.generate_followup_twiml(payload.registration, "Oh, yes.")
        self.assertIn("Science performance has declined", availability_reply)
        self.assertIn("affecting their studies", availability_reply)
        self.assertEqual(state.stage, STAGE_CONVERSATION)
        self.assertEqual(
            [message["role"] for message in state.messages],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertEqual(state.turn_count, 2)

    def test_please_tell_me_does_not_enable_brief_mode(self):
        payload = CallPayload(
            to_number="+910000000000", registration="IDENTITY-3",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_AVAILABILITY
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.AVAILABLE_BRIEFLY, confidence=1.0
        ))

        reply = service.generate_followup_twiml(payload.registration, "Please tell me.")

        self.assertIn("Science grade is C", reply)
        self.assertFalse(state.brief_mode)
        self.assertEqual(state.stage, STAGE_CONVERSATION)

    def test_full_identity_sentence_is_accepted(self):
        payload = CallPayload(
            to_number="+910000000000", registration="IDENTITY-2",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
        )
        state = ConversationState(payload, "Test School", "12345")
        self.assertEqual(state.stage, STAGE_INTRO)
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.CONFIRM_IDENTITY, confidence=1.0
        ))

        reply = service.generate_followup_twiml(
            payload.registration, "Yes, I am the parent."
        )

        self.assertIn("Do you have a few minutes", reply)
        self.assertEqual(state.stage, STAGE_AVAILABILITY)

    def test_opening_is_split_and_ends_with_identity_question(self):
        payload = CallPayload(
            to_number="+910000000000", registration="OPENING-1",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
        )
        service = object.__new__(TwoWayAIVoiceService)
        service._ngrok_url = "https://example.test"
        service._phone = "12345"
        service._school = "Test School"

        opening = service._opening_twiml(payload)

        self.assertIn("This is Priya calling from Test School", opening)
        self.assertIn('<Pause length="1"/>', opening)
        self.assertIn("Am I speaking with Neha Joshi?", opening)
        self.assertNotIn("regarding your child", opening)

    def test_each_dimension_uses_a_specific_open_question(self):
        service = object.__new__(TwoWayAIVoiceService)
        expected = {
            "attendance": "affecting the attendance?",
            "performance": "affecting their studies?",
            "behavior": "things been at home recently?",
        }
        for dimension, ending in expected.items():
            with self.subTest(dimension=dimension):
                payload = CallPayload(
                    to_number="+910000000000", registration=f"DIM-{dimension}",
                    student_name="Saanvi", parent_name="Neha",
                    dimension=dimension, risk_level="MEDIUM", details="A concern was recorded.",
                )
                state = ConversationState(payload, "Test School", "12345")
                reply = service._begin_concern_discussion(state)
                self.assertTrue(reply.endswith(ending))

    def test_high_risk_concern_moves_to_verified_meeting_question(self):
        payload = CallPayload(
            to_number="+910000000000", registration="RISK-HIGH-CONVERSATION",
            student_name="Aarav", parent_name="Mrs Sharma",
            dimension="attendance", risk_level="HIGH", details="Attendance is critically low.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_CONVERSATION
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.DISCUSS_CONCERN,
            confidence=1.0,
            assistant_reply="Thank you for explaining the situation.",
        ))

        reply = service.generate_followup_twiml(
            payload.registration, "We have had a difficult week at home."
        )

        self.assertIn("earliest available meeting times?", reply)
        self.assertTrue(state.awaiting_meeting_consent)
        self.assertEqual(state.stage, STAGE_SOLUTION)

    def test_low_risk_conversation_never_suggests_meeting(self):
        payload = CallPayload(
            to_number="+910000000000", registration="RISK-LOW-CONVERSATION",
            student_name="Aarav", parent_name="Mrs Sharma",
            dimension="behavior", risk_level="LOW", details="A minor concern was recorded.",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_CONVERSATION
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.DISCUSS_CONCERN,
            confidence=1.0,
            assistant_reply="Would you like a meeting with the teacher?",
        ))

        reply = service.generate_followup_twiml(
            payload.registration, "We will speak with him at home."
        )

        self.assertNotIn("meeting", reply.lower())
        self.assertTrue(reply.split("</Say>", 1)[0].rstrip().endswith("?"))

    def test_dummy_call_moves_monday_booking_to_tuesday_without_duplicate(self):
        payload = CallPayload(
            to_number="+910000000000", registration="DUMMY-CALL-1",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        monday = self.calendar.find_available_slots(payload.teacher_id, days=0, limit=1)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, monday, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        original_event_id = state.booking.event_id
        state.stage = STAGE_FAREWELL
        state.parent_requested_meeting = True

        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._conversations = {payload.registration: state}
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.SELECT_SLOT,
            confidence=0.98,
            selected_option=1,
            sentiment="positive",
        ))

        offer = service.generate_followup_twiml(
            payload.registration, "Could we move it to Tuesday instead?"
        )
        self.assertIn("Tuesday", offer)
        self.assertEqual(state.stage, STAGE_RESCHEDULE)

        confirmation = service.generate_followup_twiml(
            payload.registration, "That earliest alternative would suit us nicely."
        )
        self.assertIn("moved to", confirmation)
        self.assertEqual(state.booking.start.weekday(), 1)
        self.assertEqual(state.booking.event_id, original_event_id)
        self.assertTrue(self.calendar.is_available(payload.teacher_id, monday))
        events = self.calendar._read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], original_event_id)

    def test_tool_call_is_validated_into_understanding(self):
        arguments = {
            "intent": "reschedule_meeting",
            "confidence": 0.97,
            "target_date": "2026-06-22",
            "target_time": "11:30",
            "source_time": "09:30",
            "requires_clarification": False,
            "sentiment": "hesitant",
            "reasoning": "The parent wants the existing meeting moved.",
        }
        tool_call = SimpleNamespace(
            function=SimpleNamespace(arguments=json.dumps(arguments))
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        result = ContextualIntentInterpreter(client, "llama-3.1-8b-instant", mode="tool").interpret(
            "Schedule my 9:30 meeting to 11:30",
            {"stage": "farewell", "active_booking": {"start_time": "09:30"}},
        )

        self.assertEqual(result.intent, ParentIntent.RESCHEDULE_MEETING)
        self.assertEqual(result.source_time, "09:30")
        self.assertEqual(result.target_time, "11:30")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["tool_choice"]["function"]["name"], "route_parent_turn")

    def test_invalid_tool_call_recovers_through_json_mode(self):
        fallback_data = {
            "intent": "ask_current_date",
            "confidence": 0.99,
            "target_date": None,
            "target_time": None,
            "source_time": None,
            "selected_option": None,
            "requires_clarification": False,
            "sentiment": "neutral",
            "reasoning": "The parent asks for today's date.",
        }
        fallback_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(fallback_data)
            ))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("tool_use_failed"), fallback_response
        ]

        result = ContextualIntentInterpreter(client, "llama-3.1-8b-instant", mode="tool").interpret(
            "What is the date today?", {"current_datetime": "2026-06-22T08:00:00+05:30"}
        )

        self.assertEqual(result.intent, ParentIntent.ASK_CURRENT_DATE)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        fallback_request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(fallback_request["response_format"], {"type": "json_object"})

    def test_contextual_reschedule_uses_target_and_updates_calendar(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-1",
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
        state.stage = STAGE_FAREWELL
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.RESCHEDULE_MEETING,
            confidence=0.98,
            target_time="11:30",
            source_time="09:30",
            sentiment="neutral",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "I wanted to schedule my 9:30 meeting to 11:30."
        )

        self.assertIn("has been moved to Monday, June 22 at 11:30 AM", twiml)
        self.assertEqual((state.booking.start.hour, state.booking.start.minute), (11, 30))
        self.assertTrue(self.calendar.is_available(payload.teacher_id, original))

    def test_contextual_yes_selects_option_from_prior_question(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-2",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.offered_slots = self.calendar.find_available_slots(payload.teacher_id)
        expected = state.offered_slots[1]
        state.stage = "meeting"
        state.last_agent_message = "Would you like the second option?"
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.SELECT_SLOT,
            confidence=0.95,
            selected_option=2,
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        service.generate_followup_twiml(payload.registration, "Yes, that works.")

        self.assertIsNotNone(state.booking)
        self.assertEqual(state.booking.start, expected.start)

    def test_low_confidence_action_does_not_modify_calendar(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-3",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.RESCHEDULE_MEETING,
            confidence=0.41,
            target_time="11:30",
            requires_clarification=True,
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(payload.registration, "Maybe later instead.")

        self.assertIn("make sure I understood correctly", twiml)
        self.assertEqual(state.booking.start, original.start)

    def test_busy_but_make_it_quick_gets_brief_summary_not_hangup(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-BRIEF",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH",
            details="Neel's attendance has fallen below the required level.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.AVAILABLE_BRIEFLY,
            confidence=0.96,
            sentiment="hesitant",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "Can you make it quickly? I don't have much time."
        )

        self.assertIn("I&#x27;ll keep this brief", twiml)
        self.assertIn("earliest meeting times", twiml)
        self.assertFalse(state.ended)

    def test_schedule_intent_in_slot_stage_books_instead_of_reoffering(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-SLOT-AS-SCHEDULE",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.parent_requested_meeting = True
        state.offered_slots = self.calendar.find_available_slots(payload.teacher_id, days=7)
        state.stage = STAGE_MEETING
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.SCHEDULE_MEETING,
            confidence=1.0,
            target_date="2026-06-22",
            target_time="11:00",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "Monday, 11:00 AM works for me."
        )

        self.assertIn("meeting is booked for Monday, June 22 at 11:00 AM", twiml)
        self.assertIsNotNone(state.booking)
        self.assertEqual((state.booking.start.hour, state.booking.start.minute), (11, 0))
        self.assertEqual(state.offered_slots, [])
        self.assertEqual(state.stage, STAGE_FAREWELL)

    def test_bare_tuesday_after_monday_booking_offers_tuesday_times(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-BARE-TUESDAY",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        monday = self.calendar.find_available_slots(payload.teacher_id)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, monday, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        state.stage = STAGE_FAREWELL
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.OTHER, confidence=0.9
        ))
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(payload.registration, "Tuesday.")

        self.assertIn("available times on Tuesday, June 23", twiml)
        self.assertEqual(state.stage, STAGE_RESCHEDULE)
        self.assertEqual(state.booking.start, monday.start)

    def test_sure_with_multiple_slots_asks_for_specific_choice(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-SURE-MULTIPLE",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.parent_requested_meeting = True
        state.offered_slots = self.calendar.find_available_slots(payload.teacher_id, days=7)
        state.stage = STAGE_MEETING
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.OTHER, confidence=0.9
        ))
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(payload.registration, "Sure.")

        self.assertIn("Please choose one of these times", twiml)
        self.assertIsNone(state.booking)

    def test_sure_books_when_only_one_slot_is_pending(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-SURE-SINGLE",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.parent_requested_meeting = True
        only_slot = self.calendar.find_available_slots(payload.teacher_id, days=7, limit=1)[0]
        state.offered_slots = [only_slot]
        state.stage = STAGE_MEETING
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.OTHER, confidence=0.9
        ))
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(payload.registration, "Sure.")

        self.assertIn("meeting is booked", twiml)
        self.assertEqual(state.booking.start, only_slot.start)

    def test_date_only_reschedule_offers_times_on_requested_day(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-DATE",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.RESCHEDULE_MEETING,
            confidence=0.97,
            target_date="2026-06-23",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "Please move the Monday meeting to Tuesday."
        )

        self.assertIn("available times on Tuesday, June 23", twiml)
        self.assertEqual(state.stage, STAGE_RESCHEDULE)
        self.assertTrue(all(slot.start.date().isoformat() == "2026-06-23" for slot in state.offered_slots))
        self.assertEqual(state.booking.start, original.start)

    def test_high_risk_date_beyond_two_days_directs_parent_to_school(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-HIGH-LATE",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        original = self.calendar.find_available_slots(payload.teacher_id)[0]
        state.booking = self.calendar.book_meeting(
            payload.teacher_id, original, student_name=payload.student_name,
            parent_name=payload.parent_name, reason=payload.dimension,
        )
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.RESCHEDULE_MEETING,
            confidence=0.98,
            target_date="2026-06-26",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "Can we postpone it until Friday?"
        )

        self.assertIn("limited to the next two days", twiml)
        self.assertIn("12345", twiml)
        self.assertEqual(state.booking.start, original.start)

    def test_medium_risk_parent_request_gets_one_week_window(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-MEDIUM",
            student_name="Neel Banerjee", parent_name="Mr Banerjee",
            dimension="performance", risk_level="MEDIUM", details="Grades are declining.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.SCHEDULE_MEETING,
            confidence=0.94,
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "I would like to meet the teacher."
        )

        self.assertIn("available times", twiml)
        self.assertTrue(state.parent_requested_meeting)
        self.assertEqual(state.stage, STAGE_MEETING)
        last_allowed = self.calendar.now().date().toordinal() + 7
        self.assertTrue(all(slot.start.date().toordinal() <= last_allowed for slot in state.offered_slots))

    def test_medium_risk_does_not_suggest_meeting_without_parent_request(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-MEDIUM-NO-PUSH",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_CONVERSATION
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.DISCUSS_CONCERN,
            confidence=1.0,
            assistant_reply="Thank you for explaining. Would you like a meeting with the teacher?",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        reply = service.generate_followup_twiml(
            payload.registration, "We have already arranged extra coaching."
        )

        self.assertNotIn("meeting", reply.lower())
        self.assertIn("monitor this closely", reply)
        self.assertFalse(state.parent_requested_meeting)
        self.assertEqual(state.offered_slots, [])

    def test_medium_risk_ignores_false_schedule_classification(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-MEDIUM-FALSE-SCHEDULE",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_CONVERSATION
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.SCHEDULE_MEETING, confidence=1.0
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        reply = service.generate_followup_twiml(
            payload.registration, "Honestly, we are already trying our best."
        )

        self.assertIn("monitor this closely", reply)
        self.assertFalse(state.parent_requested_meeting)
        self.assertEqual(state.offered_slots, [])

    def test_yes_accepts_pending_meeting_even_when_classifier_would_be_wrong(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-PENDING-MEETING-YES",
            student_name="Aarav", parent_name="Mrs Sharma",
            dimension="attendance", risk_level="HIGH", details="Low attendance",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_SOLUTION
        state.awaiting_meeting_consent = True
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.DISCUSS_CONCERN, confidence=1.0
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        reply = service.generate_followup_twiml(payload.registration, "Yes.")

        self.assertIn("available times", reply)
        self.assertTrue(state.parent_requested_meeting)
        self.assertEqual(state.stage, STAGE_MEETING)

    def test_no_need_anything_closes_instead_of_reclassifying_concern(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-NO-MORE",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.stage = STAGE_SOLUTION
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.DISCUSS_CONCERN, confidence=1.0
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        reply = service.generate_followup_twiml(
            payload.registration, "No, I don't need anything."
        )

        self.assertIn("goodbye", reply.lower())
        self.assertTrue(state.ended)

    def test_explicit_meeting_confirmation_opens_calendar(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-CONFIRM-MEETING",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        state.last_agent_message = "Do you think a meeting with us would be helpful?"
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.CONFIRM_ACTION,
            confidence=1.0,
            sentiment="positive",
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "Yes, yes, you can hold a meeting."
        )

        self.assertIn("available times", twiml)
        self.assertEqual(state.stage, STAGE_MEETING)
        self.assertTrue(state.parent_requested_meeting)

    def test_current_date_is_answered_from_school_clock(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-DATE-NOW",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
        )
        state = ConversationState(payload, "Test School", "12345")
        fake = FakeInterpreter(TurnUnderstanding(
            intent=ParentIntent.ASK_CURRENT_DATE,
            confidence=0.99,
        ))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(payload.registration, "What is the date today?")

        self.assertIn("Monday, June 22, 2026", twiml)
        self.assertNotIn("check_date", twiml.lower())

    def test_meetings_at_23_is_a_date_and_returns_slots(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-DAY-23",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
            teacher_id="teacher@example.com",
        )
        state = ConversationState(payload, "Test School", "12345")
        fake = FakeInterpreter(TurnUnderstanding.unclear("simulated tool failure"))
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = fake
        service._conversations = {payload.registration: state}

        twiml = service.generate_followup_twiml(
            payload.registration, "So, is there any meetings loaded at 23?"
        )

        self.assertIn("available times on Tuesday, June 23", twiml)
        self.assertEqual(state.stage, STAGE_MEETING)
        self.assertTrue(all(slot.start.date().isoformat() == "2026-06-23" for slot in state.offered_slots))

    def test_brief_mode_persists_after_parent_explains_reason(self):
        payload = CallPayload(
            to_number="+910000000000", registration="CTX-BRIEF-PERSIST",
            student_name="Saanvi Joshi", parent_name="Neha Joshi",
            dimension="performance", risk_level="MEDIUM", details="Science grade is C.",
        )
        state = ConversationState(payload, "Test School", "12345")
        interpreter = SequenceInterpreter([
            TurnUnderstanding(
                intent=ParentIntent.AVAILABLE_BRIEFLY, confidence=0.98,
            ),
            TurnUnderstanding(
                intent=ParentIntent.DISCUSS_CONCERN, confidence=0.94,
            ),
        ])
        service = object.__new__(TwoWayAIVoiceService)
        service._calendar = self.calendar
        service._phone = "12345"
        service._ngrok_url = "https://example.test"
        service._ai_ready = True
        service._intent_interpreter = interpreter
        service._conversations = {payload.registration: state}

        service.generate_followup_twiml(
            payload.registration, "Please make it quick because I don't have much time."
        )
        twiml = service.generate_followup_twiml(
            payload.registration, "She was sick during the science examination."
        )

        self.assertTrue(state.brief_mode)
        self.assertIn("Thank you for explaining", twiml)
        self.assertIn("monitor this closely", twiml)
        self.assertNotIn("noticed anything at home", twiml)


if __name__ == "__main__":
    unittest.main()
