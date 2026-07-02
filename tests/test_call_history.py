import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from core.models import CallPayload
from services.call_history import SqlCallHistoryRepository
from services.twilio_groq_voice import ConversationState, TwoWayAIVoiceService

try:
    import sqlalchemy  # noqa: F401
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class FakeCallHistory:
    enabled = True

    def __init__(self):
        self.created = []
        self.turns = []
        self.statuses = []
        self.meetings = []
        self.summary_status = "pending"
        self.summary = None
        self.summary_error = None

    def create_call(self, call_sid, payload, opening):
        self.created.append((call_sid, payload, opening))
        self.turns.append({"speaker": "agent", "message": opening})

    def append_turn(self, call_sid, speaker, message, **kwargs):
        self.turns.append({"speaker": speaker, "message": message, **kwargs})

    def update_status(self, call_sid, status, duration=None):
        self.statuses.append((call_sid, status, duration))

    def update_meeting(self, call_sid, booking, status):
        self.meetings.append((call_sid, booking, status))

    def claim_summary(self, call_sid):
        if self.summary_status not in {"pending", "failed"}:
            return False
        self.summary_status = "generating"
        return True

    def get_turns(self, call_sid):
        return self.turns

    def get_call(self, call_sid):
        return {
            "student_name": "Saanvi",
            "dimension": "performance",
            "risk_level": "MEDIUM",
        }

    def save_summary(self, call_sid, summary):
        self.summary_status = "completed"
        self.summary = summary

    def fail_summary(self, call_sid, error):
        self.summary_status = "failed"
        self.summary_error = error


class CallHistoryIntegrationTests(unittest.TestCase):
    def payload(self):
        return CallPayload(
            to_number="+919999991234", registration="REG-HISTORY",
            student_name="Saanvi", parent_name="Neha",
            dimension="performance", risk_level="MEDIUM",
            details="Science performance declined.",
        )

    def test_make_call_creates_record_and_configures_completion_callback(self):
        history = FakeCallHistory()
        service = object.__new__(TwoWayAIVoiceService)
        service._twilio_ready = True
        service._school = "Test School"
        service._phone = "12345"
        service._ngrok_url = "https://voice.example.test"
        service._from_number = "+910000000099"
        service._conversations = {}
        service._call_history = history
        service.calls_made = []
        service._client = MagicMock()
        service._client.calls.create.return_value = SimpleNamespace(sid="CA-HISTORY-1")

        result = service.make_call(self.payload())

        self.assertTrue(result.success)
        self.assertEqual(history.created[0][0], "CA-HISTORY-1")
        options = service._client.calls.create.call_args.kwargs
        self.assertEqual(
            options["status_callback"],
            "https://voice.example.test/twilio/call-status",
        )
        self.assertIn("completed", options["status_callback_event"])
        self.assertEqual(service._conversations["CA-HISTORY-1"].call_sid, "CA-HISTORY-1")

    def test_tracked_reply_is_persisted(self):
        history = FakeCallHistory()
        state = ConversationState(self.payload(), "Test School", "12345")
        state.call_sid = "CA-HISTORY-2"
        service = object.__new__(TwoWayAIVoiceService)
        service._call_history = history
        service._ngrok_url = "https://voice.example.test"
        service._phone = "12345"

        service._tracked_twiml(state, "How can we support Saanvi?")

        self.assertEqual(history.turns[-1]["speaker"], "agent")
        self.assertEqual(history.turns[-1]["message"], "How can we support Saanvi?")

    def test_parent_and_agent_turns_are_both_persisted(self):
        history = FakeCallHistory()
        payload = self.payload()
        state = ConversationState(payload, "Test School", "12345")
        state.call_sid = "CA-HISTORY-PARENT"
        service = object.__new__(TwoWayAIVoiceService)
        service._call_history = history
        service._calendar = SimpleNamespace(timezone=ZoneInfo("Asia/Kolkata"))
        service._phone = "12345"
        service._ngrok_url = "https://voice.example.test"
        service._ai_ready = True
        service._conversations = {state.call_sid: state}
        service._intent_interpreter = MagicMock()
        service._intent_interpreter.interpret.return_value = SimpleNamespace(
            intent=SimpleNamespace(value="other"), confidence=1.0,
            target_date=None, target_time=None, source_time=None,
            range_start=None, range_end=None, selected_option=None,
            requires_clarification=False, sentiment="neutral", reasoning="",
            assistant_reply="Thank you for explaining. What would help most at home?",
            interpreted=True,
        )
        service._interpret_parent_turn = MagicMock(return_value=
            service._intent_interpreter.interpret.return_value
        )
        service._route_contextual_action = MagicMock(return_value=None)

        service.generate_followup_twiml(
            state.call_sid, "We are already arranging extra coaching."
        )

        self.assertEqual(history.turns[-2]["speaker"], "parent")
        self.assertEqual(
            history.turns[-2]["message"],
            "We are already arranging extra coaching.",
        )
        self.assertEqual(history.turns[-1]["speaker"], "agent")

    def test_summary_uses_persisted_transcript_and_is_idempotent(self):
        history = FakeCallHistory()
        history.turns = [
            {"speaker": "agent", "message": "How can we help?"},
            {"speaker": "parent", "message": "Please monitor her science work."},
        ]
        summary = {
            "brief_summary": "The parent requested monitoring.",
            "parent_concerns": ["Science work"],
            "school_observations": [],
            "agreed_actions": ["Monitor progress"],
            "unresolved_questions": [],
            "follow_up_required": True,
            "parent_sentiment": "concerned",
        }
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(summary))
        )])
        service = object.__new__(TwoWayAIVoiceService)
        service._call_history = history
        service._ai_ready = True
        service._groq = MagicMock()
        service._groq.chat.completions.create.return_value = response

        service.finalize_call_summary("CA-HISTORY-3")
        service.finalize_call_summary("CA-HISTORY-3")

        self.assertEqual(history.summary_status, "completed")
        self.assertEqual(history.summary["brief_summary"], summary["brief_summary"])
        self.assertEqual(service._groq.chat.completions.create.call_count, 1)


@unittest.skipUnless(
    SQLALCHEMY_AVAILABLE,
    "Install requirements.txt to run PostgreSQL/SQLite repository tests.",
)
class SqlCallHistoryRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = Path(self.temp_dir.name) / "call-history.db"
        self.repository = SqlCallHistoryRepository(f"sqlite:///{database.as_posix()}")
        self.payload = CallPayload(
            to_number="+919999991234", registration="REG-SQL",
            student_name="Saanvi", parent_name="Neha",
            dimension="performance", risk_level="MEDIUM",
            details="Science performance declined.",
        )

    def tearDown(self):
        self.repository.engine.dispose()
        self.temp_dir.cleanup()

    def test_complete_call_lifecycle_is_persisted(self):
        call_sid = "CA-SQL-1"
        self.repository.create_call(call_sid, self.payload, "Am I speaking with Neha?")
        self.repository.append_turn(call_sid, "parent", "Yes, please continue.")
        self.repository.append_turn(call_sid, "agent", "What have you noticed at home?")
        booking = SimpleNamespace(
            event_id="event-123",
            start=SimpleNamespace(),
            end=SimpleNamespace(),
        )
        from datetime import datetime, timezone
        booking.start = datetime(2026, 6, 23, 10, 0, tzinfo=timezone.utc)
        booking.end = datetime(2026, 6, 23, 10, 30, tzinfo=timezone.utc)
        self.repository.update_meeting(call_sid, booking, "booked")
        self.repository.update_status(call_sid, "answered")
        self.repository.update_status(call_sid, "completed", 95)

        self.assertTrue(self.repository.claim_summary(call_sid))
        self.assertFalse(self.repository.claim_summary(call_sid))
        self.repository.save_summary(call_sid, {
            "brief_summary": "The parent agreed to continued support.",
            "parent_concerns": ["Science performance"],
            "school_observations": ["Performance declined"],
            "agreed_actions": ["Meet the teacher"],
            "unresolved_questions": [],
            "follow_up_required": True,
            "parent_sentiment": "concerned",
        })

        call = self.repository.get_call(call_sid)
        turns = self.repository.get_turns(call_sid)
        self.assertEqual(call["parent_phone_masked"], "******1234")
        self.assertEqual(call["call_status"], "completed")
        self.assertEqual(call["duration_seconds"], 95)
        self.assertEqual(call["meeting_status"], "booked")
        self.assertEqual(call["meeting_event_id"], "event-123")
        self.assertEqual(call["summary_status"], "completed")
        self.assertEqual([turn["turn_number"] for turn in turns], [1, 2, 3])
        self.assertEqual([turn["speaker"] for turn in turns], ["agent", "parent", "agent"])

    def test_delete_calls_removes_selected_call_and_its_transcript_only(self):
        first_sid = "CA-SQL-DELETE-1"
        second_sid = "CA-SQL-DELETE-2"
        self.repository.create_call(first_sid, self.payload, "First opening")
        self.repository.append_turn(first_sid, "parent", "First answer")
        self.repository.create_call(second_sid, self.payload, "Second opening")

        deleted = self.repository.delete_calls([first_sid, first_sid, "missing"])

        self.assertEqual(deleted, 1)
        self.assertIsNone(self.repository.get_call(first_sid))
        self.assertEqual(self.repository.get_turns(first_sid), [])
        self.assertIsNotNone(self.repository.get_call(second_sid))
        self.assertEqual(len(self.repository.get_turns(second_sid)), 1)

    def test_reset_summary_allows_stuck_generation_to_be_retried(self):
        call_sid = "CA-SQL-RETRY"
        self.repository.create_call(call_sid, self.payload, "Opening")
        self.assertTrue(self.repository.claim_summary(call_sid))

        self.assertTrue(self.repository.reset_summary(call_sid))
        self.assertTrue(self.repository.claim_summary(call_sid))

    def test_completed_summary_can_be_cleared_and_regenerated(self):
        call_sid = "CA-SQL-REGENERATE"
        self.repository.create_call(call_sid, self.payload, "Opening")
        self.assertTrue(self.repository.claim_summary(call_sid))
        self.repository.save_summary(call_sid, {
            "brief_summary": "Old summary",
            "parent_concerns": ["Old concern"],
        })

        self.assertTrue(self.repository.reset_summary(call_sid))
        call = self.repository.get_call(call_sid)
        self.assertEqual(call["summary_status"], "pending")
        self.assertIsNone(call["brief_summary"])
        self.assertEqual(call["parent_concerns"], [])
        self.assertTrue(self.repository.claim_summary(call_sid))


if __name__ == "__main__":
    unittest.main()
