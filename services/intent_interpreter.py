"""Context-aware parent intent classification using Groq local tool calling."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ParentIntent(str, Enum):
    CONFIRM_IDENTITY = "confirm_identity"
    AVAILABLE_TO_TALK = "available_to_talk"
    AVAILABLE_BRIEFLY = "available_briefly"
    DISCUSS_CONCERN = "discuss_concern"
    SCHEDULE_MEETING = "schedule_meeting"
    SELECT_SLOT = "select_slot"
    CHECK_AVAILABILITY = "check_availability"
    RESCHEDULE_MEETING = "reschedule_meeting"
    CANCEL_MEETING = "cancel_meeting"
    CONFIRM_ACTION = "confirm_action"
    DECLINE_ACTION = "decline_action"
    ASK_CURRENT_DATE = "ask_current_date"
    END_CALL = "end_call"
    OTHER = "other"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class TurnUnderstanding:
    intent: ParentIntent
    confidence: float
    target_date: str | None = None
    target_time: str | None = None
    source_time: str | None = None
    range_start: str | None = None
    range_end: str | None = None
    selected_option: int | None = None
    requires_clarification: bool = False
    sentiment: str = "neutral"
    reasoning: str = ""
    assistant_reply: str | None = None
    interpreted: bool = True

    @classmethod
    def unclear(cls, reasoning: str = "") -> "TurnUnderstanding":
        return cls(
            intent=ParentIntent.UNCLEAR,
            confidence=0.0,
            requires_clarification=True,
            reasoning=reasoning,
            interpreted=False,
        )


ROUTE_PARENT_TURN_TOOL = {
    "type": "function",
    "function": {
        "name": "route_parent_turn",
        "description": (
            "Interpret the parent's current turn using the supplied conversation state. "
            "This tool classifies intent and extracts scheduling details; it never executes an action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [intent.value for intent in ParentIntent],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "target_date": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Resolved destination date as YYYY-MM-DD; omit when unknown.",
                },
                "target_time": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Requested destination time as HH:MM in 24-hour time; omit when unknown.",
                },
                "source_time": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Existing/source meeting time as HH:MM when explicitly mentioned.",
                },
                "range_start": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Requested range start as YYYY-MM-DD, or null.",
                },
                "range_end": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Requested range end as YYYY-MM-DD, or null.",
                },
                "selected_option": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": 3},
                        {"type": "null"},
                    ],
                },
                "requires_clarification": {"type": "boolean"},
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "worried", "frustrated", "hesitant"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "One short explanation grounded in the conversation state.",
                },
                "assistant_reply": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Natural spoken reply for non-calendar turns; null for calendar actions.",
                },
            },
            "required": [
                "intent", "confidence", "target_date", "target_time", "source_time",
                "range_start", "range_end", "selected_option", "requires_clarification",
                "sentiment", "reasoning", "assistant_reply",
            ],
        },
    },
}


class ContextualIntentInterpreter:
    MIN_ACTION_CONFIDENCE = 0.72

    def __init__(self, groq_client, model: str, mode: str = "json"):
        self._groq = groq_client
        self._model = model
        self._mode = mode

    def interpret(self, parent_speech: str, context: dict[str, Any]) -> TurnUnderstanding:
        if self._mode != "tool":
            return self._interpret_json_fallback(parent_speech, context)
        user_content = (
            "LATEST PARENT SPEECH:\n"
            f"{parent_speech}\n\n"
            "CONVERSATION STATE:\n"
            f"{json.dumps(context, ensure_ascii=True)}\n\n"
            "Call route_parent_turn with flat arguments matching its schema. "
            "Do not pass parent_speech or conversation_context as arguments. "
            "Use null for unknown dates, times, ranges, selected option, and reply."
        )
        try:
            response = self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a routing component for a school phone assistant. "
                            "Interpret the latest parent turn in context. Resolve words such as yes, no, "
                            "it, that, instead, and another time from the last agent message and active action. "
                            "Classify available_briefly only when the latest speech itself says the parent "
                            "is time-limited or asks you to be quick; never carry that intent into later turns. "
                            "Classify questions asking today's/current date as ask_current_date. "
                            "For rescheduling, distinguish the old/source time from the new/target time. "
                            "When only a new time is supplied, inherit the active booking date as target_date. "
                            "Call route_parent_turn exactly once. Do not answer the parent."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                tools=[ROUTE_PARENT_TURN_TOOL],
                tool_choice={"type": "function", "function": {"name": "route_parent_turn"}},
                temperature=0,
                max_tokens=220,
            )
            calls = response.choices[0].message.tool_calls or []
            if not calls:
                return self._interpret_json_fallback(parent_speech, context)
            arguments = calls[0].function.arguments
            data = json.loads(arguments) if isinstance(arguments, str) else arguments
            return _validate_understanding(data)
        except Exception as exc:
            logger.warning("Intent interpretation failed: %s", exc)
            return self._interpret_json_fallback(parent_speech, context)

    def _interpret_json_fallback(
        self, parent_speech: str, context: dict[str, Any]
    ) -> TurnUnderstanding:
        try:
            response = self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return one JSON object only. Classify the latest parent speech using the "
                            "conversation state. Required keys: intent, confidence, target_date, "
                            "target_time, source_time, range_start, range_end, selected_option, "
                            "requires_clarification, sentiment, reasoning, assistant_reply. Use null "
                            "for unknown entities. For non-calendar turns, assistant_reply must be a "
                            "natural one- or two-sentence English response grounded only in the student "
                            "facts. Its final sentence must be a relevant question unless it is the final "
                            "goodbye. For calendar actions use null. Never claim a calendar action succeeded. "
                            "For discuss_concern, acknowledge the concrete information in the latest "
                            "parent statement, summarize it briefly, and move to a suitable next step. "
                            "Never repeat the question in last_agent_message or ask the parent to explain "
                            "the same concern again. For MEDIUM risk, do not mention, suggest, or ask about "
                            "a meeting unless the latest parent speech itself requests one. The default "
                            "MEDIUM outcome is monitoring and school follow-up if the concern continues. "
                            "For HIGH risk, acknowledge the concern and ask whether the parent wants the "
                            "earliest verified teacher meeting options. For LOW risk, do not suggest a "
                            "meeting; give encouragement and a practical check-in question. "
                            "Valid intents: "
                            + ", ".join(intent.value for intent in ParentIntent)
                            + ". Interpret available_briefly only from the latest speech. Resolve "
                            "today's date questions as ask_current_date. If the latest speech explicitly "
                            "agrees to hold or schedule a meeting, use schedule_meeting rather than "
                            "confirm_action."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "latest_parent_speech": parent_speech,
                            "conversation_state": context,
                        }, ensure_ascii=True),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=220,
            )
            return _validate_understanding(json.loads(response.choices[0].message.content))
        except Exception as exc:
            logger.warning("JSON intent fallback failed: %s", exc)
            return TurnUnderstanding.unclear(str(exc))


def _validate_understanding(data: Any) -> TurnUnderstanding:
    if not isinstance(data, dict):
        return TurnUnderstanding.unclear("Tool arguments were not an object.")
    try:
        intent = ParentIntent(data.get("intent", "unclear"))
    except ValueError:
        intent = ParentIntent.UNCLEAR
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0

    target_date = _valid_date(data.get("target_date"))
    target_time = _valid_time(data.get("target_time"))
    source_time = _valid_time(data.get("source_time"))
    range_start = _valid_date(data.get("range_start"))
    range_end = _valid_date(data.get("range_end"))
    option = data.get("selected_option")
    option = option if isinstance(option, int) and 1 <= option <= 3 else None
    sentiment = data.get("sentiment", "neutral")
    if sentiment not in {"positive", "neutral", "worried", "frustrated", "hesitant"}:
        sentiment = "neutral"
    assistant_reply = data.get("assistant_reply")
    if not isinstance(assistant_reply, str) or not assistant_reply.strip():
        assistant_reply = None
    else:
        assistant_reply = assistant_reply.strip()[:500]

    return TurnUnderstanding(
        intent=intent,
        confidence=confidence,
        target_date=target_date,
        target_time=target_time,
        source_time=source_time,
        range_start=range_start,
        range_end=range_end,
        selected_option=option,
        requires_clarification=bool(data.get("requires_clarification", False)),
        sentiment=sentiment,
        reasoning=str(data.get("reasoning", ""))[:240],
        assistant_reply=assistant_reply,
        interpreted=True,
    )


def _valid_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _valid_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", value)
    return value if match else None
