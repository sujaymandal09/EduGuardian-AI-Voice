"""
mock_call_test.py
Simulates a full parent call without Twilio.
Directly drives generate_followup_twiml turn by turn and prints what Priya says.
"""
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

from core.models import CallPayload
from services.twilio_groq_voice import TwoWayAIVoiceService, ConversationState

REGISTRATION = "MOCK_TEST_001"

PAYLOAD = CallPayload(
    to_number="+919999999999",
    registration=REGISTRATION,
    student_name="Neel Banerjee",
    parent_name="Tapas Banerjee",
    dimension="attendance",
    risk_level="HIGH",
    details=(
        "Neel has attended only 58 out of 90 classes this semester — 64% attendance. "
        "He has been absent for 6 consecutive days this month. "
        "He risks being barred from the end-term exams if attendance drops below 75%."
    ),
    recommended_action="Immediate face-to-face meeting with parents and student."
)

TURNS = [
    "Yes, this is Tapas Banerjee.",
    "Yes I have time, please go ahead.",
    "Oh I see. He told me he was attending. What exactly are the numbers?",
    "That is quite bad. We had no idea. What can we do now?",
    "Yes a meeting sounds good. Tomorrow at ten works for me.",
    "No, I think that covers it. Thank you so much.",
    "Goodbye.",
]


def extract_spoken(twiml: str) -> list[str]:
    return re.findall(r'<Say[^>]*>(.*?)</Say>', twiml, re.DOTALL)


def run():
    print("\n" + "=" * 64)
    print("  MOCK CALL — Neel Banerjee / Tapas Banerjee [ATTENDANCE HIGH]")
    print("=" * 64)

    service = TwoWayAIVoiceService()
    state = ConversationState(PAYLOAD, service._school, service._phone)
    service._conversations[REGISTRATION] = state

    print(f"\nPriya (opening): \"Hello! Am I speaking with {PAYLOAD.parent_name}?\"\n")

    for i, speech in enumerate(TURNS):
        print(f"[Turn {i+1}] Parent : \"{speech}\"")
        t0 = time.time()
        twiml = service.generate_followup_twiml(REGISTRATION, speech)
        elapsed = int((time.time() - t0) * 1000)

        lines = extract_spoken(twiml)
        for line in lines:
            print(f"         Priya  : \"{line.strip()}\"")
        print(f"         [{elapsed}ms]")

        if state.ended:
            print("\n[CALL ENDED]")
            break
        print()

    print("\n" + "=" * 64)
    print(f"  Turns completed : {state.turn_count}")
    print(f"  Final stage     : {state.stage}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    run()
