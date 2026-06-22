# EduGuardian AI Voice

EduGuardian calls parents about attendance, performance, and behavior concerns. Its
meeting flow checks a teacher's calendar, offers verified openings, and creates a
booking only after the parent selects a time.

## Calendar configuration

The zero-setup default stores bookings in `data/teacher_calendar.json`:

```env
CALENDAR_PROVIDER=local
DEFAULT_TEACHER_ID=default
SCHOOL_TIMEZONE=Asia/Kolkata
MEETING_HOURS_START=09:00
MEETING_HOURS_END=12:00
MEETING_DURATION_MINUTES=30
MEETING_WORKING_DAYS=mon,tue,wed,thu,fri
```

For a live Google Calendar, create a Google Cloud service account, enable the
Google Calendar API, and share each teacher calendar with the service account with
permission to add events. Then configure:

```env
CALENDAR_PROVIDER=google
GOOGLE_SERVICE_ACCOUNT_FILE=C:\path\to\service-account.json
DEFAULT_TEACHER_ID=teacher@example.com
```

`DEFAULT_TEACHER_ID` is the Google Calendar ID, normally the teacher's email
address. Different teachers can later be supplied per `CallPayload.teacher_id`.

## Scheduling behavior

Action-like parent turns pass through a contextual intent interpreter using Groq
JSON mode. The interpreter receives the conversation stage, last agent message,
active booking, offered slots, pending action, recent turns, current date, and
timezone. It returns a validated intent and scheduling entities but has no
authority to modify Google Calendar. Ordinary discussion skips the extra routing
request to keep voice webhooks within Twilio's response deadline.

Calendar changes remain deterministic: Python validates the interpreted action,
checks policy and live availability, executes the calendar operation, and only
then produces a confirmation. Low-confidence actions cause a clarification
question. Existing phrase and date parsers remain only as an availability
fallback when contextual interpretation fails.

If a parent says they are busy but asks the teacher to be quick, the interpreter
uses `available_briefly`: the agent gives a concise concern summary instead of
ending the call. Brief mode remains active for later turns, preventing additional
exploratory questions. High-risk summaries ask whether the parent wants the
earliest meeting options.

Groq JSON output accepts null scheduling entities and is validated locally. The
older tool-call mode remains available for testing, with JSON mode as its fallback.

Current-date questions are answered directly from the configured school timezone.
Day-of-month requests such as `meetings on the 23rd` or `meetings at 23` resolve
to a calendar date and never to 23:00.

Meeting windows are risk-based and enforced in Python:

- High risk: offer verified slots within the next two calendar days. Later
  requests are directed to the school.
- Medium risk: offer slots only after the parent asks for a meeting, within the
  next seven calendar days. Later requests are directed to the school.
- Low risk: do not automatically schedule; direct exceptional requests to the
  school.

Meeting turns are interpreted with the full conversation state, including the
active booking, previously offered choices, and the teacher's verified calendar.
Natural confirmations such as "the earliest one suits us" select an offered slot;
date corrections such as "not Tuesday, Wednesday instead" use the corrected day.
Relative ranges including next week, the week after next, and next month are
resolved before the risk window is enforced. Rescheduling updates the original
calendar event instead of creating a second meeting.

A date-only move, such as Monday to Tuesday, does not modify the existing event.
It first offers the teacher's free times on Tuesday and patches the original event
only after the parent selects one.

After a booking, a short date or time correction such as `Tuesday` or `11 AM
instead` is treated as a reschedule even when the parent does not repeat the word
`reschedule`. Short affirmations such as `sure` book automatically when exactly
one slot is pending; when several slots are available, the agent asks the parent
to choose one rather than guessing.

The voice service supports creating, rescheduling, and cancelling meetings during
the same call. A reschedule updates the original Google Calendar event; it does
not create a second event. Every create or move rechecks availability immediately
before changing the calendar.

Recognized date expressions include `today`, `tomorrow`, `day after tomorrow`,
weekday names, `next Monday`, `June 22`, `22 June 2026`, `22/06/2026`, and
`2026-06-22`. Recognized times include `at 10`, `at ten`, `9 AM`, `9:30 AM`,
and `11.00 AM`. Parents can also select the `first`, `second`, or `third` offered
option.

Working days, meeting hours, duration, current time, and existing busy calendar
events are all applied before a slot is offered. Incomplete requests produce a
clarification question, while non-working days and out-of-hours requests produce
a policy explanation followed by valid alternatives.

Install dependencies and run the focused tests with:

```powershell
pip install -r requirements.txt
python -m unittest tests.test_calendar_service
```
