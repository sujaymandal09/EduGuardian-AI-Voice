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
