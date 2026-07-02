# Amazon RDS PostgreSQL setup

## 1. Create the database

In AWS Console, open **RDS**, choose **Create database**, select PostgreSQL,
and use the Free tier or the smallest development template available to your
account. Suggested values:

- DB identifier: `eduguardian-db`
- Initial database name: `eduguardian`
- Master username: `eduguardian_admin`
- Storage encryption: enabled
- Automated backups: enabled

Keep the password in a password manager. Do not commit it to Git.

## 2. Choose connectivity

For testing while Flask runs on the current PC through ngrok, RDS must be
reachable from that PC. Enable public access temporarily and configure its
security group to allow PostgreSQL TCP port 5432 from **your public IP only**.
Never leave port 5432 open to `0.0.0.0/0`.

For production on AWS, place RDS in private subnets and allow port 5432 only
from the security group attached to the application service (EC2, ECS, or
Elastic Beanstalk). Disable public access.

## 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Apply the schema

Install PostgreSQL `psql`, then run:

```powershell
$env:PGPASSWORD="your-password"
psql -h your-rds-endpoint.amazonaws.com -U eduguardian_admin -d eduguardian -f migrations/001_call_history.sql
```

The application also calls SQLAlchemy `create_all` at startup, but applying the
versioned SQL file explicitly is recommended.

## 5. Configure the application

Add to `.env` locally, or to the hosting provider's secret variables:

```env
CALL_HISTORY_ENABLED=true
DATABASE_URL=postgresql+psycopg://eduguardian_admin:URL_ENCODED_PASSWORD@your-rds-endpoint.amazonaws.com:5432/eduguardian?sslmode=require
```

If the password contains `@`, `:`, `/`, `?`, or `#`, URL-encode it. Generate an
encoded value with:

```powershell
python -c "import urllib.parse; print(urllib.parse.quote_plus(input('Password: ')))"
```

## 6. Verify locally

Start Flask and open `/calls`. It should show an empty call-history page instead
of the disabled message. Make one controlled Twilio call. When it ends, Twilio
posts to `/twilio/call-status`, and the summary status should move from pending
to generating to completed.

Verify the database directly:

```sql
select call_sid, student_name, call_status, summary_status
from calls
order by started_at desc;

select call_sid, turn_number, speaker, message
from conversation_turns
order by call_sid, turn_number;
```

## 7. Production safeguards

Before real school use, protect `/calls` with staff authentication, validate
Twilio signatures, remove full transcript logging, use AWS Secrets Manager for
the database password, and define a transcript retention/deletion policy.
