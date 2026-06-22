create table if not exists calls (
    call_sid varchar(64) primary key,
    registration varchar(100) not null,
    student_name varchar(200) not null,
    parent_name varchar(200) not null,
    parent_phone_masked varchar(32) not null default '',
    dimension varchar(40) not null,
    risk_level varchar(20) not null,
    call_status varchar(30) not null default 'initiated',
    started_at timestamptz not null,
    answered_at timestamptz,
    ended_at timestamptz,
    duration_seconds integer,
    summary_status varchar(30) not null default 'pending',
    brief_summary text,
    parent_concerns jsonb not null default '[]'::jsonb,
    school_observations jsonb not null default '[]'::jsonb,
    agreed_actions jsonb not null default '[]'::jsonb,
    unresolved_questions jsonb not null default '[]'::jsonb,
    follow_up_required boolean not null default false,
    parent_sentiment varchar(40),
    meeting_status varchar(30) not null default 'none',
    meeting_event_id varchar(255),
    meeting_start timestamptz,
    meeting_end timestamptz,
    summary_error text
);

create index if not exists ix_calls_registration on calls (registration);
create index if not exists ix_calls_started_at on calls (started_at desc);
create index if not exists ix_calls_dimension on calls (dimension);
create index if not exists ix_calls_risk_level on calls (risk_level);

create table if not exists conversation_turns (
    id bigserial primary key,
    call_sid varchar(64) not null references calls(call_sid) on delete cascade,
    turn_number integer not null,
    speaker varchar(20) not null,
    message text not null,
    interpreted_intent varchar(60),
    conversation_stage varchar(40),
    created_at timestamptz not null,
    constraint uq_call_turn_number unique (call_sid, turn_number)
);

create index if not exists ix_turns_call_sid on conversation_turns (call_sid);
create index if not exists ix_turns_call_order on conversation_turns (call_sid, turn_number);
