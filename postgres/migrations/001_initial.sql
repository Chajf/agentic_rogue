CREATE TABLE game_sessions (
    id uuid PRIMARY KEY,
    episode_id uuid UNIQUE,
    seed bigint CHECK (seed BETWEEN 0 AND 4294967295),
    status text NOT NULL CHECK (status IN ('starting', 'running', 'won', 'dead', 'quit', 'error', 'interrupted', 'limit_reached')),
    model_name text NOT NULL,
    prompt_version text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz,
    error text
);

CREATE TABLE model_calls (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES game_sessions(id),
    input_messages jsonb NOT NULL CHECK (jsonb_typeof(input_messages) = 'array'),
    raw_response jsonb,
    parsed_action jsonb,
    decision_rationale text,
    validation_error text,
    response_metadata jsonb,
    usage_metadata jsonb,
    duration_ms integer CHECK (duration_ms >= 0),
    status text NOT NULL CHECK (status IN ('pending', 'valid', 'invalid', 'error')),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX model_calls_session_order_idx ON model_calls(session_id, id);

CREATE TABLE game_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES game_sessions(id),
    model_call_id bigint REFERENCES model_calls(id),
    kind text NOT NULL CHECK (kind IN ('reset', 'action')),
    action_request jsonb,
    api_step integer CHECK (api_step >= 0),
    http_status integer,
    observation jsonb,
    outcome text NOT NULL CHECK (outcome IN ('pending', 'succeeded', 'failed', 'uncertain')),
    error text,
    requested_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    CHECK ((kind = 'reset' AND action_request IS NULL AND model_call_id IS NULL)
        OR (kind = 'action' AND action_request IS NOT NULL))
);

CREATE INDEX game_events_session_order_idx ON game_events(session_id, id);
CREATE INDEX game_events_api_step_idx ON game_events(session_id, api_step)
    WHERE api_step IS NOT NULL;
