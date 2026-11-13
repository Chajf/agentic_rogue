CREATE TABLE pilot_seeds (
    seed bigint PRIMARY KEY CHECK (seed BETWEEN 0 AND 4294967295),
    split text NOT NULL CHECK (split IN ('development', 'pilot_evaluation')),
    UNIQUE (seed, split)
);

CREATE TABLE pilot_series (
    id uuid PRIMARY KEY,
    split text NOT NULL CHECK (split IN ('development', 'pilot_evaluation')),
    manifest jsonb NOT NULL CHECK (jsonb_typeof(manifest) = 'object'),
    started_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz,
    stop_reason text,
    UNIQUE (id, split)
);

CREATE TABLE pilot_trials (
    series_id uuid NOT NULL,
    trial_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    split text NOT NULL,
    requested_seed bigint NOT NULL,
    context_policy text NOT NULL CHECK (context_policy IN ('history_only', 'working_memory')),
    session_id uuid UNIQUE REFERENCES game_sessions(id),
    outcome text NOT NULL DEFAULT 'planned'
        CHECK (outcome IN ('planned', 'running', 'ended', 'limited', 'failed', 'interrupted', 'unstarted')),
    reason text,
    started_at timestamptz,
    ended_at timestamptz,
    PRIMARY KEY (series_id, trial_id),
    UNIQUE (series_id, ordinal),
    FOREIGN KEY (series_id, split) REFERENCES pilot_series(id, split),
    FOREIGN KEY (requested_seed, split) REFERENCES pilot_seeds(seed, split)
);
