ALTER TABLE game_sessions
    ADD COLUMN context_policy text CHECK (context_policy IN ('history_only', 'working_memory')),
    ADD CONSTRAINT game_sessions_id_episode_key UNIQUE (id, episode_id);

ALTER TABLE model_calls
    ADD COLUMN purpose text NOT NULL DEFAULT 'decision'
        CHECK (purpose IN ('decision', 'episode_summary')),
    ADD COLUMN input_token_count integer CHECK (input_token_count >= 0),
    ADD COLUMN input_token_count_method text,
    ADD CONSTRAINT model_calls_session_id_id_key UNIQUE (session_id, id);

CREATE TABLE episode_memory_revisions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL,
    episode_id uuid NOT NULL,
    previous_revision_id bigint,
    through_event_id bigint NOT NULL,
    schema_version integer NOT NULL,
    policy_version text NOT NULL,
    content jsonb NOT NULL CHECK (jsonb_typeof(content) = 'object'),
    gap_event_ids bigint[] NOT NULL,
    model_call_id bigint NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, id),
    UNIQUE (session_id, through_event_id),
    FOREIGN KEY (session_id, episode_id) REFERENCES game_sessions(id, episode_id),
    FOREIGN KEY (session_id, through_event_id) REFERENCES game_events(session_id, id),
    FOREIGN KEY (session_id, previous_revision_id) REFERENCES episode_memory_revisions(session_id, id),
    FOREIGN KEY (session_id, model_call_id) REFERENCES model_calls(session_id, id)
);

ALTER TABLE model_calls
    ADD COLUMN memory_revision_id bigint,
    ADD CONSTRAINT model_calls_memory_revision_fk
        FOREIGN KEY (session_id, memory_revision_id) REFERENCES episode_memory_revisions(session_id, id);

ALTER TABLE game_events
    ADD COLUMN progress jsonb CHECK (jsonb_typeof(progress) = 'object');
