ALTER TABLE game_events
    ADD CONSTRAINT game_events_session_id_id_key UNIQUE (session_id, id);

ALTER TABLE model_calls
    ADD COLUMN observation_event_id bigint,
    ADD CONSTRAINT model_calls_observation_event_fk
        FOREIGN KEY (session_id, observation_event_id) REFERENCES game_events(session_id, id);
