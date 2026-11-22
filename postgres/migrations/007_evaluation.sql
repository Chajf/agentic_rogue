ALTER TABLE pilot_seeds
    DROP CONSTRAINT pilot_seeds_split_check,
    ADD CONSTRAINT pilot_seeds_split_check CHECK (split IN ('development', 'pilot_evaluation', 'held_out'));

ALTER TABLE pilot_series
    DROP CONSTRAINT pilot_series_split_check,
    ADD CONSTRAINT pilot_series_split_check CHECK (split IN ('development', 'pilot_evaluation', 'held_out'));

ALTER TABLE game_events
    ADD COLUMN scripted_input_event_id bigint,
    ADD CONSTRAINT game_events_scripted_input_fk
        FOREIGN KEY (session_id, scripted_input_event_id) REFERENCES game_events(session_id, id),
    ADD CONSTRAINT game_events_scripted_action_check CHECK (
        scripted_input_event_id IS NULL OR (
            kind = 'action' AND model_call_id IS NULL AND scripted_input_event_id < id
        )
    );
