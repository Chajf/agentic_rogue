ALTER TABLE model_calls
    DROP CONSTRAINT model_calls_purpose_check,
    ADD CONSTRAINT model_calls_purpose_check CHECK (purpose IN ('decision', 'episode_summary', 'goal_selection')),
    ADD COLUMN decision_metadata jsonb CHECK (jsonb_typeof(decision_metadata) = 'object');
