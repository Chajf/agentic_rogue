ALTER TABLE model_calls
    ADD COLUMN history_window jsonb CHECK (jsonb_typeof(history_window) = 'object');
