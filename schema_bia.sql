
----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------- CREATING TABLES FOR "BUY IT AGAIN" RECOMMENDATION SYSTEM --------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------


-- schema_bia.sql

-- წავშალოთ ცხრილები საპირისპირო თანმიმდევრობით, დამოკიდებულებების გამო
DROP TABLE IF EXISTS bia_recommendations;
DROP TABLE IF EXISTS bia_sessions;
DROP TABLE IF EXISTS bia_batch_runs;

-- ცხრილი თითოეული გაშვების (batch) მეტა-მონაცემებისთვის
CREATE TABLE bia_batch_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    algorithm_version VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'Running',
    processed_users_count INT DEFAULT 0,
    total_recommendations_generated INT DEFAULT 0,
    duration_seconds REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    error_message TEXT
);
COMMENT ON TABLE bia_batch_runs IS 'Tracks each execution run of the Buy It Again recommender.';

-- ცხრილი თითოეული მომხმარებლის დამუშავების სესიისთვის კონკრეტული გაშვების ფარგლებში
CREATE TABLE bia_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_run_id UUID NOT NULL REFERENCES bia_batch_runs(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    recommendations_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_bia_sessions_batch_run_id ON bia_sessions(batch_run_id);
CREATE INDEX idx_bia_sessions_user_id ON bia_sessions(user_id);
COMMENT ON TABLE bia_sessions IS 'Tracks the recommendation generation session for a single user within a batch run.';


-- ცხრილი გენერირებული რეკომენდაციებისთვის (განახლებული სტრუქტურა)
CREATE TABLE bia_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_run_id UUID NOT NULL REFERENCES bia_batch_runs(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES bia_sessions(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL, -- <<< დამატებულია user_id
    product_id BIGINT NOT NULL,
    rank INT NOT NULL,
    last_purchase_date DATE,
    predicted_replenish_date DATE,
    days_overdue INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_bia_recs_session_id ON bia_recommendations(session_id);
CREATE INDEX idx_bia_recs_product_id ON bia_recommendations(product_id);
CREATE INDEX idx_bia_recs_user_id ON bia_recommendations(user_id); -- <<< დამატებულია ინდექსი user_id-ზე
COMMENT ON TABLE bia_recommendations IS 'Stores the final ranked Buy It Again recommendations for each user session.';