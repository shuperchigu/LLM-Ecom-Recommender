-- ვშლით ძველ ცხრილებს, თუ არსებობენ, რომ თავიდან დავწეროთ
DROP TABLE IF EXISTS profiling_subcategory_details;
DROP TABLE IF EXISTS profiling_category_details;
DROP TABLE IF EXISTS profiling_batch_runs;

-- ვქმნით ცხრილს პროცესის გაშვების (batch) აღსარიცხად
CREATE TABLE profiling_batch_runs (
    id SERIAL PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    processed_users_count INT,
    duration_seconds FLOAT,
    algorithm_version VARCHAR(50)
);

-- ვქმნით მთავარი კატეგორიების დეტალების ცხრილს
CREATE TABLE profiling_category_details (
    id SERIAL PRIMARY KEY,
    batch_run_id INT REFERENCES profiling_batch_runs(id),
    user_id INT NOT NULL,
    category_name VARCHAR(255) NOT NULL,
    score FLOAT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (batch_run_id, user_id, category_name) -- უნიკალურობის უზრუნველყოფა
);

-- ვქმნით ქვეკატეგორიების დეტალების ცხრილს
CREATE TABLE profiling_subcategory_details (
    id SERIAL PRIMARY KEY,
    batch_run_id INT REFERENCES profiling_batch_runs(id),
    user_id INT NOT NULL,
    category_name VARCHAR(255) NOT NULL,
    subcategory_name VARCHAR(255) NOT NULL,
    score FLOAT NOT NULL,
    preferred_brands TEXT[], -- ბრენდების მასივი
    price_range_min NUMERIC(10, 2),
    price_range_max NUMERIC(10, 2),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (batch_run_id, user_id, subcategory_name) -- უნიკალურობის უზრუნველყოფა
);

-- ინდექსები სწრაფი ძებნისთვის
CREATE INDEX idx_prof_cat_user_id ON profiling_category_details(user_id);
CREATE INDEX idx_prof_subcat_user_id ON profiling_subcategory_details(user_id);