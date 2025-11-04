-- ====================================================================
-- სარეკომენდაციო სისტემის ანალიტიკური ცხრილების სქემა (ვერსია 2.0)
-- 
-- ცვლილებები:
-- 1. `batch_run_id` დამატებულია ყველა შვილობილ ცხრილში დენორმალიზაციის გზით,
--    რაც აჩქარებს და ამარტივებს ანალიტიკურ მოთხოვნებს.
-- 2. `batch_run_id` სვეტზე დამატებულია ინდექსები.
-- 3. `batch_run_id` და `session_id` ტიპი შეცვლილია UUID-ზე მეტი უნიკალურობისთვის.
-- 4. გამოყენებულია `ON DELETE CASCADE` რელაციური მთლიანობისთვის.
-- ====================================================================

-- --- დასუფთავების ბლოკი ---
-- წაშლის ძველ ცხრილებს, თუ არსებობს, რათა თავიდან ავიცილოთ კონფლიქტები.
DROP TABLE IF EXISTS 
    recommendations_final_list, 
    recommendations_subcategory_details, 
    recommendations_category_details, 
    recommendations_sessions,
    recommendations_batch_runs CASCADE;


-- --- ცხრილების შექმნა ---

-- ცხრილი 0: ბეჩის გაშვებები (მთავარი ცხრილი)
-- ინახავს ინფორმაციას სკრიპტის თითოეული სრული გაშვების შესახებ.
CREATE TABLE recommendations_batch_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_timestamp TIMESTAMPTZ DEFAULT NOW(),
    algorithm_version VARCHAR(50),
    processed_users_count INTEGER,
    duration_seconds FLOAT
);

COMMENT ON TABLE recommendations_batch_runs IS 'Stores metadata for each complete batch run of the recommender script.';
COMMENT ON COLUMN recommendations_batch_runs.id IS 'Unique UUID identifier for the entire batch run.';


-- ცხრილი 1: რეკომენდაციის სესიები (თითოეული მომხმარებლისთვის)
-- ეს ცხრილი აკავშირებს კონკრეტულ მომხმარებელს კონკრეტულ "ბეჩთან".
CREATE TABLE recommendations_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_run_id UUID NOT NULL REFERENCES recommendations_batch_runs(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    session_timestamp TIMESTAMPTZ DEFAULT NOW(),
    algorithm_version VARCHAR(50)
);

COMMENT ON TABLE recommendations_sessions IS 'Stores metadata for each individual user recommendation within a batch run.';
COMMENT ON COLUMN recommendations_sessions.batch_run_id IS 'Foreign key to the recommendations_batch_runs table, grouping sessions by run.';


-- ცხრილი 2: კატეგორიების დეტალები სესიაზე
-- ინახავს, თუ რა ინტერესის ქულა და რამდენი სლოტი გამოიყო თითოეული მთავარი კატეგორიისთვის.
CREATE TABLE recommendations_category_details (
    id SERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL, -- დენორმალიზაცია ანალიტიკისთვის
    session_id UUID NOT NULL REFERENCES recommendations_sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    category_name VARCHAR(255) NOT NULL,
    score FLOAT,
    slots_allocated INTEGER
);

COMMENT ON TABLE recommendations_category_details IS 'Stores details about main categories for each session. Denormalized with batch_run_id for faster queries.';


-- ცხრილი 3: ქვეკატეგორიების დეტალები სესიაზე
-- ინახავს დეტალურ პროფილს თითოეული ქვეკატეგორიისთვის: ქულა, ბრენდები, ფასის დიაპაზონი.
CREATE TABLE recommendations_subcategory_details (
    id SERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL, -- დენორმალიზაცია ანალიტიკისთვის
    session_id UUID NOT NULL REFERENCES recommendations_sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    category_name VARCHAR(255) NOT NULL,
    subcategory_name VARCHAR(255) NOT NULL,
    score FLOAT,
    preferred_brands TEXT[],
    price_range_min DECIMAL(10, 2),
    price_range_max DECIMAL(10, 2)
);

COMMENT ON TABLE recommendations_subcategory_details IS 'Stores detailed user preferences for each subcategory per session. Denormalized with batch_run_id.';


-- ცხრილი 4: საბოლოო რეკომენდაციების სია
-- ინახავს საბოლოო, დალაგებულ პროდუქტების სიას და მათ პოზიციას.
CREATE TABLE recommendations_final_list (
    id SERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL, -- დენორმალიზაცია ანალიტიკისთვის
    session_id UUID NOT NULL REFERENCES recommendations_sessions(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL,
    rank INTEGER NOT NULL
);

COMMENT ON TABLE recommendations_final_list IS 'Stores the final ranked list of recommended products for each session. Denormalized with batch_run_id.';


-- --- ინდექსების შექმნა ---
-- ინდექსები აჩქარებს მონაცემების წამოღებას (SELECT მოთხოვნებს).

-- მთავარი ცხრილების ინდექსები
CREATE INDEX IF NOT EXISTS idx_recs_batch_runs_timestamp ON recommendations_batch_runs(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_recs_sessions_batch_run_id ON recommendations_sessions(batch_run_id);
CREATE INDEX IF NOT EXISTS idx_recs_sessions_user_id ON recommendations_sessions(user_id);

-- დენორმალიზებული სვეტების ინდექსები
CREATE INDEX IF NOT EXISTS idx_recs_category_details_batch_run_id ON recommendations_category_details(batch_run_id);
CREATE INDEX IF NOT EXISTS idx_recs_subcategory_details_batch_run_id ON recommendations_subcategory_details(batch_run_id);
CREATE INDEX IF NOT EXISTS idx_recs_final_list_batch_run_id ON recommendations_final_list(batch_run_id);

-- ხშირად გამოყენებული სვეტების ინდექსები
CREATE INDEX IF NOT EXISTS idx_recs_category_details_session_id ON recommendations_category_details(session_id);
CREATE INDEX IF NOT EXISTS idx_recs_subcategory_details_session_id ON recommendations_subcategory_details(session_id);
CREATE INDEX IF NOT EXISTS idx_recs_final_list_session_id ON recommendations_final_list(session_id);

-- --- დასასრული ---

DROP TABLE IF EXISTS veli_client_detailed_profile;

CREATE TABLE veli_client_detailed_profile (
    user_id INT PRIMARY KEY,
    top_3_categories JSONB,
    top_3_subcategories JSONB,
    top_3_brands JSONB,
    price_ranges JSONB,
    updated_at TIMESTAMP
);

COMMENT ON TABLE veli_client_detailed_profile IS 'Contains generated user preference profiles, including top categories, subcategories, brands, and price ranges, all stored in JSONB format.';

-- ინდექსები user_id-ზე
CREATE INDEX IF NOT EXISTS idx_view_item_agg_user_id ON bigquery.view_item_agg (user_id);
CREATE INDEX IF NOT EXISTS idx_veli_orders_user_id ON veli_orders (user_id);
CREATE INDEX IF NOT EXISTS idx_cart_cart_user_id ON cart_cart (user_id);
CREATE INDEX IF NOT EXISTS idx_wishlist_wishlist_user_id ON wishlist_wishlist (user_id);

-- ინდექსი product_id-ზე JOIN-ისთვის
CREATE INDEX IF NOT EXISTS idx_veli_inventory_sales_product_id ON veli_inventory_and_sales_fixed (product_id);