# generate_embeddings.py

import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer
import google.generativeai as genai
import time
import warnings
from sqlalchemy.exc import SAWarning

# გაფრთხილებების იგნორირება
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SAWarning)


# --- კონფიგურაცია ---
load_dotenv()
DATABASE_URL = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True) # pool_pre_ping ეხმარება კავშირის შენარჩუნებაში
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Embedding მოდელის პარამეტრები
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 768

# --- SQLAlchemy მოდელი ---
Base = declarative_base()
class ProductEmbedding(Base):
    __tablename__ = 'product_embeddings'
    product_id = Column(Integer, primary_key=True, autoincrement=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION))

# --- დამხმარე ფუნქციები ---

def get_product_catalog_for_embedding() -> pd.DataFrame:
    """იღებს პროდუქტების სრულ კატალოგს დროში შეზღუდული მეტამონაცემებით."""
    print("Fetching full product catalog with time-sensitive metadata (last 6 months)...")
    query = """
    WITH product_data AS (
        SELECT id AS product_id, headline_ka, rating::float, visible
        FROM product_product
    ),
    brand_data AS (
        SELECT DISTINCT ON (product_id) product_id, t2.headline_ka AS brand_name
        FROM product_product_link t1
        LEFT JOIN product_link t2 ON t1.link_id = t2.id
        WHERE t2.type IN ('1', '3') AND t2.headline_ka IS NOT NULL
    ),
    inventory_data AS (
        SELECT product_id, round(price::numeric, 2) AS price, round(start_price::numeric, 2) AS originalprice,
               in_stock, mother_cat_name, subcategory
        FROM veli_inventory_and_sales_fixed
    ),
    session_data AS (
        SELECT product_id, SUM(view_item_count) AS views
        FROM bigquery.view_item_agg 
        WHERE event_date::date >= (CURRENT_DATE - INTERVAL '6 months')
        GROUP BY product_id
    ),
    products_sold AS (
        SELECT product_id, sum(quantity) as sold_items
        FROM veli_orders 
        WHERE status not in ('Start Payment', 'Failed', 'Cancelled', 'Blocked') 
        AND paid_date >= (CURRENT_DATE - INTERVAL '6 months')
        AND is_juridical IS FALSE
        GROUP BY product_id
    )
    SELECT
        pd.product_id, pd.headline_ka, pd.rating,
        b.brand_name,
        inv.price, inv.originalprice, inv.mother_cat_name, inv.subcategory,
        COALESCE(sd.views, 0) as views,
        COALESCE(ps.sold_items, 0) as sold_items
    FROM product_data pd
    LEFT JOIN brand_data b ON pd.product_id = b.product_id
    LEFT JOIN inventory_data inv ON pd.product_id = inv.product_id
    LEFT JOIN session_data sd ON pd.product_id = sd.product_id
    LEFT JOIN products_sold ps ON pd.product_id = ps.product_id
    WHERE pd.visible IS TRUE 
    AND inv.price > 0 
    AND inv.in_stock > 0
    AND inv.mother_cat_name IS NOT NULL;
    """
    # ვიყენებთ psycopg2-ს, რადგან ის უფრო მარტივია ერთი მოთხოვნისთვის
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        database=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        port=os.getenv("PG_PORT")
    )
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def generate_rich_text_for_product(row):
    """ქმნის სტრუქტურირებულ და მდიდარ ტექსტს embedding-ისთვის."""
    price = row.get('price')
    price_category = "საშუალო"
    if price and price < 50: price_category = "ბიუჯეტური"
    elif price and price > 400: price_category = "პრემიუმ"
    
    discount_tag = ""
    original_price = row.get('originalprice')
    if original_price and price and original_price > price:
        discount_percent = round((1 - price / original_price) * 100)
        if discount_percent >= 50: discount_tag = "დიდი ფასდაკლება"
        elif discount_percent >= 20: discount_tag = "ფასდაკლება"

    rating = row.get('rating')
    rating_category = "საშუალო შეფასება"
    if rating and rating >= 4.7: rating_category = "უმაღლესი შეფასება"
    elif rating and rating >= 4.2: rating_category = "მაღალი შეფასება"
    
    sold_items = row.get('sold_items', 0)
    popularity_category = "სტანდარტული პოპულარობა"
    if sold_items > 200: popularity_category = "ბესტსელერი"
    elif sold_items > 50: popularity_category = "პოპულარული"

    parts = [
        f"პროდუქტი: {row.get('headline_ka', '')}",
        f"ბრენდი: {row.get('brand_name', 'N/A')}",
        f"კატეგორია: {row.get('mother_cat_name', '')}",
        f"ქვეკატეგორია: {row.get('subcategory', '')}",
        f"ფასი: {price_category}",
        f"რეიტინგი: {rating_category}",
        f"პოპულარობა: {popularity_category}"
    ]
    if discount_tag:
        parts.append(f"სტატუსი: {discount_tag}")
    return "; ".join(parts) + "."

def embed_texts(texts, batch_size=100):
    """ტექსტების ჯგუფურად ვექტორიზაცია, API ლიმიტების გათვალისწინებით."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  - Generating embedding for batch {i//batch_size + 1}...")
        try:
            result = genai.embed_content(model=EMBEDDING_MODEL, content=batch, task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(result['embedding'])
        except Exception as e:
            print(f"    - Error in batch, retrying in 5 seconds... Error: {e}")
            time.sleep(5)
            try:
                result = genai.embed_content(model=EMBEDDING_MODEL, content=batch, task_type="RETRIEVAL_DOCUMENT")
                all_embeddings.extend(result['embedding'])
            except Exception as e2:
                print(f"    - Retry failed for batch. Skipping. Error: {e2}")
                all_embeddings.extend([None] * len(batch))
        time.sleep(1)
    return all_embeddings

def main():
    print("--- Starting Rich Embedding Generation Process ---")
    
    # 1. ცხრილის შექმნა
    Base.metadata.create_all(bind=engine)
    
    # --- ეტაპი 1: მონაცემების მომზადება ---
    products_df = get_product_catalog_for_embedding()
    if products_df.empty:
        print("No products found to process.")
        return
    print(f"Fetched {len(products_df)} products from the catalog.")

    # +++ სწორი ადგილი დუბლიკატების ამოსაშლელად +++
    # ვშლით დუბლიკატებს, სანამ ტექსტის გენერაციას დავიწყებთ
    initial_count = len(products_df)
    products_df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)
    final_count = len(products_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate product_ids. Processing {final_count} unique products.")
    # +++ შესწორება მთავრდება აქ +++

    print("Generating structured text descriptions for each product...")
    products_df['text_to_embed'] = products_df.apply(generate_rich_text_for_product, axis=1)
    
    print("\n--- Example of generated rich texts: ---")
    print(products_df[['product_id', 'text_to_embed']].head(3).to_string())
    print("----------------------------------------\n")

    # --- ეტაპი 2: Embedding-ების გენერაცია (ყველაზე ხანგრძლივი ნაწილი) ---
    print("Generating new embeddings... This will take a while.")
    texts_to_embed = products_df['text_to_embed'].tolist()
    embeddings = embed_texts(texts_to_embed)
    products_df['embedding'] = embeddings
    
    valid_products_df = products_df.dropna(subset=['embedding']).copy()
    print(f"Successfully generated {len(valid_products_df)} embeddings.")
    if valid_products_df.empty:
        print("No valid embeddings were generated. Exiting.")
        return

    # --- ეტაპი 3: ბაზაში ჩაწერა (ახალი, ცოცხალი კავშირით) ---
    print("Upserting embeddings into the database in chunks...")
    session = SessionLocal()
    update_count = 0
    insert_count = 0
    
    try:
        chunk_size = 500
        total_chunks = len(valid_products_df) // chunk_size + 1
        for i in range(0, len(valid_products_df), chunk_size):
            chunk_df = valid_products_df.iloc[i:i+chunk_size]
            print(f"  - Processing DB chunk {i//chunk_size + 1} of {total_chunks}...")
            
            pids_in_chunk = [int(pid) for pid in chunk_df['product_id']]
            existing_pids = set(
                r[0] for r in session.query(ProductEmbedding.product_id).filter(ProductEmbedding.product_id.in_(pids_in_chunk))
            )
            
            to_update = []
            to_insert = []
            
            for _, row in chunk_df.iterrows():
                pid = int(row['product_id'])
                if pid in existing_pids:
                    to_update.append({'product_id': pid, 'embedding': np.array(row['embedding'])})
                else:
                    to_insert.append({'product_id': pid, 'embedding': np.array(row['embedding'])})
            
            if to_insert:
                session.bulk_insert_mappings(ProductEmbedding, to_insert)
                insert_count += len(to_insert)
            if to_update:
                session.bulk_update_mappings(ProductEmbedding, to_update)
                update_count += len(to_update)

            session.commit()

    except Exception as e:
        print(f"An error occurred during database upsert: {e}")
        session.rollback()
    finally:
        session.close()
        
    print(f"\nProcess finished. Inserted: {insert_count}, Updated: {update_count}.")

    # --- ეტაპი 4: ინდექსის განახლება ---
    try:
        print("Re-creating HNSW index for optimal performance...")
        with engine.connect() as conn:
            # ვიყენებთ execute-ს ტრანზაქციის გარეშე, თუ საჭიროა
            conn.execution_options(isolation_level="AUTOCOMMIT").execute(text("DROP INDEX IF EXISTS ix_product_embeddings_embedding;"))
            conn.execution_options(isolation_level="AUTOCOMMIT").execute(text(f"CREATE INDEX ix_product_embeddings_embedding ON product_embeddings USING hnsw (embedding vector_l2_ops);"))
        print("Index re-created successfully.")
    except Exception as e:
        print(f"Could not re-create index (it might already exist or another issue occurred): {e}")

if __name__ == "__main__":
    main()