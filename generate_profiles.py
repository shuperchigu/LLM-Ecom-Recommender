

import os
import json
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
import sys

# --- საწყისი კონფიგურაცია ---
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- გლობალური კონფიგურაცია ---
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_PORT = os.getenv("PG_PORT")


class FastUserProfiler:
    def __init__(self):
        """კლასის ინიციალიზაცია: იქმნება კავშირების ფული და იტვირთება კატეგორიების რუკა."""
        logging.info("Initializing FastUserProfiler...")
        try:
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1, maxconn=10, host=PG_HOST, database=PG_DATABASE,
                user=PG_USER, password=PG_PASSWORD, port=PG_PORT
            )
            logging.info("Database connection pool created successfully.")
        except (Exception, psycopg2.Error) as error:
            logging.critical(f"FATAL: Could not create database connection pool. Error: {error}")
            raise
        
        # ვქმნით ქვეკატეგორია -> მთავარი კატეგორიის რუკას, რომ მონაცემები სწორად დავაკავშიროთ
        self.sub_to_main_cat_map = self._get_category_map()
        if not self.sub_to_main_cat_map:
            logging.critical("FATAL: Subcategory to Main Category map could not be created.")
            raise ValueError("Category map is empty.")

    def _get_category_map(self) -> dict:
        """მოაქვს ბაზიდან უნიკალური კავშირები ქვეკატეგორიებსა და მთავარ კატეგორიებს შორის."""
        query = "SELECT DISTINCT subcategory, mother_cat_name FROM veli_inventory_and_sales_fixed WHERE subcategory IS NOT NULL AND mother_cat_name IS NOT NULL;"
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(query)
                # ვქმნით dict-ს: {'ქვეკატეგორია': 'მთავარი კატეგორია'}
                return {row[0]: row[1] for row in cur.fetchall()}
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Failed to load category map. Error: {error}", exc_info=True)
            return {}
        finally:
            if conn:
                self.db_pool.putconn(conn)

    def get_user_ids_from_query(self, query: str) -> list[int]:
        """ასრულებს SQL მოთხოვნას და აბრუნებს user_id-ების სიას."""
        conn = None
        logging.info("Executing query to fetch user IDs...")
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(query)
                user_ids = [row[0] for row in cur.fetchall()]
            logging.info(f"Successfully fetched {len(user_ids)} user IDs.")
            return user_ids
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Failed to fetch user IDs. Error: {error}", exc_info=True)
            return []
        finally:
            if conn: self.db_pool.putconn(conn)
    
    def get_user_activity(self, user_id: int) -> pd.DataFrame:
        """მოაქვს კონკრეტული მომხმარებლის ბოლო 500 აქტივობა."""
        query = f"""
        WITH user_actions AS (
            SELECT user_id::int, event_date::date, product_id, sum(view_item_count) as quantity, 'item_view' as attribute FROM bigquery.view_item_agg WHERE user_id::INT = {user_id} GROUP BY 1, 2, 3
            UNION ALL
            SELECT user_id::int, paid_date, product_id, quantity, 'item_purchase' as attribute FROM veli_orders WHERE user_id = {user_id} AND status NOT IN ('Start Payment', 'Failed', 'Cancelled', 'Blocked')
            UNION ALL
            SELECT user_id::int, (record_date + '04:00:00')::date, product_id, quantity, 'item_in_cart' as attribute FROM cart_cart WHERE user_id = {user_id}
            UNION ALL
            SELECT user_id::int, (record_date + '04:00:00')::date, product_id, 1 as quantity, 'item_in_wishlist' as attribute FROM wishlist_wishlist WHERE user_id = {user_id}
        )
        SELECT ua.user_id, ua.quantity, ua.event_date, ua.attribute, ua.product_id, vi.mother_cat_name, vi.subcategory, vi.brand, vi.price, vi.product_name
        FROM user_actions ua
        LEFT JOIN veli_inventory_and_sales_fixed vi ON ua.product_id = vi.product_id
        WHERE vi.product_name IS NOT NULL AND vi.mother_cat_name IS NOT NULL AND vi.subcategory IS NOT NULL
        ORDER BY ua.event_date DESC LIMIT 500;
        """
        conn = None
        try:
            conn = self.db_pool.getconn()
            return pd.read_sql_query(query, conn)
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Failed to get user activity for user_id {user_id}. Error: {error}", exc_info=True)
            return pd.DataFrame()
        finally:
            if conn: self.db_pool.putconn(conn)
    
    def calculate_profile(self, history: pd.DataFrame) -> dict:
        """აერთიანებს პროფილის ყველა კომპონენტის გამოთვლას."""
        if history.empty:
            return {}
        
        today = datetime.now().date()
        weights = {'item_purchase': 5.0, 'item_in_cart': 3.0, 'item_in_wishlist': 2.5, 'item_view': 1.0}
        
        history_with_decay = history.copy()
        history_with_decay['days_ago'] = (today - pd.to_datetime(history_with_decay['event_date']).dt.date).apply(lambda x: x.days)
        history_with_decay['decay'] = 0.99 ** history_with_decay['days_ago']
        history_with_decay['score'] = history_with_decay['attribute'].map(weights) * history_with_decay['decay']

        # 1. კატეგორიების ქულები
        category_scores = history_with_decay.groupby('mother_cat_name')['score'].sum().to_dict()

        # 2. ქვეკატეგორიების ქულები
        subcategory_scores = history_with_decay.groupby('subcategory')['score'].sum().to_dict()

        # 3. ფასების დიაპაზონები
        price_ranges = {}
        price_actions = history[history['attribute'].isin(['item_purchase', 'item_in_cart'])].dropna(subset=['price', 'subcategory'])
        if not price_actions.empty:
            avg_prices = price_actions.groupby('subcategory')['price'].mean()
            for subcat, avg_price in avg_prices.items():
                price_ranges[subcat] = {"min": round(avg_price * 0.5, 2), "max": round(avg_price * 2.0, 2)}
        
        # 4. ბრენდების პრეფერენციები
        brand_prefs = {}
        brand_actions = history[history['attribute'].isin(['item_purchase', 'item_in_cart', 'item_in_wishlist'])].dropna(subset=['subcategory', 'brand'])
        if not brand_actions.empty:
            brand_counts = brand_actions.groupby(['subcategory', 'brand']).size().reset_index(name='counts')
            for subcat, group in brand_counts.groupby('subcategory'):
                brand_prefs[subcat] = group.sort_values('counts', ascending=False)['brand'].head(3).tolist()

        return {
            "category_scores": category_scores,
            "subcategory_scores": subcategory_scores,
            "price_ranges": price_ranges,
            "brand_preferences": brand_prefs
        }

    def save_profile_to_db(self, user_id: int, profile_data: dict, batch_run_id: int):
        """ინახავს პროფილს ორ რელაციურ ცხრილში."""
        if not profile_data:
            return

        category_scores = profile_data.get("category_scores", {})
        subcategory_scores = profile_data.get("subcategory_scores", {})
        price_ranges = profile_data.get("price_ranges", {})
        brand_preferences = profile_data.get("brand_preferences", {})

        # ვამზადებთ მონაცემებს batch insert-ისთვის
        cat_details_data = [
            (batch_run_id, user_id, name, score)
            for name, score in category_scores.items()
        ]
        
        subcat_details_data = []
        for sub, score in subcategory_scores.items():
            main_cat = self.sub_to_main_cat_map.get(sub, 'Unknown') 
            pr = price_ranges.get(sub)
            subcat_details_data.append((
                batch_run_id, user_id, main_cat, sub, score, 
                brand_preferences.get(sub, []), 
                pr.get("min") if pr else None, 
                pr.get("max") if pr else None
            ))

        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                if cat_details_data:
                    psycopg2.extras.execute_batch(cur, "INSERT INTO profiling_category_details (batch_run_id, user_id, category_name, score) VALUES (%s, %s, %s, %s)", cat_details_data)
                if subcat_details_data:
                    psycopg2.extras.execute_batch(cur, "INSERT INTO profiling_subcategory_details (batch_run_id, user_id, category_name, subcategory_name, score, preferred_brands, price_range_min, price_range_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", subcat_details_data)
            conn.commit()
            logging.debug(f"Successfully saved profile for user {user_id} (Batch: {batch_run_id}).")
        except (Exception, psycopg2.Error) as error:
            logging.error(f"DB Save Error for user {user_id} (Batch: {batch_run_id}): {error}", exc_info=True)
            if conn: conn.rollback()
        finally:
            if conn: self.db_pool.putconn(conn)

    def generate_and_save_profile(self, user_id: int, batch_run_id: int):
        """აერთიანებს პროფილის გენერირების და შენახვის სრულ ციკლს."""
        try:
            user_history = self.get_user_activity(user_id)
            if user_history.empty:
                return f"Skipped: User {user_id} has no activity."
            
            profile_data = self.calculate_profile(user_history)
            if not profile_data:
                return f"Skipped: Could not generate profile for user {user_id}."
            
            self.save_profile_to_db(user_id, profile_data, batch_run_id)
            return f"Success: Profile generated for user {user_id}."
        except Exception as e:
            logging.error(f"CRITICAL ERROR for user {user_id}: {e}", exc_info=True)
            return f"Failed: An error occurred for user {user_id}."


# --- სკრიპტის გაშვების ბლოკი ---
if __name__ == "__main__":
    user_fetch_query = """
    select user_id
      from veli_orders
      where status not in ('Start Payment', 'Failed', 'Cancelled', 'Blocked')
      and user_id <> 2
      group by user_id
      order by count(user_id) desc 
      LIMIT 200000;
    """
    ALGORITHM_VERSION = "profiling-v1.0"
    MAX_WORKERS = 8 # CPU-bound ამოცანაა, ამიტომ 8-16 კარგი არჩევანია

    start_time = time.time()
    profiler = None
    try:
        profiler = FastUserProfiler()
        user_ids_to_process = profiler.get_user_ids_from_query(user_fetch_query)

        if not user_ids_to_process:
            logging.warning("No user IDs to process. Exiting.")
            sys.exit(0)
        
        # ვქმნით ახალ batch run-ს
        conn = profiler.db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO profiling_batch_runs (algorithm_version) VALUES (%s) RETURNING id;", (ALGORITHM_VERSION,))
            batch_run_id = cur.fetchone()[0]
            conn.commit()
        profiler.db_pool.putconn(conn)
        logging.info(f"Created new batch run with ID: {batch_run_id}")

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(profiler.generate_and_save_profile, uid, batch_run_id): uid for uid in user_ids_to_process}
            progress_bar = tqdm(as_completed(futures), total=len(user_ids_to_process), desc=f"Generating Profiles (Batch ID: {batch_run_id})")
            for future in progress_bar:
                try:
                    result_message = future.result()
                    results.append(result_message)
                except Exception as exc:
                    user_id = futures[future]
                    error_message = f"User {user_id} generated an unhandled exception: {exc}"
                    logging.error(error_message, exc_info=True)
                    results.append(error_message)

        total_time = time.time() - start_time
        
        # ვანახლებთ batch run-ის სტატისტიკას
        conn = profiler.db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("UPDATE profiling_batch_runs SET end_time = %s, processed_users_count = %s, duration_seconds = %s WHERE id = %s;", 
                        (datetime.now(timezone.utc), len(user_ids_to_process), total_time, batch_run_id))
            conn.commit()
        profiler.db_pool.putconn(conn)
        logging.info(f"Finalized batch run {batch_run_id}.")

        success_count = sum(1 for r in results if r and r.startswith("Success"))
        skipped_count = sum(1 for r in results if r and r.startswith("Skipped"))
        failed_count = len(results) - success_count - skipped_count

        print("\n" + "="*60 + "\n        PROFILING SUMMARY\n" + "="*60)
        print(f"Batch ID: {batch_run_id}")
        print(f"Total time: {total_time:.2f} seconds.")
        print(f"Processed {len(user_ids_to_process)} users.")
        if user_ids_to_process: print(f"Average time per user: {total_time / len(user_ids_to_process):.2f} seconds.")
        print("-" * 20 + f"\nSuccessful profiles: {success_count}\nSkipped: {skipped_count}\nFailed: {failed_count}\n" + "="*60)

    except (Exception, psycopg2.Error) as e:
        logging.critical(f"A critical error occurred in the main execution block: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if profiler and profiler.db_pool:
            profiler.db_pool.closeall()

            logging.info("Database connection pool has been closed.")
