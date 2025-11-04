import os
import logging
import time
import pandas as pd
import psycopg2
import warnings 
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm # <<< 2. დამატებულია tqdm პროგრეს-ბარისთვის

# --- კონფიგურაცია ---
load_dotenv()
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# მონაცემთა ბაზის კავშირის პარამეტრები .env ფაილიდან
DB_NAME = os.getenv("PG_DATABASE")
DB_USER = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT")

# ალგორითმის პარამეტრები
ALGORITHM_VERSION = "BIA-v1.1-tqdm"
MAX_WORKERS = 10 
USER_ACTIVITY_DAYS = 365 
MAX_RECOMMENDATIONS_PER_USER = 10 

# --- დამხმარე ფუნქციები ---

def get_db_connection():
    """ქმნის და აბრუნებს მონაცემთა ბაზასთან კავშირს."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"მონაცემთა ბაზასთან დაკავშირების შეცდომა: {e}")
        raise

def load_rules(conn):
    """
    ტვირთავს წესებს veli_recommendation_rules ცხრილიდან Pandas DataFrame-ში.
    """
    logging.info("წესების ჩატვირთვა...")
    query = "SELECT rule_level, name, purchase_type, default_replenishment_days  as default_replenishment_days FROM veli_recommendation_rules;"
    try:
        rules_df = pd.read_sql(query, conn)
        logging.info(f"{len(rules_df)} წესი წარმატებით ჩაიტვირთა.")
        return rules_df
    except Exception as e:
        logging.error(f"წესების ჩატვირთვის შეცდომა: {e}")
        raise

def get_target_users(conn, days_active):
    """
    აბრუნებს იმ მომხმარებლების სიას, რომლებსაც ბოლო N დღის განმავლობაში ჰქონდათ შეკვეთა.
    """
    logging.info(f"დასამუშავებელი მომხმარებლების სიის წამოღება (ბოლო {days_active} დღის აქტივობა)...")
    query = f"""
        SELECT DISTINCT user_id
        FROM veli_orders 
        WHERE paid_date >= NOW() - INTERVAL '{days_active} days'
          AND status NOT IN ('Start Payment', 'Failed', 'Cancelled', 'Blocked');
    """
    try:
        users_df = pd.read_sql(query, conn)
        user_ids = users_df['user_id'].dropna().astype(int).tolist()
        logging.info(f"ნაპოვნია {len(user_ids)} აქტიური მომხმარებელი.")
        return user_ids
    except Exception as e:
        logging.error(f"მომხმარებლების სიის წამოღების შეცდომა: {e}")
        return []

def get_user_purchase_history(conn, user_id):
    """
    აბრუნებს მომხმარებლის მიერ შეძენილი პროდუქტების ისტორიას.
    """
    query = """
        SELECT
            oi.product_id,
            oi.product_name AS product_name,
            oi.mother_cat_name AS category_name,
            oi.subcategory AS subcategory_name,
            oi.paid_date AS purchase_date
        FROM veli_orders oi
        WHERE oi.user_id = %s AND oi.status NOT IN ('Start Payment', 'Failed', 'Cancelled', 'Blocked');
    """
    try:
        history_df = pd.read_sql(query, conn, params=(user_id,))
        return history_df
    except Exception as e:
        logging.error(f"მომხმარებლის {user_id} ისტორიის წამოღების შეცდომა: {e}")
        return pd.DataFrame()


def process_user(user_id, rules_df, batch_run_id):
    """
    ამუშავებს ერთ მომხმარებელს: იღებს ისტორიას, იყენებს წესებს, არანჟირებს და ინახავს შედეგებს.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO bia_sessions (batch_run_id, user_id) VALUES (%s, %s) RETURNING id;",
                (batch_run_id, user_id)
            )
            session_id = cur.fetchone()[0]

            history_df = get_user_purchase_history(conn, user_id)
            if history_df.empty:
                return (user_id, 'SUCCESS_NO_HISTORY', 0)

            history_df['purchase_date'] = pd.to_datetime(history_df['purchase_date'])
            last_purchases_df = history_df.sort_values('purchase_date').drop_duplicates('product_id', keep='last')

            merged_df = pd.merge(
                last_purchases_df,
                rules_df[rules_df['rule_level'] == 'subcategory'],
                how='left',
                left_on='subcategory_name',
                right_on='name'
            )
            merged_df = pd.merge(
                merged_df,
                rules_df[rules_df['rule_level'] == 'category'],
                how='left',
                left_on='category_name',
                right_on='name',
                suffixes=('_sub', '_cat')
            )

            merged_df['final_purchase_type'] = merged_df['purchase_type_sub'].fillna(merged_df['purchase_type_cat'])
            merged_df['final_replenishment_days'] = merged_df['default_replenishment_days_sub'].fillna(merged_df['default_replenishment_days_cat'])

            candidates_df = merged_df[merged_df['final_purchase_type'] == 'repeatable'].copy()
            candidates_df.dropna(subset=['final_replenishment_days'], inplace=True)
            if candidates_df.empty:
                return (user_id, 'SUCCESS_NO_REPEATABLE_ITEMS', 0)

            candidates_df['final_replenishment_days'] = candidates_df['final_replenishment_days'].astype(int)
            candidates_df['predicted_replenish_date'] = candidates_df.apply(
                lambda row: row['purchase_date'] + timedelta(days=row['final_replenishment_days']), axis=1
            )
            
            today = datetime.now(candidates_df['predicted_replenish_date'].iloc[0].tz)

            due_recommendations_df = candidates_df[candidates_df['predicted_replenish_date'] <= today].copy()
            if due_recommendations_df.empty:
                return (user_id, 'SUCCESS_NO_DUE_ITEMS', 0)
            
            due_recommendations_df['days_overdue'] = (today - due_recommendations_df['predicted_replenish_date']).dt.days
            
            final_recs_df = due_recommendations_df.sort_values('days_overdue', ascending=False).head(MAX_RECOMMENDATIONS_PER_USER)
            final_recs_df['rank'] = range(1, len(final_recs_df) + 1)
            final_recs_df['user_id'] = user_id

            records_to_insert = final_recs_df[[
                'user_id', 'product_id', 'rank', 'purchase_date', 'predicted_replenish_date', 'days_overdue'
            ]].to_dict('records')
            insert_query = """
                INSERT INTO bia_recommendations (
                    batch_run_id, session_id, user_id, product_id, rank, last_purchase_date,
                    predicted_replenish_date, days_overdue
                ) VALUES %s;
            """
            execute_values(
                cur,
                insert_query,
                [(batch_run_id, session_id, r['user_id'], r['product_id'], r['rank'], r['purchase_date'].date(),
                  r['predicted_replenish_date'].date(), r['days_overdue']) for r in records_to_insert]
            )
            
            cur.execute(
                "UPDATE bia_sessions SET recommendations_count = %s WHERE id = %s;",
                (len(final_recs_df), session_id)
            )

            conn.commit()
            return (user_id, 'SUCCESS', len(final_recs_df))

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"შეცდომა მომხმარებლის {user_id} დამუშავებისას: {e}")
        return (user_id, 'FAILED', 0)
    finally:
        if conn:
            conn.close()

# --- მთავარი ფუნქცია ---

def main():
    start_time = time.time()
    conn = get_db_connection()
    batch_run_id = None
    processed_users_count = 0
    total_recs_generated = 0

    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO bia_batch_runs (algorithm_version) VALUES (%s) RETURNING id;",
                (ALGORITHM_VERSION,)
            )
            batch_run_id = cur.fetchone()[0]
            conn.commit()
            logging.info(f"დაიწყო ახალი გაშვება: batch_run_id = {batch_run_id}")

        rules_df = load_rules(conn)
        user_ids = get_target_users(conn, USER_ACTIVITY_DAYS)

        if not user_ids:
            logging.warning("აქტიური მომხმარებლები ვერ მოიძებნა. პროცესი სრულდება.")
            if batch_run_id:
                duration = time.time() - start_time
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE bia_batch_runs SET status = 'Completed', duration_seconds = %s, processed_users_count = 0, total_recommendations_generated = 0 WHERE id = %s;",
                        (duration, batch_run_id)
                    )
                    conn.commit()
            return

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_user, user_id, rules_df, batch_run_id) for user_id in user_ids}
            progress_bar = tqdm(as_completed(futures), total=len(user_ids), desc="მომხმარებლების დამუშავება")
            for future in progress_bar:
                try:
                    user_id, status, rec_count = future.result()
                    if status.startswith('SUCCESS'):
                        processed_users_count += 1
                        total_recs_generated += rec_count
                    else:
                        logging.warning(f"მომხმარებელი {user_id} ვერ დამუშავდა.")
                except Exception as exc:
                    logging.error(f"Future-ის დამუშავებისას მოხდა შეცდომა: {exc}")
        
        logging.info("ყველა მომხმარებლის დამუშავება დასრულდა.")

    except Exception as e:
        logging.critical(f"კრიტიკული შეცდომა სკრიპტის მუშაობისას: {e}", exc_info=True)
        if batch_run_id and conn:
            with conn.cursor() as cur:
                duration = time.time() - start_time
                cur.execute(
                    "UPDATE bia_batch_runs SET status = 'Failed', duration_seconds = %s, error_message = %s WHERE id = %s;",
                    (duration, str(e), batch_run_id)
                )
                conn.commit()
        return
    
    finally:
        if batch_run_id and conn:
            duration = time.time() - start_time
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM bia_batch_runs WHERE id = %s", (batch_run_id,))
                current_status = cur.fetchone()[0]
                if current_status != 'Failed':
                    cur.execute(
                        """
                        UPDATE bia_batch_runs
                        SET status = 'Completed',
                            duration_seconds = %s,
                            processed_users_count = %s,
                            total_recommendations_generated = %s
                        WHERE id = %s;
                        """,
                        (duration, processed_users_count, total_recs_generated, batch_run_id)
                    )
                    conn.commit()
            logging.info(f"გაშვება {batch_run_id} დასრულდა. დრო: {duration:.2f} წამი. დამუშავდა {processed_users_count} მომხმარებელი, დაგენერირდა {total_recs_generated} რეკომენდაცია.")
        if conn:
            conn.close()


if __name__ == "__main__":

    main()
