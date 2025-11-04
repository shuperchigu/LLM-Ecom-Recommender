import os
import json
import time
import logging
import warnings
import random
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from sqlalchemy.exc import SAWarning
import sys

# Gemini API-სთვის საჭირო import-ი
import google.generativeai as genai

# --- საწყისი კონფიგურაცია ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SAWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- გლობალური კონფიგურაცია ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_PORT = os.getenv("PG_PORT")

# Gemini API-ს კონფიგურაცია API გასაღებით
genai.configure(api_key=GOOGLE_API_KEY)


# --- დამხმარე დეკორატორი ხელახალი ცდებისთვის ---
def retry_with_exponential_backoff(func):
    """
    დეკორატორი, რომელიც ფუნქციის შეცდომის შემთხვევაში ცდილობს მის ხელახალ გაშვებას
    ექსპონენციალური დაყოვნებით. იდეალურია არასტაბილურ ქსელურ API-სთან სამუშაოდ.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5  # მაქსიმუმ 5 ცდა
        base_delay = 2   # პირველი დაყოვნება 2 წამი
        
        for attempt in range(max_retries):
            try:
                # ვცდილობთ ფუნქციის გაშვებას
                return func(*args, **kwargs)
            except Exception as e:
                # თუ ეს ბოლო ცდა იყო, ვაბრუნებთ შეცდომას და ვასრულებთ
                if attempt == max_retries - 1:
                    logging.error(f"Function {func.__name__} failed after {max_retries} retries. Final error: {e}")
                    raise  # საბოლოოდ ვაბრუნებთ შეცდომას, რომ ზედა დონემ დაიჭიროს

                # ვიანგარიშებთ შემდეგ დაყოვნებას (2, 4, 8, 16 წამი) და ვამატებთ მცირე შემთხვევითობას (Jitter)
                delay = (base_delay ** (attempt + 1)) + (random.uniform(0, 1))
                logging.warning(f"Function {func.__name__} failed with error: {e}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)  # ველოდებით
    return wrapper


class RecommenderSystem:
    def __init__(self):
        """კლასის ინიციალიზაცია: იქმნება მოდელების ინსტანციები, კავშირების ფული და იტვირთება კატალოგი."""
        logging.info("Initializing RecommenderSystem with Gemini API...")

        try:
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=PG_HOST,
                database=PG_DATABASE,
                user=PG_USER,
                password=PG_PASSWORD,
                port=PG_PORT
            )
            logging.info("Database connection pool created successfully.")
        except (Exception, psycopg2.Error) as error:
            logging.critical(f"FATAL: Could not create database connection pool. Error: {error}")
            raise

        self.generative_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.product_catalog = self._get_product_catalog()
        if self.product_catalog.empty:
            logging.critical("FATAL: Product catalog is empty. Aborting initialization.")
            raise ValueError("Product catalog could not be loaded.")

        self.sub_to_main_cat_map = self.product_catalog.dropna(subset=['mother_cat_name', 'subcategory']).set_index('subcategory')['mother_cat_name'].to_dict()
        logging.info("RecommenderSystem Initialized.")

    def get_user_ids_from_query(self, query: str) -> list[int]:
        """ასრულებს მოცემულ SQL მოთხოვნას და აბრუნებს user_id-ების სიას."""
        user_ids = []
        conn = None
        logging.info(f"Executing query to fetch user IDs...")
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(query)
                user_ids = [row[0] for row in cur.fetchall()]
            logging.info(f"Successfully fetched {len(user_ids)} user IDs from the database.")
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Failed to fetch user IDs from DB. Error: {error}", exc_info=True)
            return []
        finally:
            if conn:
                self.db_pool.putconn(conn)
        return user_ids

    def _embed_batch(self, contents: list[str]) -> dict[str, list[float]]:
        """
        დამხმარე ფუნქცია ტექსტების პარალელურად და გამძლეობით დასაამბედებლად.
        იყენებს retry ლოგიკას და შემცირებულ პარალელიზმს.
        """
        embeddings = {}

        @retry_with_exponential_backoff
        def embed_single_with_retry(content):
            """ასრულებს ერთ მოთხოვნას Gemini-სთან ხელახალი ცდების ლოგიკით."""
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=content,
                task_type="RETRIEVAL_QUERY"
            )
            return content, result['embedding']

        def safe_embed_wrapper(content):
            try:
                return embed_single_with_retry(content)
            except Exception:
                return content, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(safe_embed_wrapper, text) for text in contents]
            for future in as_completed(futures):
                content, embedding = future.result()
                if embedding:
                    embeddings[content] = embedding
        return embeddings

    def _get_product_catalog(self) -> pd.DataFrame:
        """მოაქვს პროდუქტების სრული და აქტიური კატალოგი ბაზიდან."""
        query = "SELECT p.product_id, p.product_name, p.price, p.mother_cat_name, p.subcategory, p.brand FROM veli_inventory_and_sales_fixed p WHERE p.price > 0 AND p.in_stock > 0 AND p.mother_cat_name IS NOT NULL;"
        conn = None
        try:
            conn = self.db_pool.getconn()
            df = pd.read_sql_query(query, conn)
            return df
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Failed to load product catalog. Error: {error}", exc_info=True)
            return pd.DataFrame()
        finally:
            if conn:
                self.db_pool.putconn(conn)

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
        SELECT ua.user_id, ua.quantity, ua.event_date, ua.attribute, ua.product_id, vi.mother_cat_name, vi.subcategory, vi.brand, vi.supplier_name, vi.start_price, vi.price, vi.product_name
        FROM user_actions ua
        LEFT JOIN veli_inventory_and_sales_fixed vi ON ua.product_id = vi.product_id
        WHERE vi.product_name IS NOT NULL ORDER BY ua.event_date DESC LIMIT 500;
        """
        conn = None
        try:
            conn = self.db_pool.getconn()
            df = pd.read_sql_query(query, conn)
            return df
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Failed to get user activity for user_id {user_id}. Error: {error}", exc_info=True)
            return pd.DataFrame()
        finally:
            if conn:
                self.db_pool.putconn(conn)

    def calculate_category_scores(self, history: pd.DataFrame) -> dict:
        """ითვლის მომხმარებლის ინტერესის ქულებს მთავარი კატეგორიების მიხედვით."""
        scores = {}
        today = datetime.now().date()
        weights = {'item_purchase': 5.0, 'item_in_cart': 3.0, 'item_in_wishlist': 2.5, 'item_view': 1.0}
        history = history.dropna(subset=['mother_cat_name'])
        for _, row in history.iterrows():
            days_ago = (today - pd.to_datetime(row['event_date']).date()).days
            decay = 0.99 ** max(0, days_ago)
            scores[row['mother_cat_name']] = scores.get(row['mother_cat_name'], 0) + (weights.get(row['attribute'], 0.5) * decay)
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def calculate_subcategory_scores(self, history: pd.DataFrame) -> dict:
        """ითვლის მომხმარებლის ინტერესის ქულებს ქვეკატეგორიების მიხედვით."""
        scores = {}
        today = datetime.now().date()
        weights = {'item_purchase': 5.0, 'item_in_cart': 3.0, 'item_in_wishlist': 2.5, 'item_view': 1.0}
        history = history.dropna(subset=['subcategory'])
        for _, row in history.iterrows():
            days_ago = (today - pd.to_datetime(row['event_date']).date()).days
            decay = 0.99 ** max(0, days_ago)
            scores[row['subcategory']] = scores.get(row['subcategory'], 0) + (weights.get(row['attribute'], 0.5) * decay)
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    def calculate_price_ranges(self, history: pd.DataFrame) -> dict:
        """ითვლის მომხმარებლის სასურველ ფასების დიაპაზონს."""
        prefs = {}
        actions = history[history['attribute'].isin(['item_purchase', 'item_in_cart'])].dropna(subset=['price', 'subcategory'])
        if actions.empty: return {}
        avg_prices = actions.groupby('subcategory')['price'].mean()
        for subcat, avg_price in avg_prices.items():
            prefs[subcat] = {"min": round(avg_price * 0.5, 2), "max": round(avg_price * 2.0, 2)}
        return prefs

    def calculate_brand_preferences(self, history: pd.DataFrame) -> dict:
        """ითვლის მომხმარებლის ბრენდის ლოიალობას."""
        prefs = {}
        actions = history[history['attribute'].isin(['item_purchase', 'item_in_cart', 'item_in_wishlist'])].dropna(subset=['subcategory', 'brand'])
        if actions.empty: return {}
        brand_counts = actions.groupby(['subcategory', 'brand']).size().reset_index(name='counts')
        for subcat, group in brand_counts.groupby('subcategory'):
            prefs[subcat] = group.sort_values('counts', ascending=False)['brand'].head(3).tolist()
        return prefs

    def get_candidate_products(self, user_id: int, cat_scores: dict, sub_scores: dict, price_ranges: dict, brand_prefs: dict, total_candidates: int = 400, oversampling_factor: float = 3.0) -> pd.DataFrame:
        """აგენერირებს კანდიდატ პროდუქტებს (პარალელური embedding-ით)."""
        all_candidate_ids = set()
        main_cat_total_score = sum(cat_scores.values())
        if main_cat_total_score == 0: return pd.DataFrame()

        texts_to_embed, subcat_targets = [], {}
        for main_category, main_score in list(cat_scores.items())[:10]:
            num_candidates_for_main_cat = int(round((main_score / main_cat_total_score) * total_candidates))
            if num_candidates_for_main_cat == 0: continue
            relevant_subcats = {s: sc for s, sc in sub_scores.items() if self.sub_to_main_cat_map.get(s) == main_category}
            sub_cat_total_score = sum(relevant_subcats.values())
            if sub_cat_total_score == 0: continue
            for sub_category, sub_score in sorted(relevant_subcats.items(), key=lambda item: item[1], reverse=True):
                target = int(round((sub_score / sub_cat_total_score) * num_candidates_for_main_cat))
                if target > 0:
                    texts_to_embed.append(f"პოპულარული პროდუქტები {sub_category} კატეგორიიდან")
                    subcat_targets[sub_category] = target

        if not texts_to_embed:
            logging.info(f"User {user_id}: No subcategories to generate candidates from.")
            return pd.DataFrame()

        logging.info(f"User {user_id}: Starting batch embedding for {len(texts_to_embed)} subcategories.")
        start_embed_time = time.time()
        embedded_texts = self._embed_batch(texts_to_embed)
        logging.info(f"User {user_id}: Finished batch embedding in {time.time() - start_embed_time:.2f} seconds.")

        if not embedded_texts:
            logging.warning(f"User {user_id}: Failed to generate any embeddings. Skipping candidate generation.")
            return pd.DataFrame()

        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                for sub_category, target in subcat_targets.items():
                    content_key = f"პოპულარული პროდუქტები {sub_category} კატეგორიიდან"
                    embedding = embedded_texts.get(content_key)
                    if not embedding: continue

                    params = {'sub': sub_category, 'uid': user_id, 'emb': str(embedding), 'lim': int(target * oversampling_factor), 'br': brand_prefs.get(sub_category, [])}
                    clauses = []
                    if params['br']: clauses.append("CASE WHEN p.brand = ANY(%(br)s) THEN 1 ELSE 2 END")
                    price_range = price_ranges.get(sub_category)
                    if price_range:
                        params.update({'min_p': price_range['min'], 'max_p': price_range['max']})
                        clauses.append("CASE WHEN p.price BETWEEN %(min_p)s AND %(max_p)s THEN 1 ELSE 2 END")
                    clauses.append("pe.embedding <=> %(emb)s")

                    sql = f"""
                        SELECT p.product_id FROM veli_inventory_and_sales_fixed p JOIN product_embeddings pe ON p.product_id = pe.product_id WHERE p.subcategory = %(sub)s AND p.in_stock > 0
                        AND NOT EXISTS (SELECT 1 FROM veli_orders up JOIN veli_recommendation_rules rr ON p.subcategory = rr.name OR p.mother_cat_name = rr.name WHERE up.product_id = p.product_id AND up.user_id = %(uid)s AND rr.purchase_type = 'one_time' AND up.status NOT IN ('Start Payment', 'Failed', 'Cancelled', 'Blocked')) 
                        ORDER BY {', '.join(clauses)} LIMIT %(lim)s;
                    """
                    cur.execute(sql, params)
                    all_candidate_ids.update([row[0] for row in cur.fetchall()][:target])
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Candidate generation DB error for user {user_id}: {error}", exc_info=True)
            if conn: conn.rollback()
        finally:
            if conn: self.db_pool.putconn(conn)
        
        return self.product_catalog[self.product_catalog['product_id'].isin(all_candidate_ids)]

    @retry_with_exponential_backoff
    def _get_gemini_recommendations_with_retry(self, prompt: str):
        """აგზავნის პრომპტს Gemini API-სთან ხელახალი ცდების ლოგიკით და timeout-ით."""
        # ვუწესებთ 120 წამიან (2 წუთი) timeout-ს.
        request_options = {"timeout": 120}
        
        response = self.generative_model.generate_content(
            prompt,
            request_options=request_options
        )

        if not response.parts:
            logging.warning("Gemini API returned an empty response.")
            return None
        
        # JSON-ის გასუფთავების გაუმჯობესება
        cleaned_response = response.text.strip()
        # ვპოულობთ პირველ '{' და ბოლო '}' სიმბოლოს
        start_index = cleaned_response.find('{')
        end_index = cleaned_response.rfind('}')
        if start_index != -1 and end_index != -1:
            json_str = cleaned_response[start_index:end_index+1]
            return json.loads(json_str)
        else:
            logging.warning(f"Could not find a valid JSON object in the response: {cleaned_response}")
            return None

    def run_generative_reranking(self, history: pd.DataFrame, candidates: pd.DataFrame, slots: dict, sub_scores: dict, target_size: int) -> list:
        """ახდენს კანდიდატების საბოლოო რერანჟირებას პარალელური და გამძლე API გამოძახებებით."""
        history_summary = "User interaction history (latest first):\n" + "\n".join([f"- Action: {row.get('attribute')}, Product: '{row.get('product_name')}'" for _, row in history.head(20).iterrows()])
        
        # --- ნაბიჯი 1: შევაგროვოთ ყველა პრომპტი, რომელიც გასაშვებია ---
        prompts_to_run = {}
        for category, total_slots in slots.items():
            if total_slots == 0: continue
            cat_candidates = candidates[candidates['mother_cat_name'] == category]
            if cat_candidates.empty: continue
            
            # ... (დანარჩენი ლოგიკა პრომპტების გენერირებისთვის უცვლელია) ...
            subcats_in_main_cat = {sub: sub_scores.get(sub, 0) for sub in cat_candidates['subcategory'].dropna().unique()}
            subcat_total_score = sum(subcats_in_main_cat.values())

            if not subcats_in_main_cat or subcat_total_score == 0:
                prompt_key = f"{category}_general"
                candidates_summary = "\n".join([f"- ID: {row.get('product_id')}, Name: {row.get('product_name')}" for _, row in cat_candidates.iterrows()])
                prompts_to_run[prompt_key] = self._create_prompt(history_summary, candidates_summary, total_slots)
                continue

            subcat_slots, distributed = {}, 0
            for sub, score in sorted(subcats_in_main_cat.items(), key=lambda x: x[1], reverse=True):
                s = int(round((score / subcat_total_score) * total_slots)) if subcat_total_score > 0 else 0
                subcat_slots[sub] = s
                distributed += s
            
            i = 0
            while distributed < total_slots:
                if not subcat_slots: break # თუ ქვეკატეგორიები არ არის, გამოვიდეთ
                subcat_to_add = list(subcat_slots.keys())[i % len(subcat_slots)]
                subcat_slots[subcat_to_add] += 1
                distributed += 1
                i += 1

            for subcat, num_slots in subcat_slots.items():
                if num_slots == 0: continue
                subcat_candidates = cat_candidates[cat_candidates['subcategory'] == subcat]
                if subcat_candidates.empty: continue
                prompt_key = f"{category}_{subcat}"
                candidates_summary = "\n".join([f"- ID: {row.get('product_id')}, Name: {row.get('product_name')}" for _, row in subcat_candidates.iterrows()])
                prompts_to_run[prompt_key] = self._create_prompt(history_summary, candidates_summary, num_slots)

        # --- ნაბიჯი 2: გავუშვათ ყველა პრომპტი პარალელურად ---
        final_recs_by_category = {cat: [] for cat in slots.keys()}
        
        with ThreadPoolExecutor(max_workers=4) as executor: # ვიყენებთ იგივე რაოდენობის ვორკერებს
            # ვქმნით უსაფრთხო wrapper-ს, რომელიც არ გააჩერებს პროცესს შეცდომისას
            def safe_rerank_wrapper(prompt_key, prompt):
                try:
                    # ვიყენებთ ჩვენს ახალ, timeout-იან და retry-იან ფუნქციას
                    response = self._get_gemini_recommendations_with_retry(prompt)
                    return prompt_key, response
                except Exception as e:
                    logging.error(f"Reranking for prompt key '{prompt_key}' ultimately failed after all retries: {e}")
                    return prompt_key, None

            future_to_prompt = {executor.submit(safe_rerank_wrapper, key, prompt): key for key, prompt in prompts_to_run.items()}
            
            for future in as_completed(future_to_prompt):
                prompt_key, response = future.result()
                if response and "recommendations" in response:
                    category_name = prompt_key.split('_')[0]
                    final_recs_by_category[category_name].extend(response["recommendations"])

        # --- ნაბიჯი 3: შედეგების აწყობა (უცვლელი) ---
        final_list = []
        priority_cats = sorted(slots.keys(), key=lambda c: slots.get(c, 0), reverse=True)
        temp_recs = {cat: list(recs) for cat, recs in final_recs_by_category.items()}
        while len(final_list) < target_size:
            items_added = 0
            for category in priority_cats:
                if temp_recs.get(category):
                    final_list.append(temp_recs[category].pop(0))
                    items_added += 1
                    if len(final_list) >= target_size: break
            if items_added == 0: break
        
        return final_list

    def _create_prompt(self, history_summary: str, candidates_summary: str, num_recommendations: int) -> str:
        """ქმნის პრომპტს Gemini-სთვის."""
        return f"""You are an expert e-commerce recommender system. Your task is to analyze a user's detailed history and use it to rank the products from the provided "Available Products" list.
        **Instructions:**
        1. Analyze User History.
        2. Rank the Provided List based on the user's implicit needs and preferences.
        3. Assign a "relevance_score" from 1 to 100 to each product.
        4. Output ONLY in a valid JSON format. Recommend exactly {num_recommendations} products.
        **The JSON output must have this exact structure:**
        {{ "recommendations": [ {{ "product_id": <int>, "relevance_score": <int> }}, ... ] }}
        ---
        **User History (Most Important):**
        {history_summary}
        ---
        **Available Products to Recommend From (Rank these):**
        {candidates_summary}
        ---
        **Your JSON Response:**"""

    def save_session_to_db(self, data: dict, version: str, batch_run_id: int):
        """ინახავს სესიის მონაცემებს PostgreSQL-ის რელაციურ ცხრილებში."""
        user_id = data.get("user_id")
        if not data.get("recommended_product_ids") or not user_id: return

        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute("INSERT INTO recommendations_sessions (batch_run_id, user_id, algorithm_version) VALUES (%s, %s, %s) RETURNING id;", (batch_run_id, user_id, version))
                session_id = cur.fetchone()[0]
                profile_data, slots = data.get("profile", {}), data.get("execution_plan", {}).get("slot_distribution", {})
                
                cat_details_data = [(batch_run_id, session_id, user_id, name, details.get("main_category_score"), slots.get(name)) for name, details in profile_data.items()]
                subcat_details_data = []
                for cat, details in profile_data.items():
                    for sub, pref in details.get("subcategory_preferences", {}).items():
                        pr = pref.get("price_range")
                        subcat_details_data.append((batch_run_id, session_id, user_id, cat, sub, pref.get("score"), pref.get("brand_loyalty", []), pr.get("min") if pr else None, pr.get("max") if pr else None))
                rec_data = [(batch_run_id, session_id, item.get('product_id'), rank + 1) for rank, item in enumerate(data.get("final_recs_with_scores", []))]

                if cat_details_data: psycopg2.extras.execute_batch(cur, "INSERT INTO recommendations_category_details (batch_run_id, session_id, user_id, category_name, score, slots_allocated) VALUES (%s, %s, %s, %s, %s, %s)", cat_details_data)
                if subcat_details_data: psycopg2.extras.execute_batch(cur, "INSERT INTO recommendations_subcategory_details (batch_run_id, session_id, user_id, category_name, subcategory_name, score, preferred_brands, price_range_min, price_range_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", subcat_details_data)
                if rec_data: psycopg2.extras.execute_batch(cur, "INSERT INTO recommendations_final_list (batch_run_id, session_id, product_id, rank) VALUES (%s, %s, %s, %s)", rec_data)
                
                conn.commit()
                logging.info(f"Successfully saved recommendation session {session_id} for user {user_id} (Batch: {batch_run_id}).")
        except (Exception, psycopg2.Error) as error:
            logging.error(f"DB Save Error for user {user_id} (Batch: {batch_run_id}): {error}", exc_info=True)
            if conn: conn.rollback()
        finally:
            if conn: self.db_pool.putconn(conn)

    def generate_for_user(self, user_id: int, batch_run_id: int):
        """ასრულებს რეკომენდაციის გენერირების სრულ ციკლს ერთი მომხმარებლისთვის."""
        try:
            user_history = self.get_user_activity(user_id)
            if user_history.empty: return f"Skipped: User {user_id} has no activity history."
            
            category_scores = self.calculate_category_scores(user_history)
            if not category_scores: return f"Skipped: Could not calculate category scores for user {user_id}."
            
            subcategory_scores, price_ranges, brand_preferences = self.calculate_subcategory_scores(user_history), self.calculate_price_ranges(user_history), self.calculate_brand_preferences(user_history)
            candidates = self.get_candidate_products(user_id, category_scores, subcategory_scores, price_ranges, brand_preferences)
            if candidates.empty: return f"Skipped: No candidate products found for user {user_id}."

            final_list_target_size, max_slots_per_category, slots_per_category, total_score = 30, 10, {}, sum(category_scores.values())
            if total_score > 0:
                for cat, score in category_scores.items(): slots_per_category[cat] = int(round((score / total_score) * final_list_target_size))
                unassigned_slots = sum(max(0, s - max_slots_per_category) for s in slots_per_category.values())
                for cat, slots in slots_per_category.items():
                    if slots > max_slots_per_category: slots_per_category[cat] = max_slots_per_category
                eligible_cats = {c: s for c, s in category_scores.items() if slots_per_category.get(c, 0) < max_slots_per_category}
                eligible_total_score = sum(eligible_cats.values())
                if unassigned_slots > 0 and eligible_total_score > 0:
                    for cat, score in eligible_cats.items(): slots_per_category[cat] += int(round((score / eligible_total_score) * unassigned_slots))
            
            distributed, i, sorted_cats_by_score = sum(slots_per_category.values()), 0, sorted(category_scores.keys(), key=lambda c: category_scores.get(c, 0), reverse=True)
            while distributed < final_list_target_size:
                cat_to_add = sorted_cats_by_score[i % len(sorted_cats_by_score)]
                if slots_per_category.get(cat_to_add, 0) < max_slots_per_category:
                    slots_per_category[cat_to_add] = slots_per_category.get(cat_to_add, 0) + 1; distributed += 1
                i += 1;
                if i > len(sorted_cats_by_score) * 2: break 

            final_list = self.run_generative_reranking(user_history, candidates, slots_per_category, subcategory_scores, final_list_target_size)
            
            profile_data = {}
            for cat, score in category_scores.items(): profile_data[cat] = {"main_category_score": score, "subcategory_preferences": {}}
            for sub, score in subcategory_scores.items():
                main_cat = self.sub_to_main_cat_map.get(sub)
                if main_cat in profile_data: profile_data[main_cat]["subcategory_preferences"][sub] = {"score": score, "brand_loyalty": brand_preferences.get(sub, []), "price_range": price_ranges.get(sub, None)}

            output_data = {
                "user_id": user_id, "request_timestamp": datetime.now(timezone.utc).isoformat(),
                "profile": profile_data, "execution_plan": {"slot_distribution": slots_per_category, "candidate_pool_size": len(candidates)},
                "final_recs_with_scores": final_list, "recommended_product_ids": [item.get('product_id') for item in final_list if item]
            }
            self.save_session_to_db(output_data, "v2.0-robust", batch_run_id=batch_run_id)
            return f"Success: Recommendations generated for user {user_id}."
        except Exception as e:
            logging.error(f"CRITICAL ERROR for user {user_id}: {e}", exc_info=True)
            return f"Failed: An error occurred for user {user_id}."


# --- სკრიპტის გაშვების ბლოკი ---
if __name__ == "__main__":
    user_fetch_query = """
    select distinct user_id from (select user_id, count(distinct id) as order_number, round(sum(revenue)::numeric, 2) as total_revenue, max(paid_Date) as last_order
    from veli_orders where status not in ('Start Payment', 'Failed', 'Cancelled', 'Blocked')
    group by user_id having count(distinct id) > 5 and sum(revenue)::numeric > 1000 and max(paid_Date) >= '2025-03-01'
    order by order_number desc, total_revenue desc) t1 limit 10000 OFFSET 0;
    """
    ALGORITHM_VERSION = "v2.4-robust-batching"
    MAX_WORKERS = 4

    start_time = time.time()
    recommender = None
    try:
        recommender = RecommenderSystem()
        user_ids_to_process = recommender.get_user_ids_from_query(user_fetch_query)

        if not user_ids_to_process:
            logging.warning("No user IDs to process based on the query. The script will now exit.")
            sys.exit(0)
        
        logging.info(f"Found {len(user_ids_to_process)} users to process. Starting batch run...")
        
        conn = recommender.db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO recommendations_batch_runs (algorithm_version) VALUES (%s) RETURNING id;", (ALGORITHM_VERSION,))
            batch_run_id = cur.fetchone()[0]
            conn.commit()
        recommender.db_pool.putconn(conn)
        logging.info(f"Created new batch run with ID: {batch_run_id}")

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(recommender.generate_for_user, uid, batch_run_id): uid for uid in user_ids_to_process}
            progress_bar = tqdm(as_completed(futures), total=len(user_ids_to_process), desc=f"Processing Batch ID: {batch_run_id}")
            for future in progress_bar:
                user_id = futures[future]
                try:
                    result_message = future.result()
                    results.append(result_message)
                    progress_bar.set_postfix_str(f"Last: {result_message}")
                except Exception as exc:
                    error_message = f"User {user_id} generated an unhandled exception: {exc}"
                    logging.error(error_message, exc_info=True)
                    results.append(error_message)

        total_time = time.time() - start_time
        
        conn = recommender.db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("UPDATE recommendations_batch_runs SET processed_users_count = %s, duration_seconds = %s WHERE id = %s;", (len(user_ids_to_process), total_time, batch_run_id))
            conn.commit()
        recommender.db_pool.putconn(conn)
        logging.info(f"Finalized batch run {batch_run_id}.")

        success_count = sum(1 for r in results if r and r.startswith("Success"))
        skipped_count = sum(1 for r in results if r and r.startswith("Skipped"))
        failed_count = len(results) - success_count - skipped_count

        print("\n" + "="*60 + "\n        BATCH PROCESSING SUMMARY\n" + "="*60)
        print(f"Batch ID: {batch_run_id}")
        print(f"Total time: {total_time:.2f} seconds.")
        print(f"Processed {len(user_ids_to_process)} users.")
        if user_ids_to_process: print(f"Average time per user: {total_time / len(user_ids_to_process):.2f} seconds.")
        print("-" * 20 + f"\nSuccessful generations: {success_count}\nSkipped (no history, etc.): {skipped_count}\nFailed (errors): {failed_count}\n" + "="*60)

        if failed_count > 0:
            print("\n" + "-"*60 + "\n        DETAILS ON FAILED/SKIPPED USERS\n" + "-"*60)
            for res in results:
                if not res.startswith("Success"): print(res)
            print("-" * 60)

    except (Exception, psycopg2.Error) as e:
        logging.critical(f"A critical error occurred in the main execution block: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if recommender and recommender.db_pool:
            recommender.db_pool.closeall()
            logging.info("Database connection pool has been closed.")