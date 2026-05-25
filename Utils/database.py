import time
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, DATABASE_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)
_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase 클라이언트 생성됨")
    return _client


def execute_query(table: str, query_fn) -> Optional[Any]:
    client = get_client()
    for attempt in range(DATABASE_CONFIG["max_retries"]):
        try:
            result = query_fn(client.table(table))
            return result.data
        except Exception as e:
            logger.error(f"쿼리 실패 (시도 {attempt + 1}): {e}")
            if attempt < DATABASE_CONFIG["max_retries"] - 1:
                time.sleep(DATABASE_CONFIG["retry_delay"] * (attempt + 1))
    return None


def upsert(table: str, rows: List[Dict], on_conflict: str) -> bool:
    if not rows:
        return True
    try:
        get_client().table(table).upsert(rows, on_conflict=on_conflict).execute()
        logger.info(f"{table} upsert {len(rows)}건 완료")
        return True
    except Exception as e:
        logger.error(f"{table} upsert 실패: {e}")
        return False


def test_connection() -> bool:
    try:
        result = get_client().table("horses").select("horse_id").limit(1).execute()
        return result.data is not None
    except Exception:
        return False
