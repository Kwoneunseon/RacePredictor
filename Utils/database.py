import time
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from postgrest._sync.request_builder import SyncQueryRequestBuilder
from config import SUPABASE_URL, SUPABASE_KEY, DATABASE_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)
_client: Optional[Client] = None

# race_entries 4천여 건 조회부터 수집기 API 백필까지 장시간 실행되는 스크립트들이
# 간헐적인 DNS/연결 끊김(getaddrinfo failed, RemoteProtocolError 등)으로 중간에
# 죽는 사고가 반복되어, PostgREST 쿼리 빌더의 execute()를 한 곳에서 재시도하도록 패치한다.
# .table()/.select()/.upsert() 등 모든 호출부가 공통으로 이 execute()를 거치므로
# 개별 호출부를 일일이 수정하지 않아도 된다.
_original_execute = SyncQueryRequestBuilder.execute


def _execute_with_retry(self, *args, **kwargs):
    max_retries = DATABASE_CONFIG["max_retries"]
    retry_delay = DATABASE_CONFIG["retry_delay"]
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return _original_execute(self, *args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Supabase 쿼리 실패, 재시도 {attempt + 1}/{max_retries}: {e}"
                )
                time.sleep(retry_delay * (attempt + 1))
    raise last_error


SyncQueryRequestBuilder.execute = _execute_with_retry


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
