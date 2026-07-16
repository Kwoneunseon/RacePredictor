# -*- coding: utf-8 -*-
"""
경마시행당일 경주결과종합 수집기 — 부담중량·(2단계)성적·(3단계)통과순위
API ID : 15119524
Endpoint: https://apis.data.go.kr/B551015/API299/Race_Result_total
Request : meet(1/2/3), rc_date(YYYYMMDD), pageNo, numOfRows
"""
import requests
import time

from config import API_KEY
from Utils import parse_date, safe_int, safe_float, safe_str
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

ENDPOINT = "https://apis.data.go.kr/B551015/API299/Race_Result_total"


def fetch_race_detail(meet: int, rc_date: str, page: int = 1, per_page: int = 200):
    """
    meet: 1(서울)/2(제주)/3(부산경남), rc_date: 'YYYYMMDD'
    Returns list of raw item dicts, or None on error.
    """
    params = {
        "serviceKey": API_KEY,
        "meet": meet,
        "rc_date": rc_date,
        "pageNo": page,
        "numOfRows": per_page,
        "_type": "json",
    }
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning("경주결과종합 API 403 Forbidden — data.go.kr 활용신청 승인 확인 필요")
            return None
        if resp.status_code != 200:
            logger.error(f"경주결과종합 API 오류: {resp.status_code}")
            return None
        data = resp.json()
        body = data.get("response", {}).get("body", {})
        items = body.get("items") or {}
        raw = items.get("item", [])
        if isinstance(raw, dict):
            raw = [raw]
        return raw
    except Exception as exc:
        logger.error(f"경주결과종합 API 호출 오류: {exc}")
        return None


def _build_race_id(item: dict) -> str | None:
    race_date = parse_date(item.get("rcDate"))
    meet_str = safe_str(item.get("meet"))
    race_no = safe_int(item.get("rcNo"))
    if not race_date or not meet_str or not race_no:
        return None
    return f"{race_date}_{meet_str}_{race_no}"


def parse_item(item: dict) -> dict | None:
    """1단계: assigned_weight(wgBudam)만 파싱한다. 2·3단계에서 이 함수를 확장한다."""
    race_id = _build_race_id(item)
    horse_id = safe_str(item.get("hrNo"))
    if not race_id or not horse_id:
        return None
    return {
        "race_id": race_id,
        "horse_id": horse_id,
        "assigned_weight": safe_float(item.get("wgBudam"), default=None),
    }


def save_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    client = get_client()
    client.table("race_entry_detail").upsert(
        rows, on_conflict="race_id,horse_id"
    ).execute()
    return len(rows)


def collect_meet_date(meet: int, rc_date: str) -> int:
    """특정 경마장·경주일자의 전체 데이터 수집 (페이지네이션)."""
    # (race_id, horse_id) 기준 dict로 모아서 중복 제거 — API가 동일 레코드를 중복 반환하는 경우가 있어
    # (예: 2026-02-15_서울_10 경주 0060361), 한 upsert 배치 내 중복 키가 있으면 Postgres가
    # "ON CONFLICT DO UPDATE command cannot affect row a second time" 오류를 낸다.
    rows_by_key: dict[tuple[str, str], dict] = {}
    page = 1
    while True:
        items = fetch_race_detail(meet, rc_date, page=page)
        if items is None:
            return 0
        if not items:
            break
        for it in items:
            parsed = parse_item(it)
            if parsed:
                rows_by_key[(parsed["race_id"], parsed["horse_id"])] = parsed
        if len(items) < 200:
            break
        page += 1
        time.sleep(0.2)

    saved = save_rows(list(rows_by_key.values()))
    logger.info(f"경주결과종합 저장: meet={meet} {rc_date} → {saved}건")
    return saved


def collect_range(start_date: str, end_date: str) -> int:
    """start_date, end_date: 'YYYY-MM-DD'. 기간 내 실제 경주가 있었던 (경마장,날짜) 조합만 수집."""
    client = get_client()
    res = client.table("races").select("race_date, meet_code")\
        .gte("race_date", start_date).lte("race_date", end_date).execute()
    combos = sorted({(r["race_date"], r["meet_code"]) for r in (res.data or [])})

    total = 0
    for race_date, meet_code in combos:
        rc_date = race_date.replace("-", "")
        cnt = collect_meet_date(meet_code, rc_date)
        total += cnt
        time.sleep(0.2)
    logger.info(f"경주결과종합 수집 완료: {start_date}~{end_date} → 총 {total}건")
    return total


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python -m library.collector_race_detail START END  (YYYY-MM-DD)")
        sys.exit(1)
    collect_range(sys.argv[1], sys.argv[2])
