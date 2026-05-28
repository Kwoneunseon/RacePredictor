# -*- coding: utf-8 -*-
"""
조교현황 수집기 — 한국마사회_홈페이지자료실_경마기록실_서울출발조교현황
API ID : 15140265
Endpoint: https://apis.data.go.kr/B551015/API329/textDataSeGtscol
Request : trng_dt (YYYYMMDD), pageNo, numOfRows
Response: trngDt, beloNo, beloTrngNo, hrnm, ridrNm, remkTxt

NOTE: API 이용 신청 필요 → data.go.kr에서 15140265 API 신청 후 사용 가능.
      현재 serviceKey가 등록되지 않으면 403이 반환됩니다.
"""
import requests
import time
from datetime import datetime, timedelta

from config import API_KEY
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

ENDPOINT = "https://apis.data.go.kr/B551015/API329/textDataSeGtscol"


def fetch_training_data(trng_dt: str, page: int = 1, per_page: int = 1000):
    """
    trng_dt: 'YYYYMMDD' 형식 훈련일자
    Returns list of raw item dicts, or None on error.
    """
    params = {
        "serviceKey": API_KEY,
        "trng_dt": trng_dt,
        "pageNo": page,
        "numOfRows": per_page,
        "_type": "json",
    }
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning(
                "조교현황 API 403 Forbidden — data.go.kr에서 API ID 15140265 이용 신청이 필요합니다."
            )
            return None
        if resp.status_code != 200:
            logger.error(f"조교현황 API 오류: {resp.status_code}")
            return None
        data = resp.json()
        body = data.get("response", {}).get("body", {})
        items = body.get("items") or {}
        raw = items.get("item", [])
        if isinstance(raw, dict):
            raw = [raw]
        return raw
    except Exception as exc:
        logger.error(f"조교현황 API 호출 오류: {exc}")
        return None


def parse_item(item: dict) -> dict:
    return {
        "trng_dt":      item.get("trngDt", "")[:10],   # YYYY-MM-DD or YYYYMMDD
        "hrnm":         str(item.get("hrnm", "") or "").strip(),
        "belo_no":      str(item.get("beloNo", "") or "").strip(),
        "belo_trng_no": str(item.get("beloTrngNo", "") or "").strip(),
        "ridr_nm":      str(item.get("ridrNm", "") or "").strip(),
        "remk_txt":     str(item.get("remkTxt", "") or "").strip(),
    }


def save_sessions(rows: list[dict]) -> int:
    if not rows:
        return 0
    client = get_client()
    # upsert on (trng_dt, hrnm, belo_trng_no)
    client.table("horse_training_sessions").upsert(
        rows, on_conflict="trng_dt,hrnm,belo_trng_no"
    ).execute()
    return len(rows)


def collect_training_date(trng_dt: str) -> int:
    """특정 훈련일자의 전체 데이터 수집 (페이지네이션)."""
    all_rows = []
    page = 1
    while True:
        items = fetch_training_data(trng_dt, page=page)
        if items is None:
            return 0
        if not items:
            break
        all_rows.extend(parse_item(i) for i in items)
        if len(items) < 1000:
            break
        page += 1
        time.sleep(0.2)

    saved = save_sessions(all_rows)
    logger.info(f"조교현황 저장: {trng_dt} → {saved}건")
    return saved


def collect_range(start_date: str, end_date: str) -> int:
    """
    start_date, end_date: 'YYYY-MM-DD' 형식
    경주일과 인접한 훈련일자(경주일 -1 ~ -7일)를 수집.
    """
    from Utils.database import get_client as _gc
    client = _gc()

    # 수집 대상 경주일 조회
    res = client.table("races").select("race_date").gte("race_date", start_date)\
        .lte("race_date", end_date).execute()
    race_dates = sorted({r["race_date"] for r in (res.data or [])})

    total = 0
    for rd in race_dates:
        dt = datetime.strptime(rd, "%Y-%m-%d")
        for delta in range(1, 8):               # 경주일 전 1~7일
            trng_date = (dt - timedelta(days=delta)).strftime("%Y%m%d")
            cnt = collect_training_date(trng_date)
            total += cnt
            if cnt == 0:
                break                           # 해당 날짜 데이터 없으면 더 이전 날짜도 없음
    logger.info(f"조교현황 수집 완료: {start_date}~{end_date} → 총 {total}건")
    return total
