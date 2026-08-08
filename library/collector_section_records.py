# -*- coding: utf-8 -*-
"""
마필 구간별 경주기록 수집기
API ID : 15057859
Endpoint: http://apis.data.go.kr/B551015/API37_1/sectionRecord_1
Request : meet(1/2/3), rc_date(YYYYMMDD), pageNo, numOfRows
Response: 말별·거리별 구간(S1F,1C~4C,G3F,G1F) 최고(F)/평균(A)/최저(S) 기록
"""
import requests
import time

from config import API_KEY
from library.parsing import parse_date, safe_str
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

ENDPOINT = "http://apis.data.go.kr/B551015/API37_1/sectionRecord_1"


def _safe_num_or_none(value):
    try:
        return float(value) if value not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None


def fetch_section_records(meet: int, rc_date: str, page: int = 1, per_page: int = 200):
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
            logger.warning("구간기록 API 403 Forbidden — data.go.kr 활용신청 승인 확인 필요")
            return None
        if resp.status_code != 200:
            logger.error(f"구간기록 API 오류: {resp.status_code}")
            return None
        data = resp.json()
        body = data.get("response", {}).get("body", {})
        items = body.get("items") or {}
        raw = items.get("item", [])
        if isinstance(raw, dict):
            raw = [raw]
        return raw
    except Exception as exc:
        logger.error(f"구간기록 API 호출 오류: {exc}")
        return None


def parse_item(item: dict, snapshot_race_date: str) -> dict | None:
    horse_id = safe_str(item.get("hrNo"))
    race_distance = item.get("rcDist")
    if not horse_id or not race_distance:
        return None
    return {
        "horse_id": horse_id,
        "race_distance": int(race_distance),
        "snapshot_race_date": snapshot_race_date,
        "s1f_best": _safe_num_or_none(item.get("FRcTimeS1f")),
        "s1f_avg": _safe_num_or_none(item.get("ARcTimeS1f")),
        "s1f_worst": _safe_num_or_none(item.get("SRcTimeS1f")),
        "c1_best": _safe_num_or_none(item.get("FRcTime_1c")),
        "c1_avg": _safe_num_or_none(item.get("ARcTime_1c")),
        "c1_worst": _safe_num_or_none(item.get("SRcTime_1c")),
        "c2_best": _safe_num_or_none(item.get("FRcTime_2c")),
        "c2_avg": _safe_num_or_none(item.get("ARcTime_2c")),
        "c2_worst": _safe_num_or_none(item.get("SRcTime_2c")),
        "c3_best": _safe_num_or_none(item.get("FRcTime_3c")),
        "c3_avg": _safe_num_or_none(item.get("ARcTime_3c")),
        "c3_worst": _safe_num_or_none(item.get("SRcTime_3c")),
        "c4_best": _safe_num_or_none(item.get("FRcTime_4c")),
        "c4_avg": _safe_num_or_none(item.get("ARcTime_4c")),
        "c4_worst": _safe_num_or_none(item.get("SRcTime_4c")),
        "g3f_best": _safe_num_or_none(item.get("FRcTimeG3f")),
        "g3f_avg": _safe_num_or_none(item.get("ARcTimeG3f")),
        "g3f_worst": _safe_num_or_none(item.get("SRcTimeG3f")),
        "g1f_best": _safe_num_or_none(item.get("FRcTimeG1f")),
        "g1f_avg": _safe_num_or_none(item.get("ARcTimeG1f")),
        "g1f_worst": _safe_num_or_none(item.get("SRcTimeG1f")),
    }


def save_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    client = get_client()
    client.table("horse_section_records").upsert(
        rows, on_conflict="horse_id,race_distance,snapshot_race_date"
    ).execute()
    return len(rows)


def collect_meet_date(meet: int, rc_date: str) -> int:
    all_rows = []
    page = 1
    snapshot_date = parse_date(rc_date)
    while True:
        items = fetch_section_records(meet, rc_date, page=page)
        if items is None:
            return 0
        if not items:
            break
        for it in items:
            parsed = parse_item(it, snapshot_date)
            if parsed:
                all_rows.append(parsed)
        if len(items) < 200:
            break
        page += 1
        time.sleep(0.2)

    saved = save_rows(all_rows)
    logger.info(f"구간기록 저장: meet={meet} {rc_date} → {saved}건")
    return saved


def collect_range(start_date: str, end_date: str) -> int:
    client = get_client()
    all_rows = []
    page_size, offset = 1000, 0
    while True:
        res = client.table("races").select("race_date, meet_code")\
            .gte("race_date", start_date).lte("race_date", end_date)\
            .order("race_date")\
            .range(offset, offset + page_size - 1).execute()
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(res.data) < page_size:
            break
        offset += page_size
    combos = sorted({(r["race_date"], r["meet_code"]) for r in all_rows})

    total = 0
    for race_date, meet_code in combos:
        rc_date = race_date.replace("-", "")
        cnt = collect_meet_date(meet_code, rc_date)
        total += cnt
        time.sleep(0.2)
    logger.info(f"구간기록 수집 완료: {start_date}~{end_date} → 총 {total}건")
    return total


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python -m library.collector_section_records START END  (YYYY-MM-DD)")
        sys.exit(1)
    collect_range(sys.argv[1], sys.argv[2])
