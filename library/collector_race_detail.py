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
from library.parsing import parse_date, safe_int, safe_float, safe_str
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

ENDPOINT = "https://apis.data.go.kr/B551015/API299/Race_Result_total"

# race_entries/races 테이블의 race_id는 "부산경남"을 쓰는데(library/collector_races.py의
# MEET_CODE_MAP 참고), API299는 같은 트랙을 "부경"으로 반환한다. item.get("meet")을 그대로
# race_id에 쓰면 부산경남 경주(전체의 약 31%)가 항상 조인 실패해 신규 피처가 전부 결측으로
# 빠지는 사고로 이어진다(2026-07-20 발견). 요청 시점에 이미 알고 있는 meet 코드로 canonical
# 라벨을 강제해 이 불일치를 원천 차단한다.
MEET_LABEL = {1: "서울", 2: "제주", 3: "부산경남"}


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


def _build_race_id(item: dict, meet: int) -> str | None:
    race_date = parse_date(item.get("rcDate"))
    meet_str = MEET_LABEL.get(meet)
    race_no = safe_int(item.get("rcNo"))
    if not race_date or not meet_str or not race_no:
        return None
    return f"{race_date}_{meet_str}_{race_no}"


def _safe_int_or_none(value):
    """safe_int는 0을 falsy로 취급해 default로 치환해버리므로(승수 0은 유효값),
    성적 필드 파싱에는 이 헬퍼를 쓴다."""
    try:
        return int(value) if value not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None


def _safe_ord_or_none(value):
    """통과순위는 1등부터 시작하므로 0은 실제 순위가 아니라 "미해당"(짧은 거리에서
    1C/2C 통과지점이 없음, 구간기록 API의 0.0과 같은 패턴)을 뜻한다. 0도 None으로 취급한다."""
    parsed = _safe_int_or_none(value)
    return None if parsed == 0 else parsed


def parse_item(item: dict, meet: int) -> dict | None:
    """1단계: 부담중량 + 2단계: 말/기수/조교사 통산·최근1년 성적. 3단계에서 통과순위를 추가한다.

    라이브 API 확인 결과(2026-07-19), 계획서가 가정한 ord1CntT/ord1CntY/rcCntT/rcCntY는
    말의 성적이 아니다(값이 항상 T==Y이고 비정상적으로 작음 — 용도 불명, 사용하지 않음).
    말의 실제 통산 성적은 hrOrd1CntT(통산 1착)/hrOrd2CntT(통산 2착)/hrRcCntT(통산 출전)이며,
    이 API에는 말의 "최근 1년" 성적 필드가 존재하지 않는다(Y variant 없음) — 그래서
    horse_recent1y_wins/starts는 채우지 않고 NULL로 남긴다. 기수/조교사는 T/Y 필드가
    모두 존재하고 통산>=최근1년 관계도 확인되어 계획대로 사용한다.

    meet: 요청한 경마장 코드(1/2/3). race_id의 트랙 라벨은 item.get("meet")이 아니라
    이 코드로 정규화한다(MEET_LABEL 주석 참고).
    """
    race_id = _build_race_id(item, meet)
    horse_id = safe_str(item.get("hrNo"))
    if not race_id or not horse_id:
        return None
    return {
        "race_id": race_id,
        "horse_id": horse_id,
        "assigned_weight": safe_float(item.get("wgBudam"), default=None),
        "horse_career_wins": _safe_int_or_none(item.get("hrOrd1CntT")),
        "horse_career_starts": _safe_int_or_none(item.get("hrRcCntT")),
        "jockey_career_wins": _safe_int_or_none(item.get("jkOrd1CntT")),
        "jockey_career_starts": _safe_int_or_none(item.get("jkRcCntT")),
        "jockey_recent1y_wins": _safe_int_or_none(item.get("jkOrd1CntY")),
        "jockey_recent1y_starts": _safe_int_or_none(item.get("jkRcCntY")),
        "trainer_career_wins": _safe_int_or_none(item.get("trOrd1CntT")),
        "trainer_career_starts": _safe_int_or_none(item.get("trRcCntT")),
        "trainer_recent1y_wins": _safe_int_or_none(item.get("trOrd1CntY")),
        "trainer_recent1y_starts": _safe_int_or_none(item.get("trRcCntY")),
        "sj_1c_ord": _safe_ord_or_none(item.get("sj_1cOrd")),
        "sj_2c_ord": _safe_ord_or_none(item.get("sj_2cOrd")),
        "sj_3c_ord": _safe_ord_or_none(item.get("sj_3cOrd")),
        "sj_4c_ord": _safe_ord_or_none(item.get("sj_4cOrd")),
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
            parsed = parse_item(it, meet)
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
    logger.info(f"경주결과종합 수집 완료: {start_date}~{end_date} → 총 {total}건")
    return total


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python -m library.collector_race_detail START END  (YYYY-MM-DD)")
        sys.exit(1)
    collect_range(sys.argv[1], sys.argv[2])
