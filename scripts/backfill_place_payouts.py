# -*- coding: utf-8 -*-
"""
서울·제주·부산경남 연승(place) 확정 배당 소급 백필 스크립트.

`library/collector_races.py`가 이미 쓰는 공공데이터포털 공용 API
(raceResult_3 + meet, rc_month=YYYYMM 방식)는
과거 날짜 소급조회가 가능하고, 응답의 개별 말 `plcOdds` 필드가 실제
확정 연승 배당임이 검증되었다(winOdds가 race_entries.win_odds와 완전
일치 - [[project_race_payouts_scraper]] 참고). ScoreInfo.do(소급 불가
확정, 2026-08-02)와 달리 이 경로는 진짜 과거 데이터를 준다.

서울 전용 SeoulRace_1에는 말 개별 plcOdds 필드가 없으므로, 서울도 공용
API(common_api=True)로 조회한다 - PLC_ODDS_SUPPORTED_MEET_CODES = {1, 2, 3}.

주의: 이 스크립트는 `races`/`race_entries` 테이블은 건드리지 않는다.
`race_payouts.race_id`가 `races(race_id)` FK이므로, 대상 기간의 races가
이미 DB에 있다는 전제(과거 정기 수집으로 이미 채워져 있음)로 동작한다 -
races에 없는 race_id로 race_payouts를 upsert하면 FK 위반으로 실패하며,
이는 의도된 안전장치다(엉뚱한 라벨로 잘못 채워지는 사고 방지).

사용:
    python scripts/backfill_place_payouts.py --start 2026-05-01 --end 2026-06-25
"""
import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library.collector_races import (
    fetch_race_data_all,
    parse_and_normalize_race_data,
    filter_duplicates_in_payouts,
    PLC_ODDS_SUPPORTED_MEET_CODES,
)
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

MEET_NAMES = {1: "서울", 2: "제주", 3: "부산경남"}


def _months_between(start: str, end: str) -> list[str]:
    """'2026-05-01'~'2026-06-25' -> ['202605', '202606']"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months = []
    cur = s.replace(day=1)
    while cur <= e:
        months.append(cur.strftime("%Y%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


def fetch_month_payouts(meet_code: int, rc_month: str, start: str, end: str) -> list[dict]:
    """meet_code의 rc_month 전체를 페이지네이션으로 수집 -> [start, end] 범위의
    race_payouts(bet_type='place')만 필터해서 반환."""
    all_payouts = []
    page = 1
    per_page = 1000
    while True:
        api_data, _ = fetch_race_data_all(page=page, per_page=per_page, meet_code=meet_code, end_date=rc_month, common_api=True)
        if not api_data:
            logger.warning(f"  {MEET_NAMES.get(meet_code)} {rc_month} 페이지 {page}: API 응답 없음, 중단")
            break

        body = api_data.get("response", {}).get("body", {})
        items = body.get("items", []) if body else []
        item_list = items.get("item", []) if items else []
        n_items = len(item_list) if isinstance(item_list, list) else (1 if item_list else 0)

        parsed = parse_and_normalize_race_data(api_data)
        payouts = parsed.get("race_payouts", [])
        # race_id = "YYYY-MM-DD_meet_raceNo" - 날짜 prefix로 범위 필터
        in_range = [p for p in payouts if start <= p["race_id"][:10] <= end]
        all_payouts.extend(in_range)

        logger.info(f"  {MEET_NAMES.get(meet_code)} {rc_month} 페이지 {page}: "
                    f"{n_items}건 조회, place payouts {len(payouts)}건(범위내 {len(in_range)}건)")

        if n_items < per_page:
            break
        page += 1
        time.sleep(0.2)

    return all_payouts


def backfill(start: str, end: str, meet_codes: list[int] = None) -> dict:
    meet_codes = meet_codes or sorted(PLC_ODDS_SUPPORTED_MEET_CODES)
    months = _months_between(start, end)
    client = get_client()

    results = {}
    for meet_code in meet_codes:
        meet_name = MEET_NAMES.get(meet_code, str(meet_code))
        logger.info(f"=== {meet_name}(meet_code={meet_code}) 백필 시작: {start}~{end} (월: {months}) ===")

        all_payouts = []
        api_failed = False
        for rc_month in months:
            try:
                month_payouts = fetch_month_payouts(meet_code, rc_month, start, end)
                if not month_payouts:
                    # 페이지 1에서부터 API가 아예 실패했는지(500 등) 구분하기 위해
                    # 최소 한 번은 원시 응답을 확인한다.
                    probe, _ = fetch_race_data_all(page=1, per_page=1, meet_code=meet_code, end_date=rc_month, common_api=True)
                    if not probe:
                        api_failed = True
                all_payouts.extend(month_payouts)
            except Exception as e:
                logger.error(f"  {meet_name} {rc_month} 처리 중 오류: {e}")
                api_failed = True

        if api_failed and not all_payouts:
            logger.error(f"{meet_name}: API 호출 실패로 이 경마장은 스킵합니다 (외부 API 장애 가능성).")
            results[meet_name] = {"fetched": 0, "upserted": 0, "api_failed": True}
            continue

        deduped = filter_duplicates_in_payouts(all_payouts)
        logger.info(f"{meet_name}: 수집 {len(all_payouts)}건 -> dedup 후 {len(deduped)}건")

        upserted = 0
        batch_size = 200
        for i in range(0, len(deduped), batch_size):
            batch = deduped[i:i + batch_size]
            try:
                res = client.table("race_payouts").upsert(
                    batch, on_conflict="race_id,bet_type,entry_numbers"
                ).execute()
                upserted += len(res.data) if res.data else 0
            except Exception as e:
                logger.error(f"  배치 upsert 실패({i}~{i+len(batch)}): {str(e)[:200]}")
                # FK 위반(races에 없는 race_id) 등은 개별 재시도로 원인 격리
                for row in batch:
                    try:
                        r = client.table("race_payouts").upsert(
                            [row], on_conflict="race_id,bet_type,entry_numbers"
                        ).execute()
                        upserted += len(r.data) if r.data else 0
                    except Exception as e2:
                        logger.warning(f"    개별 upsert 실패 {row['race_id']} entry={row['entry_numbers']}: {str(e2)[:150]}")
            time.sleep(0.1)

        logger.info(f"{meet_name}: upsert 완료 {upserted}/{len(deduped)}행")
        results[meet_name] = {"fetched": len(deduped), "upserted": upserted, "api_failed": False}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="제주·부산경남 연승(place) 확정 배당 백필")
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end", default="2026-06-25")
    parser.add_argument("--meets", default="2,3", help="meet_code 콤마구분 (기본: 2,3 = 제주,부산경남)")
    args = parser.parse_args()

    meet_codes = [int(x) for x in args.meets.split(",")]
    print(f"백필 기간: {args.start} ~ {args.end}, 대상 meet_code: {meet_codes}")
    results = backfill(args.start, args.end, meet_codes)
    print("\n=== 결과 요약 ===")
    for name, r in results.items():
        status = "API 실패(스킵)" if r["api_failed"] else "완료"
        print(f"  {name}: {status} - fetched={r['fetched']}, upserted={r['upserted']}")
