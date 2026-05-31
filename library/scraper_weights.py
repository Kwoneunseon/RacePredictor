# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from Utils.logger import get_logger

logger = get_logger(__name__)

WEIGHT_URL = "https://race.kra.co.kr/raceFastreport/ChuljumaWeightList.do"
MEET_NAME_MAP = {1: "서울", 2: "제주", 3: "부산경남"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": WEIGHT_URL,
}


def fetch_race_weights(meet_code: int, race_no: int, race_date: str) -> dict[int, dict]:
    """
    KRA 출주마체중 페이지에서 특정 경주의 마체중을 가져옴.

    반환: {출전번호: {"horse_weight": int, "horse_weight_diff": int}}
    """
    date_str = race_date.replace("-", "")  # YYYY-MM-DD → YYYYMMDD
    data = {"meet": str(meet_code), "rcNo": str(race_no), "date": date_str, "Act": "03", "Sub": "4"}

    try:
        resp = requests.post(WEIGHT_URL, data=data, headers=HEADERS, timeout=15)
        resp.encoding = "euc-kr"
        if resp.status_code != 200:
            logger.warning(f"마체중 페이지 응답 오류: {resp.status_code}")
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        if len(tables) < 3:
            logger.warning(f"마체중 테이블 없음 (meet={meet_code}, rcNo={race_no}, date={date_str})")
            return {}

        result = {}
        # 3번째 테이블: 번호 | 말명 | 마체중 | 전회차 | ...
        rows = tables[2].find_all("tr")
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 3:
                continue
            try:
                entry_no = int(cells[0])
                weight = int(cells[2]) if cells[2] else 0
                diff_str = cells[3] if len(cells) > 3 else "0"
                diff = int(diff_str) if diff_str and diff_str not in ("-", "") else 0
                result[entry_no] = {"horse_weight": weight, "horse_weight_diff": diff}
            except (ValueError, IndexError):
                continue

        meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))
        logger.info(f"마체중 수집 완료: {meet_name} {race_no}R → {len(result)}마리")
        return result

    except Exception as e:
        logger.error(f"마체중 수집 실패 (meet={meet_code}, rcNo={race_no}): {e}")
        return {}


def fetch_all_weights_for_date(meet_code: int, race_date: str, max_races: int = 12) -> dict[int, dict[int, dict]]:
    """
    특정 경마장·날짜의 전 경주 마체중을 수집.

    반환: {race_no: {출전번호: {"horse_weight": int, "horse_weight_diff": int}}}
    """
    all_weights = {}
    for race_no in range(1, max_races + 1):
        weights = fetch_race_weights(meet_code, race_no, race_date)
        if not weights:
            break
        all_weights[race_no] = weights
    return all_weights


def update_weights_to_db(meet_code: int, race_date: str, max_races: int = 12) -> int:
    """
    스크래핑한 마체중을 race_entries DB에 upsert.
    horse_weight=0인 행만 대상으로 업데이트.

    반환: 업데이트된 건수
    """
    from Utils.database import get_client

    client = get_client()
    date_str = race_date.replace("-", "")
    meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))

    # 오늘 해당 경마장의 horse_weight=0인 race_entries 조회
    res = client.table("race_entries")\
        .select("race_id, horse_id, entry_number, horse_weight")\
        .eq("race_date", race_date)\
        .execute()

    if not res.data:
        logger.info(f"{meet_name} {race_date} race_entries 없음")
        return 0

    # meet_code 필터 (race_id 형식: YYYY-MM-DD_경마장명_경주번호)
    target_entries = [
        r for r in res.data
        if meet_name in r["race_id"] and (not r["horse_weight"] or r["horse_weight"] == 0)
    ]

    if not target_entries:
        logger.info(f"{meet_name} {race_date} 업데이트할 horse_weight=0 없음")
        return 0

    # race_no별로 그룹화
    from collections import defaultdict
    race_entry_map: dict[int, list] = defaultdict(list)
    for entry in target_entries:
        try:
            race_no = int(entry["race_id"].split("_")[-1])
            race_entry_map[race_no].append(entry)
        except (ValueError, IndexError):
            continue

    updated = 0
    for race_no, entries in race_entry_map.items():
        weights = fetch_race_weights(meet_code, race_no, race_date)
        if not weights:
            continue

        # entry_number 기준으로 매칭
        entry_no_map = {e["entry_number"]: e for e in entries if e["entry_number"]}
        for entry_no, wt in weights.items():
            if entry_no not in entry_no_map or wt["horse_weight"] == 0:
                continue
            entry = entry_no_map[entry_no]
            client.table("race_entries").update(
                {"horse_weight": wt["horse_weight"]}
            ).eq("race_id", entry["race_id"]).eq("horse_id", entry["horse_id"]).execute()
            updated += 1

    logger.info(f"{meet_name} {race_date} 마체중 DB 업데이트: {updated}건")
    return updated
