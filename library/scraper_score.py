# -*- coding: utf-8 -*-
"""
경주 결과 배당 스크래퍼
ScoreInfo.do → 단승 적중 말 번호 + 배당 추출 → race_entries.win_odds 업데이트

사용:
    from library.scraper_score import update_results_to_db
    updated = update_results_to_db(meet_code=3, race_date="2026-06-05")
"""
import re
import requests
from bs4 import BeautifulSoup
from Utils.logger import get_logger

logger = get_logger(__name__)

SCORE_URL = "https://race.kra.co.kr/raceFastreport/ScoreInfo.do"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://race.kra.co.kr/raceFastreport/ChuljumaWeightList.do",
}
MEET_NAME_MAP = {1: "서울", 2: "제주", 3: "부산경남"}

# 원문자 ①~⑳ → 정수 변환
def _parse_circled(text: str) -> tuple[int | None, float | None]:
    """'② 4.3' → (2, 4.3)"""
    for char in text:
        code = ord(char)
        if 0x2460 <= code <= 0x2473:          # ① ~ ⑳
            entry_no = code - 0x245F           # ① = 1
            nums = re.findall(r'\d+\.?\d*', text[text.index(char) + 1:])
            odds = float(nums[0]) if nums else None
            return entry_no, odds
    return None, None


def fetch_score_info(meet: int, date: str) -> list[dict]:
    """
    ScoreInfo.do 스크래핑.
    반환: [{"race_no": int, "win_entry_no": int, "win_odds": float}, ...]
    완료되지 않은 경주('완료' 미표시)는 제외.
    """
    date_str = date.replace("-", "")
    try:
        resp = requests.post(
            SCORE_URL,
            data={"meet": str(meet), "rcNo": "1", "date": date_str, "Act": "03", "Sub": "3"},
            headers=HEADERS,
            timeout=15,
        )
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding="euc-kr")
    except Exception as e:
        logger.error(f"ScoreInfo 요청 실패 (meet={meet}, date={date}): {e}")
        return []

    tables = soup.find_all("table")
    if not tables:
        logger.warning(f"ScoreInfo 테이블 없음 (meet={meet}, date={date})")
        return []

    results = []
    for row in tables[0].find_all("tr")[1:]:  # 헤더 행 제외
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 9:
            continue

        # 완료 여부 확인 (상세성적 열 = "완료")
        if cells[8] != "완료":
            continue

        try:
            race_no = int(cells[0])
        except ValueError:
            continue

        win_entry_no, win_odds = _parse_circled(cells[1])  # 단승식 셀
        if win_entry_no is None or win_odds is None:
            continue

        results.append({
            "race_no":      race_no,
            "win_entry_no": win_entry_no,
            "win_odds":     win_odds,
        })

    meet_name = MEET_NAME_MAP.get(meet, str(meet))
    logger.info(f"ScoreInfo 수집: {meet_name} {date} → {len(results)}경주 완료")
    return results


def update_results_to_db(meet_code: int, race_date: str) -> int:
    """
    ScoreInfo 스크래핑 결과로 race_entries의 win_odds 업데이트.
    - 단승 적중 말(entry_number 기준)의 win_odds를 실제 배당으로 갱신
    반환: 업데이트된 건수
    """
    from Utils.database import get_client

    score_list = fetch_score_info(meet_code, race_date)
    if not score_list:
        return 0

    client = get_client()
    meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))
    updated = 0

    for item in score_list:
        race_no      = item["race_no"]
        win_entry_no = item["win_entry_no"]
        win_odds     = item["win_odds"]

        # race_id 형식: YYYY-MM-DD_경마장명_경주번호
        race_id = f"{race_date}_{meet_name}_{race_no}"

        try:
            res = client.table("race_entries") \
                .update({"win_odds": win_odds}) \
                .eq("race_id", race_id) \
                .eq("entry_number", win_entry_no) \
                .execute()

            if res.data:
                updated += 1
                logger.debug(f"  win_odds 업데이트: {race_id} entry={win_entry_no} → {win_odds}배")
            else:
                logger.warning(f"  매칭 없음: {race_id} entry={win_entry_no}")
        except Exception as e:
            logger.error(f"  DB 업데이트 실패 ({race_id} entry={win_entry_no}): {e}")

    logger.info(f"win_odds 업데이트 완료: {meet_name} {race_date} → {updated}/{len(score_list)}경주")
    return updated


def update_final_ranks(race_date: str, meet_code: int) -> int:
    """
    경마장별 공공데이터 API(SeoulRace_1 / raceResult_2 / raceResult_3)에서
    결과를 재조회해 final_rank / win_odds 업데이트.
    제주·부산경남 API는 월(rc_month) 단위로만 조회되므로, 응답에서 요청한 날짜와
    정확히 일치하는 기록만 걸러서 반영한다.
    반환: 업데이트된 race_entries 건수
    """
    from library.collector_races import fetch_race_data_all, parse_and_normalize_race_data
    from Utils.database import get_client

    meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))
    date_str = race_date.replace("-", "")
    try:
        api_data, _ = fetch_race_data_all(
            page=1, per_page=1000, meet_code=meet_code,
            start_date=date_str, end_date=date_str,
        )
    except Exception as e:
        logger.error(f"{meet_name} API 요청 실패 ({race_date}): {e}")
        return 0

    if not api_data:
        logger.warning(f"{meet_name} API 응답 없음 ({race_date})")
        return 0

    parsed = parse_and_normalize_race_data(api_data)
    entries = [
        e for e in parsed.get("race_entries", [])
        if e.get("final_rank") and e.get("race_date") == race_date
    ]

    if not entries:
        logger.warning(f"{meet_name} final_rank 데이터 없음 ({race_date}) - 경마일이 아니거나 아직 결과 미공개")
        return 0

    client = get_client()
    updated = 0
    for entry in entries:
        update_data: dict = {"final_rank": entry["final_rank"]}
        if entry.get("win_odds"):
            update_data["win_odds"] = entry["win_odds"]
        try:
            res = (
                client.table("race_entries")
                .update(update_data)
                .eq("race_id", entry["race_id"])
                .eq("horse_id", entry["horse_id"])
                .execute()
            )
            if res.data:
                updated += 1
        except Exception as e:
            logger.error(f"{meet_name} final_rank 업데이트 실패 ({entry.get('race_id')}, {entry.get('horse_id')}): {e}")

    logger.info(f"{meet_name} final_rank 업데이트: {race_date} → {updated}건")
    return updated


def update_final_ranks_seoul(race_date: str) -> int:
    """서울 전용 래퍼 (하위 호환용)."""
    return update_final_ranks(race_date, meet_code=1)


def update_all_final_ranks(race_date: str) -> dict:
    """전체 경마장(서울·제주·부산경남) final_rank 일괄 업데이트.
    반환: {"서울": int, "제주": int, "부산경남": int}
    """
    counts = {}
    for meet_code, name in MEET_NAME_MAP.items():
        counts[name] = update_final_ranks(race_date, meet_code)
    return counts


def update_all_meets(race_date: str) -> dict:
    """
    전체 경마장(서울·제주·부산) 결과 일괄 업데이트.
    반환: {"서울": int, "제주": int, "부산경남": int}
    """
    counts = {}
    for meet_code, name in MEET_NAME_MAP.items():
        cnt = update_results_to_db(meet_code, race_date)
        counts[name] = cnt
    return counts


if __name__ == "__main__":
    import sys
    from datetime import datetime

    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    print(f"결과 업데이트: {date}")
    result = update_all_meets(date)
    for name, cnt in result.items():
        print(f"  {name}: {cnt}경주 업데이트")
