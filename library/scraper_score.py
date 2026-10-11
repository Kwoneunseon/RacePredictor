# -*- coding: utf-8 -*-
"""
경주 결과 배당 스크래퍼
ScoreInfo.do → 확정 배당(단승/연승/복승/쌍승/복연승/삼복승/삼쌍승) 추출
→ race_entries.win_odds 업데이트 + (스키마 확정 후) race_payouts 저장

ScoreInfo.do 응답 테이블의 td 구조 (colspan/rowspan 없음, 원문자 ①~⑳로
출전번호 인코딩):
    0: 경주번호
    1: 단승식   - 1개 그룹, 1두 조합   예) "⑩ 2.6"
    2: 연승식   - 3개 그룹, 1두 조합   예) "⑩1.2 ⑤2.7 ⑪4.9" (1~3착 각자)
    3: 복승식   - 1개 그룹, 2두 조합(순서무관) 예) "⑩⑤ 18.5"
    4: 쌍승식   - 1개 그룹, 2두 조합(순서있음/exacta) 예) "⑩⑤ 27.5"
    5: 복연승식 - 3개 그룹, 2두 조합(순서무관) 예) "⑩⑤5.7 ⑩⑪9.4 ⑤⑪22.5"
    6: 삼복승식 - 1개 그룹, 3두 조합(순서무관/trio) 예) "⑩⑤⑪ 85.4"
    7: 삼쌍승식 - 1개 그룹, 3두 조합(순서있음/trifecta) 예) "⑩⑤⑪ 275.8"
    8: 상세성적 - "완료" 여부
    9: 동영상

사용:
    from library.scraper_score import update_results_to_db
    updated = update_results_to_db(meet_code=3, race_date="2026-06-05")

    from library.scraper_score import update_race_payouts_to_db
    updated = update_race_payouts_to_db(meet_code=3, race_date="2026-06-05")  # race_payouts 테이블 필요
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

# ScoreInfo.do 응답의 td 인덱스 → 베팅 종류 키 매핑 (td 0=경주번호, td 8/9=성적/영상 제외)
BET_TYPE_CELL_INDEX = {
    1: "win",             # 단승식
    2: "place",            # 연승식
    3: "quinella",          # 복승식 (2두, 순서무관)
    4: "exacta",            # 쌍승식 (2두, 순서있음)
    5: "quinella_place",    # 복연승식 (2두, 순서무관, 3조합)
    6: "trio",               # 삼복승식 (3두, 순서무관)
    7: "trifecta",           # 삼쌍승식 (3두, 순서있음)
}

# 원문자(①~⑳) 런 + 숫자(콤마 허용) 그룹을 셀 안에서 전부 찾는 패턴.
# 한 그룹은 "말번호 1~3개 연속 + 그 조합의 배당" 형태이며, 여러 그룹이
# 공백으로 이어 붙어 있을 수 있다(연승식/복연승식).
_ODDS_GROUP_RE = re.compile(r'([①-⑳]+)\s*([\d,]+\.?\d*)')


def _entry_numbers_from_circled(chars: str) -> tuple[int, ...]:
    """'⑩⑤' → (10, 5)"""
    return tuple(ord(c) - 0x245F for c in chars)  # ① = U+2460 → 1


def _parse_odds_groups(text: str) -> list[tuple[tuple[int, ...], float]]:
    """
    셀 텍스트에서 (말번호 튜플, 배당) 쌍을 전부 추출한다. 튜플 순서는
    원문 표기 순서(착순이 유의미한 쌍승/삼쌍승은 이 순서가 곧 착순)를 보존한다.

    '⑩ 2.6'                          → [((10,), 2.6)]
    '⑩1.2 ⑤2.7 ⑪4.9'                 → [((10,), 1.2), ((5,), 2.7), ((11,), 4.9)]
    '⑩⑤ 18.5'                        → [((10, 5), 18.5)]
    '⑩⑤5.7 ⑩⑪9.4 ⑤⑪22.5'            → [((10, 5), 5.7), ((10, 11), 9.4), ((5, 11), 22.5)]
    '⑩⑤⑪ 85.4'                       → [((10, 5, 11), 85.4)]
    """
    groups = []
    for m in _ODDS_GROUP_RE.finditer(text):
        entries = _entry_numbers_from_circled(m.group(1))
        try:
            odds = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if entries:
            groups.append((entries, odds))
    return groups


def _parse_circled(text: str) -> tuple[int | None, float | None]:
    """'② 4.3' → (2, 4.3). 단승식처럼 1두 조합 1개짜리 셀 전용 (하위 호환용)."""
    groups = _parse_odds_groups(text)
    if not groups:
        return None, None
    entries, odds = groups[0]
    if len(entries) != 1:
        return None, None
    return entries[0], odds


def fetch_score_info(meet: int, date: str) -> list[dict]:
    """
    ScoreInfo.do 스크래핑.
    반환: [{
        "race_no": int,
        "win_entry_no": int,          # 하위 호환용 (단승 적중마 번호)
        "win_odds": float,            # 하위 호환용 (단승 배당)
        "payouts": {
            "win":             [{"entry_numbers": [10], "odds": 2.6}],
            "place":           [{"entry_numbers": [10], "odds": 1.2},
                                 {"entry_numbers": [5],  "odds": 2.7},
                                 {"entry_numbers": [11], "odds": 4.9}],
            "quinella":        [{"entry_numbers": [10, 5], "odds": 18.5}],
            "exacta":          [{"entry_numbers": [10, 5], "odds": 27.5}],
            "quinella_place":  [{"entry_numbers": [10, 5],  "odds": 5.7},
                                 {"entry_numbers": [10, 11], "odds": 9.4},
                                 {"entry_numbers": [5, 11],  "odds": 22.5}],
            "trio":            [{"entry_numbers": [10, 5, 11], "odds": 85.4}],
            "trifecta":        [{"entry_numbers": [10, 5, 11], "odds": 275.8}],
        },
    }, ...]
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

        # td 1~7 (단승/연승/복승/쌍승/복연승/삼복승/삼쌍승) 전부 파싱.
        # 셀 인덱스가 부족하거나(방어적) 파싱 결과가 비어도 해당 베팅
        # 종류만 빈 리스트로 두고 나머지는 계속 진행한다 (단승만 있어도 유효).
        payouts: dict[str, list[dict]] = {}
        for cell_idx, bet_type in BET_TYPE_CELL_INDEX.items():
            if cell_idx >= len(cells):
                payouts[bet_type] = []
                continue
            groups = _parse_odds_groups(cells[cell_idx])
            payouts[bet_type] = [
                {"entry_numbers": list(entries), "odds": odds}
                for entries, odds in groups
            ]

        results.append({
            "race_no":      race_no,
            "win_entry_no": win_entry_no,
            "win_odds":     win_odds,
            "payouts":      payouts,
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


def update_race_payouts_to_db(meet_code: int, race_date: str) -> int:
    """
    ScoreInfo 스크래핑 결과(payouts: 단승/연승/복승/쌍승/복연승/삼복승/삼쌍승)를
    race_payouts 테이블에 upsert.

    주의: race_payouts 테이블은 아직 스키마에 없다(db-expert 마이그레이션
    선행 필요, `entry_numbers INT[]` + UNIQUE(race_id, bet_type, entry_numbers)
    설계 제안 참고). 테이블이 없으면 매 경주 upsert가 예외로 실패하고 0을
    반환하며 로그에 원인이 남는다 — 스키마 확정 전에도 안전하게 호출 가능.

    반환: upsert된 (경주 x 베팅종류 x 조합) 행 수
    """
    from Utils.database import get_client

    score_list = fetch_score_info(meet_code, race_date)
    if not score_list:
        return 0

    client = get_client()
    meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))

    rows = []
    for item in score_list:
        race_id = f"{race_date}_{meet_name}_{item['race_no']}"
        for bet_type, groups in item["payouts"].items():
            for g in groups:
                rows.append({
                    "race_id":       race_id,
                    "bet_type":      bet_type,
                    "entry_numbers": g["entry_numbers"],
                    "odds":          g["odds"],
                })

    if not rows:
        logger.warning(f"race_payouts 업로드할 행 없음 ({meet_name} {race_date})")
        return 0

    try:
        res = client.table("race_payouts") \
            .upsert(rows, on_conflict="race_id,bet_type,entry_numbers") \
            .execute()
        updated = len(res.data) if res.data else 0
    except Exception as e:
        logger.error(f"race_payouts upsert 실패 ({meet_name} {race_date}, {len(rows)}행 시도): {e}")
        return 0

    logger.info(f"race_payouts 업데이트 완료: {meet_name} {race_date} → {updated}/{len(rows)}행")
    return updated


def update_final_ranks(race_date: str, meet_code: int) -> int:
    """
    공용 경주 결과 API(raceResult_3, meet 지정)에서 해당 날짜 결과를 재조회해
    final_rank / win_odds 업데이트. 서울도 이 API를 쓴다 - SeoulRace_1에는
    말 개별 단승 배당이 없다. 응답에서 요청한 날짜와 정확히 일치하는 기록만
    반영한다.
    반환: 업데이트된 race_entries 건수
    """
    from library.collector_races import fetch_race_data_all, parse_and_normalize_race_data
    from Utils.database import get_client

    meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))
    date_str = race_date.replace("-", "")
    try:
        api_data, _ = fetch_race_data_all(
            page=1, per_page=1000, meet_code=meet_code,
            start_date=date_str, end_date=date_str, common_api=True,
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
        if entry.get("finish_time"):
            update_data["finish_time"] = entry["finish_time"]
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
    win_odds 갱신에 더해 race_payouts(단승/연승/복승/쌍승/복연승/삼복승/삼쌍승)도
    같이 저장한다 - 이전에는 update_race_payouts_to_db가 구현만 되어 있고
    여기서 호출되지 않아, 실제로는 한 번(2026-06-05, meet=3) 수동 실행분밖에
    쌓이지 않았다. race_payouts 결과는 반환값 counts에는 포함하지 않는다
    (main.py가 이 반환값으로 win_odds 갱신 건수 합계를 내는 기존 계약을
    유지하기 위함) - 저장 건수는 update_race_payouts_to_db 내부 로그로 확인.
    반환: {"서울": int, "제주": int, "부산경남": int}
    """
    counts = {}
    for meet_code, name in MEET_NAME_MAP.items():
        cnt = update_results_to_db(meet_code, race_date)
        counts[name] = cnt
        update_race_payouts_to_db(meet_code, race_date)
    return counts


if __name__ == "__main__":
    import sys
    from datetime import datetime

    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    print(f"결과 업데이트: {date}")
    result = update_all_meets(date)
    for name, cnt in result.items():
        print(f"  {name}: {cnt}경주 업데이트")
