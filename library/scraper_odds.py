# -*- coding: utf-8 -*-
"""
경주 현재 단승배당 스크래퍼
KRA 경마속보 배당 페이지 → 말별 현재 단승배당 반환

사용:
    from library.scraper_odds import fetch_race_win_odds
    odds = fetch_race_win_odds(meet_code=1, race_no=3, race_date="2026-06-07")
    # {1: 5.2, 2: 12.0, 3: 3.1, ...}  (출전번호 → 배당)
"""
import requests
from bs4 import BeautifulSoup
from Utils.logger import get_logger

logger = get_logger(__name__)

# KRA 경마속보 배당 페이지 후보 URL (경마일 실시간 갱신)
# 실제 URL 확인 방법: 브라우저에서 race.kra.co.kr/raceFastreport/ 접속 후 배당 탭 클릭
ODDS_URL = "https://race.kra.co.kr/raceFastreport/OddsInfo.do"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://race.kra.co.kr/raceFastreport/ChuljumaWeightList.do",
}
MEET_NAME_MAP = {1: "서울", 2: "제주", 3: "부산경남"}


def fetch_race_win_odds(meet_code: int, race_no: int, race_date: str) -> dict[int, float]:
    """
    KRA 경마속보에서 특정 경주의 현재 단승배당을 가져옴.

    반환: {출전번호: 단승배당}
    - 경마 시작 전/배당 미발표 시: 빈 dict 반환
    - 스크래핑 실패 시: 빈 dict 반환 (feature_builder에서 prev_win_odds로 대체)
    """
    date_str = race_date.replace("-", "")
    post_data = {
        "meet": str(meet_code),
        "rcNo": str(race_no),
        "date": date_str,
        "Act": "03",
        "Sub": "2",
    }

    try:
        resp = requests.post(ODDS_URL, data=post_data, headers=HEADERS, timeout=15)
        resp.encoding = "euc-kr"

        if resp.status_code != 200:
            logger.debug(f"배당 페이지 응답 오류: {resp.status_code} (meet={meet_code}, rcNo={race_no})")
            return {}

        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")

        result = _parse_odds_table(tables)

        if result:
            meet_name = MEET_NAME_MAP.get(meet_code, str(meet_code))
            logger.info(f"단승배당 수집: {meet_name} {race_no}R → {len(result)}마리")
        else:
            logger.debug(f"배당 데이터 없음 (meet={meet_code}, rcNo={race_no}, date={date_str})")

        return result

    except Exception as e:
        logger.debug(f"배당 수집 실패 (meet={meet_code}, rcNo={race_no}): {e}")
        return {}


def _parse_odds_table(tables: list) -> dict[int, float]:
    """
    HTML 테이블 목록에서 출전번호-배당 쌍을 파싱.
    KRA 배당 페이지 구조: 번호 | 말명 | 단승배당 | ...
    """
    for table in tables:
        rows = table.find_all("tr")
        result = {}
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 3:
                continue
            try:
                entry_no = int(cells[0])
            except (ValueError, IndexError):
                continue
            # 단승배당: 보통 2번째 또는 3번째 컬럼에 존재
            odds = _try_parse_odds(cells, candidate_indices=[2, 3, 1])
            if odds is not None and odds > 0:
                result[entry_no] = odds

        if len(result) >= 3:  # 3마리 이상 배당이 있으면 유효한 테이블로 판단
            return result

    return {}


def _try_parse_odds(cells: list, candidate_indices: list) -> float | None:
    """여러 컬럼 후보에서 유효한 배당값을 찾아 반환."""
    for idx in candidate_indices:
        if idx >= len(cells):
            continue
        s = cells[idx].replace(",", "").strip()
        if not s or s in ("-", "배당", "단승", ""):
            continue
        try:
            val = float(s)
            if 1.0 <= val <= 999.0:  # 유효 배당 범위
                return val
        except ValueError:
            continue
    return None


def fetch_all_odds_for_date(meet_code: int, race_date: str, max_races: int = 12) -> dict[int, dict[int, float]]:
    """
    특정 경마장·날짜의 전 경주 단승배당 수집.

    반환: {race_no: {출전번호: 단승배당}}
    """
    all_odds: dict[int, dict[int, float]] = {}
    for race_no in range(1, max_races + 1):
        odds = fetch_race_win_odds(meet_code, race_no, race_date)
        if not odds:
            # 배당 없는 경주가 나오면 그 이후 경주도 없다고 판단
            if race_no > 3:
                break
            continue
        all_odds[race_no] = odds
    return all_odds
