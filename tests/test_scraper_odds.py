"""scraper_odds.py 파싱 로직 단위 테스트 (HTTP 요청 없음)"""
import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup


def _make_soup_with_table(rows: list[list[str]]) -> list:
    """rows를 가진 단일 테이블을 포함한 BeautifulSoup 테이블 목록 반환."""
    html_rows = ""
    for cells in rows:
        tds = "".join(f"<td>{c}</td>" for c in cells)
        html_rows += f"<tr>{tds}</tr>"
    html = f"<html><body><table>{html_rows}</table></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    return soup.find_all("table")


def test_parse_odds_table_basic():
    """번호|말명|배당 구조에서 정상 파싱."""
    from library.scraper_odds import _parse_odds_table
    tables = _make_soup_with_table([
        ["1", "테스트마A", "5.2", "기타"],
        ["2", "테스트마B", "12.0", "기타"],
        ["3", "테스트마C", "3.1", "기타"],
        ["4", "테스트마D", "8.5", "기타"],
    ])
    result = _parse_odds_table(tables)
    assert result == {1: 5.2, 2: 12.0, 3: 3.1, 4: 8.5}


def test_parse_odds_table_skips_invalid_rows():
    """출전번호가 없거나 배당이 없는 행은 무시."""
    from library.scraper_odds import _parse_odds_table
    tables = _make_soup_with_table([
        ["번호", "말명", "단승배당"],       # 헤더 행 (숫자 아님)
        ["1", "말A", "7.0", "기타"],
        ["2", "말B", "-", "기타"],          # 배당 없음
        ["3", "말C", "4.5", "기타"],
        ["4", "말D", "9.2", "기타"],
    ])
    result = _parse_odds_table(tables)
    assert 1 in result
    assert 2 not in result  # "-" → 제외
    assert 3 in result
    assert 4 in result


def test_parse_odds_table_returns_empty_for_short_table():
    """3마리 미만이면 빈 dict 반환."""
    from library.scraper_odds import _parse_odds_table
    tables = _make_soup_with_table([
        ["1", "말A", "7.0"],
        ["2", "말B", "4.0"],
    ])
    result = _parse_odds_table(tables)
    assert result == {}


def test_try_parse_odds_valid():
    """유효한 배당값 파싱."""
    from library.scraper_odds import _try_parse_odds
    cells = ["1", "말명", "7.5", "기타"]
    assert _try_parse_odds(cells, [2]) == pytest.approx(7.5)


def test_try_parse_odds_out_of_range():
    """유효 범위(1.0~999.0) 밖이면 None 반환."""
    from library.scraper_odds import _try_parse_odds
    cells = ["1", "말명", "0.5"]   # 1.0 미만
    assert _try_parse_odds(cells, [2]) is None


def test_try_parse_odds_fallback_index():
    """첫 번째 후보가 유효하지 않으면 다음 후보에서 찾는다."""
    from library.scraper_odds import _try_parse_odds
    cells = ["1", "말명", "-", "8.0"]
    result = _try_parse_odds(cells, [2, 3])
    assert result == pytest.approx(8.0)


def test_fetch_race_win_odds_returns_empty_on_error():
    """HTTP 요청 실패 시 빈 dict 반환."""
    import requests
    from library.scraper_odds import fetch_race_win_odds
    with patch("library.scraper_odds.requests.post", side_effect=requests.ConnectionError("연결 실패")):
        result = fetch_race_win_odds(1, 1, "2026-06-07")
    assert result == {}


def test_fetch_race_win_odds_returns_empty_on_bad_status():
    """HTTP 404 응답 시 빈 dict 반환."""
    from library.scraper_odds import fetch_race_win_odds
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("library.scraper_odds.requests.post", return_value=mock_resp):
        result = fetch_race_win_odds(1, 1, "2026-06-07")
    assert result == {}
