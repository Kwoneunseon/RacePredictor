"""scraper_score.py 단위 테스트 (HTTP/DB 없음)"""
import pytest
from unittest.mock import patch, MagicMock

from library.scraper_score import (
    _parse_circled,
    _parse_odds_groups,
    _entry_numbers_from_circled,
    BET_TYPE_CELL_INDEX,
)


# ──────────────────────────────────────────────
# _entry_numbers_from_circled / _parse_odds_groups / _parse_circled
# (ScoreInfo.do 응답 td 파싱 - 단승/연승/복승/쌍승/복연승/삼복승/삼쌍승)
# ──────────────────────────────────────────────

def test_entry_numbers_from_circled_handles_full_range():
    assert _entry_numbers_from_circled("①") == (1,)
    assert _entry_numbers_from_circled("⑩") == (10,)
    assert _entry_numbers_from_circled("⑳") == (20,)
    assert _entry_numbers_from_circled("⑩⑤⑪") == (10, 5, 11)


def test_parse_odds_groups_win_single_pair():
    """단승식: 1개 그룹, 1두 조합."""
    assert _parse_odds_groups("⑩ 2.6") == [((10,), 2.6)]


def test_parse_odds_groups_place_three_single_entry_groups():
    """연승식: 3개 그룹, 각각 1두 조합 (1~3착 각자의 연승 배당)."""
    result = _parse_odds_groups("⑩1.2 ⑤2.7 ⑪4.9")
    assert result == [((10,), 1.2), ((5,), 2.7), ((11,), 4.9)]


def test_parse_odds_groups_quinella_two_entry_combo():
    """복승식: 1개 그룹, 2두 조합(순서무관)."""
    assert _parse_odds_groups("⑩⑤ 18.5") == [((10, 5), 18.5)]


def test_parse_odds_groups_exacta_two_entry_combo():
    """쌍승식: 1개 그룹, 2두 조합(순서있음) - 원문 표기 순서(착순) 보존."""
    assert _parse_odds_groups("⑩⑤ 27.5") == [((10, 5), 27.5)]


def test_parse_odds_groups_quinella_place_three_pair_groups():
    """복연승식: 3개 그룹, 각각 2두 조합(1~3착 중 2두 조합 3가지)."""
    result = _parse_odds_groups("⑩⑤5.7 ⑩⑪9.4 ⑤⑪22.5")
    assert result == [
        ((10, 5), 5.7),
        ((10, 11), 9.4),
        ((5, 11), 22.5),
    ]


def test_parse_odds_groups_trio_three_entry_combo():
    """삼복승식: 1개 그룹, 3두 조합(순서무관)."""
    assert _parse_odds_groups("⑩⑤⑪ 85.4") == [((10, 5, 11), 85.4)]


def test_parse_odds_groups_trifecta_three_entry_combo():
    """삼쌍승식: 1개 그룹, 3두 조합(순서있음)."""
    assert _parse_odds_groups("⑩⑤⑪ 275.8") == [((10, 5, 11), 275.8)]


def test_parse_odds_groups_handles_comma_thousands_separator():
    """고배당(삼쌍승 등)은 천단위 콤마가 붙을 수 있음."""
    assert _parse_odds_groups("⑩⑤⑪ 1,275.8") == [((10, 5, 11), 1275.8)]


def test_parse_odds_groups_empty_or_dash_returns_empty_list():
    assert _parse_odds_groups("") == []
    assert _parse_odds_groups("-") == []


def test_parse_circled_backward_compatible_single_entry():
    """단승식처럼 1두 조합 1개짜리 셀에서는 기존 (entry_no, odds) 튜플을 그대로 반환."""
    assert _parse_circled("② 4.3") == (2, 4.3)


def test_parse_circled_rejects_multi_entry_combo():
    """복승식처럼 2두 이상 조합인 셀은 _parse_circled 용도가 아니므로 (None, None)."""
    assert _parse_circled("⑩⑤ 18.5") == (None, None)


def test_parse_circled_no_match_returns_none_none():
    assert _parse_circled("") == (None, None)


def test_bet_type_cell_index_covers_seven_bet_types():
    """td 1~7 (단승/연승/복승/쌍승/복연승/삼복승/삼쌍승) 전부 매핑돼 있어야 한다."""
    assert BET_TYPE_CELL_INDEX == {
        1: "win",
        2: "place",
        3: "quinella",
        4: "exacta",
        5: "quinella_place",
        6: "trio",
        7: "trifecta",
    }


# ──────────────────────────────────────────────
# fetch_score_info (payouts 전체 파싱)
# ──────────────────────────────────────────────

def _make_score_info_html(cells: list[str]) -> bytes:
    row = "".join(f"<td>{c}</td>" for c in cells)
    html = f"<table><tr><th>t</th></tr><tr>{row}</tr></table>"
    return html.encode("euc-kr")


def test_fetch_score_info_parses_all_seven_bet_types():
    html = _make_score_info_html([
        "1",                                  # 경주번호
        "⑩ 2.6",                              # 단승식
        "⑩1.2 ⑤2.7 ⑪4.9",                     # 연승식
        "⑩⑤ 18.5",                            # 복승식
        "⑩⑤ 27.5",                            # 쌍승식
        "⑩⑤5.7 ⑩⑪9.4 ⑤⑪22.5",                # 복연승식
        "⑩⑤⑪ 85.4",                           # 삼복승식
        "⑩⑤⑪ 275.8",                          # 삼쌍승식
        "완료",                                # 상세성적
        "영상보기",                            # 동영상
    ])
    mock_resp = MagicMock()
    mock_resp.content = html

    with patch("requests.post", return_value=mock_resp):
        from library.scraper_score import fetch_score_info
        result = fetch_score_info(meet=1, date="2026-07-26")

    assert len(result) == 1
    item = result[0]
    # 하위 호환 필드
    assert item["race_no"] == 1
    assert item["win_entry_no"] == 10
    assert item["win_odds"] == 2.6
    # 신규 payouts
    payouts = item["payouts"]
    assert payouts["win"] == [{"entry_numbers": [10], "odds": 2.6}]
    assert payouts["place"] == [
        {"entry_numbers": [10], "odds": 1.2},
        {"entry_numbers": [5], "odds": 2.7},
        {"entry_numbers": [11], "odds": 4.9},
    ]
    assert payouts["quinella"] == [{"entry_numbers": [10, 5], "odds": 18.5}]
    assert payouts["exacta"] == [{"entry_numbers": [10, 5], "odds": 27.5}]
    assert payouts["quinella_place"] == [
        {"entry_numbers": [10, 5], "odds": 5.7},
        {"entry_numbers": [10, 11], "odds": 9.4},
        {"entry_numbers": [5, 11], "odds": 22.5},
    ]
    assert payouts["trio"] == [{"entry_numbers": [10, 5, 11], "odds": 85.4}]
    assert payouts["trifecta"] == [{"entry_numbers": [10, 5, 11], "odds": 275.8}]


def test_fetch_score_info_excludes_incomplete_races():
    html = _make_score_info_html([
        "1", "⑩ 2.6", "⑩1.2 ⑤2.7 ⑪4.9", "⑩⑤ 18.5", "⑩⑤ 27.5",
        "⑩⑤5.7 ⑩⑪9.4 ⑤⑪22.5", "⑩⑤⑪ 85.4", "⑩⑤⑪ 275.8",
        "",  # 상세성적 미표시 → 미완료 경주
        "",
    ])
    mock_resp = MagicMock()
    mock_resp.content = html

    with patch("requests.post", return_value=mock_resp):
        from library.scraper_score import fetch_score_info
        result = fetch_score_info(meet=1, date="2026-07-26")

    assert result == []


# ──────────────────────────────────────────────
# update_race_payouts_to_db
# ──────────────────────────────────────────────

def test_update_race_payouts_to_db_upserts_all_bet_type_rows():
    """1경주 x 7베팅종류(그룹수 1+3+1+1+3+1+1) = 총 11행이 upsert 요청에 포함돼야 한다."""
    fake_score_list = [{
        "race_no": 1,
        "win_entry_no": 10,
        "win_odds": 2.6,
        "payouts": {
            "win":             [{"entry_numbers": [10], "odds": 2.6}],
            "place":           [{"entry_numbers": [10], "odds": 1.2},
                                 {"entry_numbers": [5], "odds": 2.7},
                                 {"entry_numbers": [11], "odds": 4.9}],
            "quinella":        [{"entry_numbers": [10, 5], "odds": 18.5}],
            "exacta":          [{"entry_numbers": [10, 5], "odds": 27.5}],
            "quinella_place":  [{"entry_numbers": [10, 5], "odds": 5.7},
                                 {"entry_numbers": [10, 11], "odds": 9.4},
                                 {"entry_numbers": [5, 11], "odds": 22.5}],
            "trio":            [{"entry_numbers": [10, 5, 11], "odds": 85.4}],
            "trifecta":        [{"entry_numbers": [10, 5, 11], "odds": 275.8}],
        },
    }]

    mock_client = MagicMock()
    upserted_rows = [{"id": i} for i in range(11)]
    mock_client.table.return_value.upsert.return_value.execute.return_value.data = upserted_rows

    with patch("library.scraper_score.fetch_score_info", return_value=fake_score_list), \
         patch("Utils.database.get_client", return_value=mock_client):
        from library.scraper_score import update_race_payouts_to_db
        result = update_race_payouts_to_db(meet_code=1, race_date="2026-07-26")

    assert result == 11
    call_args = mock_client.table.return_value.upsert.call_args
    rows_arg = call_args.args[0]
    assert len(rows_arg) == 11
    assert {"race_id": "2026-07-26_서울_1", "bet_type": "trifecta",
            "entry_numbers": [10, 5, 11], "odds": 275.8} in rows_arg
    assert call_args.kwargs.get("on_conflict") == "race_id,bet_type,entry_numbers"


def test_update_race_payouts_to_db_returns_zero_when_no_score_list():
    with patch("library.scraper_score.fetch_score_info", return_value=[]):
        from library.scraper_score import update_race_payouts_to_db
        result = update_race_payouts_to_db(meet_code=1, race_date="2026-07-26")

    assert result == 0


def test_update_race_payouts_to_db_returns_zero_on_upsert_exception():
    fake_score_list = [{
        "race_no": 1, "win_entry_no": 10, "win_odds": 2.6,
        "payouts": {"win": [{"entry_numbers": [10], "odds": 2.6}]},
    }]
    mock_client = MagicMock()
    mock_client.table.return_value.upsert.return_value.execute.side_effect = Exception("테이블 없음")

    with patch("library.scraper_score.fetch_score_info", return_value=fake_score_list), \
         patch("Utils.database.get_client", return_value=mock_client):
        from library.scraper_score import update_race_payouts_to_db
        result = update_race_payouts_to_db(meet_code=1, race_date="2026-07-26")

    assert result == 0


# ──────────────────────────────────────────────
# update_final_ranks_seoul
# ──────────────────────────────────────────────

def _make_api_response(entries: list[dict]) -> dict:
    """SeoulRace_1 API 응답 형식 모킹."""
    items = []
    for e in entries:
        items.append({
            "rcDate": e.get("race_date", "20260606"),
            "meet":   "서울",
            "rcNo":   str(e.get("race_no", 1)),
            "hrno":   e.get("horse_id", "H1"),
            "hrName": e.get("horse_name", "테스트마"),
            "jkNo":   e.get("jk_no", "J1"),
            "jkName": "기수",
            "prtr":   e.get("trainer_id", "T1"),
            "prtrName": "조교사",
            "prow":   "O1",
            "prowName": "마주",
            "rcOrd":  str(e.get("final_rank", 1)),
            "winOdds": str(e.get("win_odds", 5.0)),
            "rcChul": str(e.get("entry_number", 1)),
            "wgHr":   "480(0)",
            "rcDist": "1400",
            "rcGrade": "일반",
            "track":  "양호",
            "weath":  "맑음",
            "rcVtdusu": "10",
            "rcTime": "86.5",
        })
    return {"response": {"body": {"items": {"item": items}}}}


def test_update_final_ranks_seoul_success():
    """API 응답 정상일 때 final_rank가 업데이트된다."""
    api_resp = _make_api_response([
        {"race_no": 1, "horse_id": "H1", "final_rank": 1, "win_odds": 4.5},
        {"race_no": 1, "horse_id": "H2", "final_rank": 2, "win_odds": 0},
    ])

    mock_client = MagicMock()
    mock_update_chain = mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value
    mock_update_chain.execute.return_value.data = [{"race_id": "x"}]

    with patch("library.collector_races.fetch_race_data_all", return_value=(api_resp, 1)), \
         patch("Utils.database.get_client", return_value=mock_client):
        from library.scraper_score import update_final_ranks_seoul
        result = update_final_ranks_seoul("2026-06-06")

    assert result == 2


def test_update_final_ranks_seoul_api_failure():
    """API 요청 실패 시 0 반환."""
    with patch("library.collector_races.fetch_race_data_all", return_value=(None, 1)):
        from library.scraper_score import update_final_ranks_seoul
        result = update_final_ranks_seoul("2026-06-06")

    assert result == 0


def test_update_final_ranks_seoul_no_rank_data():
    """API 응답에 final_rank가 없으면 0 반환."""
    api_resp = _make_api_response([
        {"race_no": 1, "horse_id": "H1", "final_rank": 0},
    ])
    with patch("library.collector_races.fetch_race_data_all", return_value=(api_resp, 1)):
        from library.scraper_score import update_final_ranks_seoul
        result = update_final_ranks_seoul("2026-06-06")

    assert result == 0


def test_update_final_ranks_seoul_exception():
    """API 호출 중 예외 발생 시 0 반환."""
    with patch("library.collector_races.fetch_race_data_all", side_effect=Exception("연결 오류")):
        from library.scraper_score import update_final_ranks_seoul
        result = update_final_ranks_seoul("2026-06-06")

    assert result == 0


# ──────────────────────────────────────────────
# update_final_ranks (제주·부산 포함 범용 버전)
# ──────────────────────────────────────────────

def _make_jeju_api_response(entries: list[dict]) -> dict:
    """raceResult_2(제주) API 응답 형식 모킹. rcDate가 서로 다른 여러 날짜(월 단위 응답)를 담을 수 있음."""
    items = []
    for e in entries:
        items.append({
            "rcDate": e.get("race_date", "20260606"),
            "meet":   "제주",
            "rcNo":   str(e.get("race_no", 1)),
            "hrno":   e.get("horse_id", "H1"),
            "hrName": e.get("horse_name", "테스트마"),
            "jkNo":   e.get("jk_no", "J1"),
            "jkName": "기수",
            "prtr":   e.get("trainer_id", "T1"),
            "prtrName": "조교사",
            "prow":   "O1",
            "prowName": "마주",
            "rcOrd":  str(e.get("final_rank", 1)),
            "winOdds": str(e.get("win_odds", 5.0)),
            "rcChul": str(e.get("entry_number", 1)),
            "wgHr":   "480(0)",
            "rcDist": "1400",
            "rcGrade": "일반",
            "track":  "양호",
            "weath":  "맑음",
            "rcVtdusu": "10",
            "rcTime": "86.5",
        })
    return {"response": {"body": {"items": {"item": items}}}}


def test_update_final_ranks_works_for_jeju_meet_code():
    """meet_code=2(제주)도 동일한 방식으로 final_rank가 업데이트된다."""
    api_resp = _make_jeju_api_response([
        {"race_date": "20260710", "race_no": 1, "horse_id": "H1", "final_rank": 1, "win_odds": 4.5},
        {"race_date": "20260710", "race_no": 1, "horse_id": "H2", "final_rank": 2, "win_odds": 0},
    ])

    mock_client = MagicMock()
    mock_update_chain = mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value
    mock_update_chain.execute.return_value.data = [{"race_id": "x"}]

    with patch("library.collector_races.fetch_race_data_all", return_value=(api_resp, 1)), \
         patch("Utils.database.get_client", return_value=mock_client):
        from library.scraper_score import update_final_ranks
        result = update_final_ranks("2026-07-10", meet_code=2)

    assert result == 2


def test_update_final_ranks_filters_out_other_dates_in_monthly_response():
    """제주·부산 API는 월 단위 응답이므로, 요청한 날짜가 아닌 기록은 업데이트에서 제외돼야 한다."""
    api_resp = _make_jeju_api_response([
        {"race_date": "20260710", "race_no": 1, "horse_id": "H1", "final_rank": 1},
        {"race_date": "20260703", "race_no": 1, "horse_id": "H2", "final_rank": 1},  # 다른 날짜 → 제외
    ])

    mock_client = MagicMock()
    mock_update_chain = mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value
    mock_update_chain.execute.return_value.data = [{"race_id": "x"}]

    with patch("library.collector_races.fetch_race_data_all", return_value=(api_resp, 1)), \
         patch("Utils.database.get_client", return_value=mock_client):
        from library.scraper_score import update_final_ranks
        result = update_final_ranks("2026-07-10", meet_code=2)

    assert result == 1


def test_update_final_ranks_seoul_delegates_to_meet_code_1():
    """update_final_ranks_seoul은 update_final_ranks(meet_code=1)의 래퍼여야 한다."""
    with patch("library.scraper_score.update_final_ranks", return_value=42) as mock_fn:
        from library.scraper_score import update_final_ranks_seoul
        result = update_final_ranks_seoul("2026-06-06")

    mock_fn.assert_called_once_with("2026-06-06", meet_code=1)
    assert result == 42


def test_update_all_final_ranks_covers_all_three_meets():
    with patch("library.scraper_score.update_final_ranks", side_effect=[5, 3, 2]) as mock_fn:
        from library.scraper_score import update_all_final_ranks
        result = update_all_final_ranks("2026-07-10")

    assert result == {"서울": 5, "제주": 3, "부산경남": 2}
    assert mock_fn.call_count == 3
