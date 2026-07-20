"""scraper_score.py 단위 테스트 (HTTP/DB 없음)"""
import pytest
from unittest.mock import patch, MagicMock


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
