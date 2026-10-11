import pytest
from unittest.mock import patch, MagicMock


def test_fetch_horse_data_returns_none_on_error(mocker):
    mocker.patch("requests.get", side_effect=Exception("network error"))
    from library.collector_horse import fetch_horse_data
    result, page = fetch_horse_data(page=1)
    assert result is None
    assert page == 1


def test_save_horse_batch_empty_returns_zero():
    from library.collector_horse import save_to_supabase_batch
    result = save_to_supabase_batch([])
    assert result == 0


def test_save_jockey_batch_empty_returns_zero():
    from library.collector_jockeys import save_to_supabase_batch
    result = save_to_supabase_batch([])
    assert result == 0


def test_parse_jockey_data_strips_whitespace_and_converts_types():
    """collector_jockeys.py가 library.parsing의 safe_str/safe_int로
    전환된 뒤에도 공백 제거·형변환 계약이 유지되는지 고정하는 회귀 테스트."""
    from library.collector_jockeys import parse_jockey_data

    api_response = {
        "response": {
            "body": {
                "items": {
                    "item": {
                        "jkNo": "  012  ",
                        "jkName": " 홍길동 ",
                        "rcCntT": "150",
                        "ord1CntT": "30",
                    }
                }
            }
        }
    }

    result = parse_jockey_data(api_response)

    assert result == [{
        "jk_no": "012",
        "name": "홍길동",
        "total_races": 150,
        "total_wins": 30,
    }]


def test_save_trainer_batch_empty_returns_zero():
    from library.collector_trainers import save_to_supabase_batch
    result = save_to_supabase_batch([])
    assert result == 0


# ---------------------------------------------------------------------------
# collector_races: 서울 win_odds 병합 / 공용 API(API4_3) 호출
# ---------------------------------------------------------------------------

def _entry(win_odds, race_id="2026-06-06_서울_1", horse_id="0053568"):
    return {"race_id": race_id, "horse_id": horse_id, "final_rank": 6, "win_odds": win_odds}


def test_dedup_fills_missing_win_odds_from_duplicate_entry():
    """SeoulRace_1 행(win_odds 0)이 먼저 와도 API4_3 행의 배당이 버려지면 안 된다."""
    from library.collector_races import filter_duplicates_in_race_and_entries

    _, entries = filter_duplicates_in_race_and_entries([], [_entry(0.0), _entry(19.0)])

    assert len(entries) == 1
    assert entries[0]["win_odds"] == 19.0


def test_dedup_keeps_first_win_odds_when_already_present():
    from library.collector_races import filter_duplicates_in_race_and_entries

    _, entries = filter_duplicates_in_race_and_entries([], [_entry(2.5), _entry(19.0)])

    assert len(entries) == 1
    assert entries[0]["win_odds"] == 2.5


def _captured_request(mocker, **kwargs):
    from library import collector_races
    send = mocker.patch("requests.Session.send")
    send.return_value.status_code = 200
    send.return_value.json.return_value = {}
    collector_races.fetch_race_data_all(**kwargs)
    return send.call_args[0][0].url


def test_fetch_jeju_uses_common_api_with_meet_param(mocker):
    """제주 전용 API4_2는 폐기됨 -> 공용 API4_3에 meet=2로 조회해야 한다."""
    url = _captured_request(mocker, meet_code=2, end_date="202606")
    assert "API4_3/raceResult_3" in url
    assert "meet=2" in url
    assert "rc_month=202606" in url


def test_fetch_seoul_default_uses_seoul_api(mocker):
    url = _captured_request(mocker, meet_code=1, end_date="202606")
    assert "SeoulRace_1" in url


def test_fetch_seoul_common_api_queries_single_day(mocker):
    """common_api=True면 서울도 API4_3(말별 winOdds 있음), 8자리 날짜는 rc_date로."""
    url = _captured_request(mocker, meet_code=1, end_date="20260606", common_api=True)
    assert "API4_3/raceResult_3" in url
    assert "meet=1" in url
    assert "rc_date=20260606" in url
    assert "rc_month" not in url


def test_collecting_race_results_returns_first_collected_date(mocker):
    from library import collector_races
    mocker.patch.object(collector_races, "get_existing_master_data", return_value={})
    mocker.patch.object(collector_races, "fetch_pages_sequential", return_value={
        "race_entries": [{"race_date": "2026-10-03"}, {"race_date": "2026-10-02"}], "races": [{}]})
    mocker.patch.object(collector_races, "save_normalized_data", return_value={"race_entries": 2})
    assert collector_races.collecting_race_results(start_date="20261003", end_date="20261003") == "2026-10-02"


def test_collecting_race_results_returns_none_when_nothing_collected(mocker):
    from library import collector_races
    mocker.patch.object(collector_races, "get_existing_master_data", return_value={})
    mocker.patch.object(collector_races, "fetch_pages_sequential", return_value={})
    assert collector_races.collecting_race_results(start_date="20261003", end_date="20261003") is None


# ---------------------------------------------------------------------------
# collector_races: 말 출생연도·성별·거세 확인일
# ---------------------------------------------------------------------------

def _obs(race_date, age, sex, horse_id="0047063"):
    return {"horse_id": horse_id, "race_date": race_date, "age": age, "sex": sex}


def test_parse_collects_horse_age_and_sex_from_common_api():
    from library.collector_races import parse_and_normalize_race_data
    item = {"rcDate": 20260606, "meet": "서울", "rcNo": 1, "hrNo": "0047063", "hrName": "말",
            "jkNo": "080001", "trNo": "070001", "owNo": "010001", "chulNo": 3, "ord": 1,
            "age": 4, "sex": "거"}
    parsed = parse_and_normalize_race_data({"response": {"body": {"items": {"item": [item]}}}})
    assert parsed["horse_obs"] == [
        {"horse_id": "0047063", "race_date": "2026-06-06", "age": 4, "sex": "거"}]


def test_summarize_horse_attrs_birth_year_and_first_gelded_date():
    from library.collector_races import summarize_horse_attrs
    attrs = summarize_horse_attrs([
        _obs("2026-06-06", 5, "거"), _obs("2024-03-02", 3, "수"), _obs("2025-03-01", 4, "거"),
    ])
    assert attrs == [{"horse_id": "0047063", "birth_year": 2021,
                      "gender": "거", "gelded_since": "2025-03-01"}]


def test_summarize_horse_attrs_non_gelding_has_no_gelded_date():
    from library.collector_races import summarize_horse_attrs
    attrs = summarize_horse_attrs([_obs("2026-06-06", 3, "암")])
    assert attrs[0]["gender"] == "암"
    assert attrs[0]["gelded_since"] is None


def test_save_horse_attrs_keeps_earlier_existing_gelded_date(mocker):
    from library import collector_races
    client = mocker.MagicMock()
    client.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [
        {"horse_id": "0047063", "name": "DB이름", "gender": "거", "gelded_since": "2025-03-01"}]
    mocker.patch.object(collector_races, "get_client", return_value=client)

    collector_races.save_horse_attrs([
        {"horse_id": "0047063", "birth_year": 2021, "gender": "수", "gelded_since": None}])

    sent = client.table.return_value.upsert.call_args[0][0][0]
    assert sent["gelded_since"] == "2025-03-01"
    assert sent["gender"] == "거"
    assert sent["name"] == "DB이름"


def _save_attrs_with_existing(mocker, existing, attrs):
    from library import collector_races
    client = mocker.MagicMock()
    client.table.return_value.select.return_value.in_.return_value.execute.return_value.data = existing
    mocker.patch.object(collector_races, "get_client", return_value=client)
    collector_races.save_horse_attrs(attrs)
    return client.table.return_value.upsert


def test_save_horse_attrs_keeps_gelding_without_gelded_date(mocker):
    """마지막 출전 뒤에 거세된 말: DB는 '거', 경주 관측은 전부 '수'."""
    upsert = _save_attrs_with_existing(
        mocker, [{"horse_id": "A", "name": "말", "gender": "거", "gelded_since": None}],
        [{"horse_id": "A", "birth_year": 2020, "gender": "수", "gelded_since": None}])
    assert upsert.call_args[0][0][0]["gender"] == "거"


def test_save_horse_attrs_skips_horses_not_in_db(mocker):
    upsert = _save_attrs_with_existing(
        mocker, [], [{"horse_id": "NEW", "birth_year": 2020, "gender": "수", "gelded_since": None}])
    upsert.assert_not_called()
