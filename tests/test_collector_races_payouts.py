# -*- coding: utf-8 -*-
"""
library/collector_races.py의 plcOdds(연승 확정 배당) 파싱 테스트.
제주(raceResult_2)·부산경남(raceResult_3) API 응답에만 있는 개별 말 plcOdds
필드를 race_payouts(bet_type='place') 형태로 변환하는 로직 검증
([[project_race_payouts_scraper]] 참고). 서울(SeoulRace_1)은 이 필드가 없어
race_payouts를 만들지 않아야 한다.
"""
from library.collector_races import (
    parse_and_normalize_race_data,
    filter_duplicates_in_payouts,
    PLC_ODDS_SUPPORTED_MEET_CODES,
)


def _make_response(items):
    return {"response": {"body": {"items": {"item": items}}}}


def _busan_item(chul_no, plc_odds, win_odds=3.5, rc_no=1, rc_date=20260531):
    return {
        "rcDate": rc_date,
        "meet": "부산경남",
        "rcNo": rc_no,
        "hrNo": f"005{chul_no:04d}",
        "hrName": f"말{chul_no}",
        "jkNo": "080001",
        "jkName": "기수",
        "prtr": "070001",
        "prtrName": "조교사",
        "owNo": "010001",
        "owName": "마주",
        "chulNo": chul_no,
        "winOdds": win_odds,
        "plcOdds": plc_odds,
        "ord": chul_no,
        "rcDist": 1300,
        "wgHr": "480(+2)",
    }


def _seoul_item(chul_no, rc_no=1, rc_date=20260502):
    return {
        "rcDate": rc_date,
        "meet": "서울",
        "rcNo": rc_no,
        "hrno": f"005{chul_no:04d}",
        "hrName": f"말{chul_no}",
        "jkNo": "080001",
        "jkName": "기수",
        "prtr": "070001",
        "prtrName": "조교사",
        "prow": "010001",
        "prowName": "마주",
        "rcChul": chul_no,
        "winOdds": 4.5,
        "chaksun": chul_no,
        "rcDist": 1200,
        "wgHr": 480,
    }


class TestPlcOddsSupportedMeets:
    def test_supported_meets_are_jeju_and_busan(self):
        assert PLC_ODDS_SUPPORTED_MEET_CODES == {2, 3}


class TestParsePlaceOddsFromPlcOdds:
    def test_busan_place_payouts_parsed(self):
        items = [_busan_item(1, 1.2), _busan_item(2, 5.6), _busan_item(3, 2.3)]
        result = parse_and_normalize_race_data(_make_response(items))

        payouts = result["race_payouts"]
        assert len(payouts) == 3
        by_entry = {p["entry_numbers"][0]: p for p in payouts}
        assert by_entry[1]["odds"] == 1.2
        assert by_entry[2]["odds"] == 5.6
        assert by_entry[3]["odds"] == 2.3
        for p in payouts:
            assert p["bet_type"] == "place"
            assert p["race_id"] == "2026-05-31_부산경남_1"
            assert len(p["entry_numbers"]) == 1

    def test_seoul_has_no_place_payouts(self):
        """서울(meet_code=1) API 응답에는 plcOdds 필드 자체가 없다 - race_payouts가
        비어 있어야 한다."""
        items = [_seoul_item(1), _seoul_item(2)]
        result = parse_and_normalize_race_data(_make_response(items))
        assert result["race_payouts"] == []
        # race_entries는 정상적으로 파싱되어야 함 (place payouts만 스킵)
        assert len(result["race_entries"]) == 2

    def test_missing_or_zero_plc_odds_skipped(self):
        item_missing = _busan_item(1, None)
        item_zero = _busan_item(2, 0)
        item_valid = _busan_item(3, 4.4)
        result = parse_and_normalize_race_data(_make_response([item_missing, item_zero, item_valid]))
        payouts = result["race_payouts"]
        assert len(payouts) == 1
        assert payouts[0]["entry_numbers"] == [3]
        assert payouts[0]["odds"] == 4.4

    def test_missing_entry_number_skipped(self):
        item = _busan_item(1, 3.3)
        item["chulNo"] = None
        result = parse_and_normalize_race_data(_make_response([item]))
        assert result["race_payouts"] == []


class TestFilterDuplicatesInPayouts:
    def test_dedup_by_race_bet_entry(self):
        payouts = [
            {"race_id": "r1", "bet_type": "place", "entry_numbers": [1], "odds": 2.0},
            {"race_id": "r1", "bet_type": "place", "entry_numbers": [1], "odds": 9.9},  # 중복 키, 다른 값
            {"race_id": "r1", "bet_type": "place", "entry_numbers": [2], "odds": 3.0},
        ]
        deduped = filter_duplicates_in_payouts(payouts)
        assert len(deduped) == 2
        assert deduped[0]["odds"] == 2.0  # 선착순(첫 값) 유지

    def test_dedup_empty_list(self):
        assert filter_duplicates_in_payouts([]) == []
