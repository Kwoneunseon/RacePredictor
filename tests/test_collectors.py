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
