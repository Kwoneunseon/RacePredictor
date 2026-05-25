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


def test_save_trainer_batch_empty_returns_zero():
    from library.collector_trainers import save_to_supabase_batch
    result = save_to_supabase_batch([])
    assert result == 0
