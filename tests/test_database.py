import pytest
from unittest.mock import patch, MagicMock
from Utils.database import get_client, upsert
import Utils.database as _db_module


@pytest.fixture(autouse=True)
def reset_db_singleton():
    _db_module._client = None
    yield
    _db_module._client = None


def test_get_client_returns_singleton(mocker):
    mocker.patch("Utils.database.create_client", return_value=MagicMock())
    client1 = get_client()
    client2 = get_client()
    assert client1 is client2


def test_upsert_empty_rows_returns_true():
    result = upsert("horses", [], "horse_id")
    assert result is True


def test_upsert_calls_supabase(mocker):
    mock_client = MagicMock()
    mocker.patch("Utils.database.get_client", return_value=mock_client)
    mock_client.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    rows = [{"horse_id": "H001", "name": "테스트말"}]
    result = upsert("horses", rows, "horse_id")
    assert result is True
    mock_client.table.assert_called_with("horses")


def test_test_connection_returns_true(mocker):
    mock_client = MagicMock()
    mocker.patch("Utils.database.get_client", return_value=mock_client)
    mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    assert _db_module.test_connection() is True


def test_test_connection_returns_false_on_exception(mocker):
    mocker.patch("Utils.database.get_client", side_effect=Exception("connection failed"))
    assert _db_module.test_connection() is False
