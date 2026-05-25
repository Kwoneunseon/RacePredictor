import pytest
from unittest.mock import patch, MagicMock
from Utils.database import get_client, upsert
import Utils.database as _db_module


def test_get_client_returns_singleton(mocker):
    mock_create = mocker.patch("Utils.database.create_client", return_value=MagicMock())
    import Utils.database as db
    db._client = None  # reset singleton
    client1 = get_client()
    client2 = get_client()
    assert client1 is client2
    mock_create.assert_called_once()


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


def test_test_connection_returns_bool(mocker):
    mock_client = MagicMock()
    mocker.patch("Utils.database.get_client", return_value=mock_client)
    mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    result = _db_module.test_connection()
    assert isinstance(result, bool)
