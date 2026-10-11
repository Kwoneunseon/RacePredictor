"""main.py / scripts/_backtest_common.py 로더의 유효 착순(1 <= final_rank < NON_FINISH_RANK_MIN) 필터 테스트.

주의: 대상 main.py·scripts/_backtest_common.py는 사용자 미커밋 변경과 섞여 있어
이 테스트 파일도 해당 파일들과 함께 커밋 여부를 결정한다.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config import NON_FINISH_RANK_MIN


class _RecordingQuery:
    """PostgREST 체인 호출을 순서 무관하게 기록하는 가짜 쿼리 빌더 (DB 미접속)."""

    def __init__(self, data, calls):
        self._data = data
        self.calls = calls

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args))
            return self
        return _record

    def execute(self):
        return MagicMock(data=self._data, count=len(self._data))


class _RecordingClient:
    def __init__(self, data=None):
        self.data = data or []
        self.calls = []

    def table(self, name):
        self.calls.append(("table", (name,)))
        return _RecordingQuery(self.data, self.calls)


def _has_valid_rank_filter(calls):
    return ("gt", ("final_rank", 0)) in calls and ("lt", ("final_rank", NON_FINISH_RANK_MIN)) in calls


def test_fetch_race_entries_query_filters_valid_rank_range():
    import main
    client = _RecordingClient(data=[])
    args = SimpleNamespace(start_date="2026-01-01", end_date="2026-06-30")
    with patch("main.get_client", return_value=client):
        main._fetch_race_entries(args)
    assert _has_valid_rank_filter(client.calls)


def _entries_with_partially_deleted_race():
    """R1은 한 마리만 deleted=1(부분 오염), R2는 전부 정상."""
    return [
        {"id": 1, "race_id": "R1", "horse_id": "H1", "final_rank": 1, "win_odds": 0.0, "deleted": 1},
        {"id": 2, "race_id": "R1", "horse_id": "H2", "final_rank": 2, "win_odds": 3.1, "deleted": 0},
        {"id": 3, "race_id": "R2", "horse_id": "H3", "final_rank": 1, "win_odds": 2.4, "deleted": 0},
        {"id": 4, "race_id": "R2", "horse_id": "H4", "final_rank": 2, "win_odds": 5.0, "deleted": 0},
    ]


def test_drop_deleted_races_removes_whole_race_and_keeps_rows_without_flag():
    from core.feature_builder import drop_deleted_races
    kept = drop_deleted_races(_entries_with_partially_deleted_race())
    assert [e["id"] for e in kept] == [3, 4]
    legacy = [{"id": 9, "race_id": "R9"}]  # deleted 키가 없는 행은 그대로 둔다
    assert drop_deleted_races(legacy) == legacy


def test_fetch_race_entries_drops_races_with_deleted_rows():
    import main
    client = _RecordingClient(data=_entries_with_partially_deleted_race())
    args = SimpleNamespace(start_date="2026-01-01", end_date="2026-06-30")
    with patch("main.get_client", return_value=client):
        df = main._fetch_race_entries(args)
    assert sorted(df["race_id"].unique()) == ["R2"]


def test_load_historical_data_drops_races_with_deleted_rows():
    from scripts import _backtest_common
    client = _RecordingClient(data=_entries_with_partially_deleted_race())
    with patch.object(_backtest_common, "get_client", return_value=client):
        df = _backtest_common.load_historical_data("2026-05-01", "2026-06-30")
    assert sorted(df["race_id"].unique()) == ["R2"]


def test_build_for_prediction_does_not_filter_deleted():
    """경주 전 출전표는 win_odds=0이라 deleted=1 - 예측 경로에 필터가 걸리면 대상이 사라진다."""
    from core.feature_builder import RaceFeatureBuilder
    client = _RecordingClient(data=[])
    with patch("core.feature_builder.get_client", return_value=client):
        RaceFeatureBuilder().build_for_prediction("2026-10-11")
    assert not any("deleted" in args for _, args in client.calls)


def test_fetch_actual_results_query_filters_valid_rank_range():
    import main
    client = _RecordingClient(data=[])
    main._fetch_actual_results("2026-07-11", client)
    assert _has_valid_rank_filter(client.calls)


def test_load_historical_data_query_filters_valid_rank_range():
    from scripts import _backtest_common
    client = _RecordingClient(data=[])
    with patch.object(_backtest_common, "get_client", return_value=client):
        _backtest_common.load_historical_data("2026-05-01", "2026-06-30")
    assert _has_valid_rank_filter(client.calls)


class _KeysetQuery:
    """id keyset(.gt("id", n).limit(k))만 흉내 내는 가짜 쿼리 빌더."""

    def __init__(self, rows, total):
        self._rows, self._total = rows, total
        self._after, self._limit = 0, None

    def gt(self, column, value):
        if column == "id":
            self._after = value
        return self

    def limit(self, n):
        self._limit = n
        return self

    def range(self, *args):
        raise AssertionError("로더는 OFFSET(range) 대신 id keyset으로 읽어야 한다")

    def __getattr__(self, name):
        return lambda *args, **kwargs: self

    def execute(self):
        page = [r for r in self._rows if r["id"] > self._after][: self._limit]
        return MagicMock(data=page, count=self._total)


class _KeysetClient:
    def __init__(self, rows, total=None):
        self.rows = rows
        self.total = len(rows) if total is None else total

    def table(self, name):
        return _KeysetQuery(self.rows, self.total)


def _many_entries(n):
    # id에 구멍을 둬서 offset 계산으로는 맞출 수 없게 한다
    return [
        {"id": i * 3, "race_id": f"R{i // 10}", "horse_id": f"H{i}", "final_rank": 1, "win_odds": 2.0, "deleted": 0}
        for i in range(1, n + 1)
    ]


def _run_main_loader(client):
    import main
    args = SimpleNamespace(start_date="2024-01-01", end_date="2026-10-11")
    with patch("main.get_client", return_value=client):
        return main._fetch_race_entries(args)


def _run_backtest_loader(client):
    from scripts import _backtest_common
    with patch.object(_backtest_common, "get_client", return_value=client):
        return _backtest_common.load_historical_data("2024-01-01", "2026-10-11")


@pytest.mark.parametrize("run_loader", [_run_main_loader, _run_backtest_loader])
def test_loader_reads_every_page_to_the_end(run_loader):
    df = run_loader(_KeysetClient(_many_entries(2500)))
    assert len(df) == 2500
    assert df["horse_id"].nunique() == 2500


@pytest.mark.parametrize("run_loader", [_run_main_loader, _run_backtest_loader])
def test_loader_reads_exact_multiple_of_page_size(run_loader):
    assert len(run_loader(_KeysetClient(_many_entries(2000)))) == 2000


@pytest.mark.parametrize("run_loader", [_run_main_loader, _run_backtest_loader])
def test_loader_raises_when_fewer_rows_than_count(run_loader):
    with pytest.raises(RuntimeError, match="로드 불완전"):
        run_loader(_KeysetClient(_many_entries(1500), total=1600))


def _entries_with_unranked_timed_row():
    """R2에는 완주기록은 있는데 착순이 없는 말이 있다(R1은 정상)."""
    return [
        {"id": 1, "race_id": "R1", "horse_id": "H1", "final_rank": 1, "finish_time": 80.0, "deleted": 0},
        {"id": 2, "race_id": "R1", "horse_id": "H2", "final_rank": 2, "finish_time": 80.5, "deleted": 0},
        {"id": 3, "race_id": "R2", "horse_id": "H3", "final_rank": 1, "finish_time": 81.0, "deleted": 0},
        {"id": 4, "race_id": "R2", "horse_id": "H4", "final_rank": 0, "finish_time": 81.4, "deleted": 0},
        {"id": 5, "race_id": "R3", "horse_id": "H5", "final_rank": 0, "finish_time": 0.0, "deleted": 0},
    ]


def test_fetch_unranked_race_ids_returns_races_with_time_but_no_rank():
    from core.feature_builder import fetch_unranked_race_ids
    client = _RecordingClient(data=_entries_with_unranked_timed_row())
    assert fetch_unranked_race_ids(client, "2026-07-01", "2026-07-31") == {"R2"}
    assert ("eq", ("final_rank", 0)) in client.calls and ("gt", ("finish_time", 0)) in client.calls


def test_fetch_race_entries_drops_races_with_unranked_timed_rows():
    import main
    client = _RecordingClient(data=_entries_with_unranked_timed_row()[:4])
    args = SimpleNamespace(start_date="2026-07-01", end_date="2026-07-31")
    with patch("main.get_client", return_value=client):
        df = main._fetch_race_entries(args)
    assert sorted(df["race_id"].unique()) == ["R1"]


def test_load_historical_data_drops_races_with_unranked_timed_rows():
    from scripts import _backtest_common
    client = _RecordingClient(data=_entries_with_unranked_timed_row()[:4])
    with patch.object(_backtest_common, "get_client", return_value=client):
        df = _backtest_common.load_historical_data("2026-07-01", "2026-07-31")
    assert sorted(df["race_id"].unique()) == ["R1"]
