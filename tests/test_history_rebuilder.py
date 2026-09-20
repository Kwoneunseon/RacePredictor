import pytest
import pandas as pd
import numpy as np
from core.history_rebuilder import compute_history


def test_compute_history_first_race_no_prior():
    """첫 경주는 직전 기록이 없으므로 prev_total_races=0"""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 1, "horse_weight": 480},
    ])
    result = compute_history(df)
    row = result[result["horse_id"] == "H1"].iloc[0]
    assert row["prev_total_races"] == 0
    assert row["prev_wins"] == 0


def test_compute_history_second_race_uses_first():
    """두 번째 경주의 통계는 첫 경주 결과를 반영해야 한다"""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 1, "horse_weight": 480},
        {"horse_id": "H1", "race_date": "2024-01-08", "final_rank": 2, "horse_weight": 482},
    ])
    result = compute_history(df)
    second = result[result["as_of_date"] == "2024-01-08"].iloc[0]
    assert second["prev_total_races"] == 1
    assert second["prev_wins"] == 1
    assert second["prev_top3"] == 1


def test_compute_history_multiple_horses():
    """말이 여러 마리일 때 각각 독립적으로 계산된다"""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 1, "horse_weight": 480},
        {"horse_id": "H2", "race_date": "2024-01-01", "final_rank": 3, "horse_weight": 500},
        {"horse_id": "H1", "race_date": "2024-01-08", "final_rank": 2, "horse_weight": 482},
    ])
    result = compute_history(df)
    assert len(result) == 3
    h2_rows = result[result["horse_id"] == "H2"]
    assert h2_rows.iloc[0]["prev_total_races"] == 0


def test_compute_history_prev5_avg_uses_last_five():
    """prev_5_avg_rank는 최근 5경주 평균이다"""
    ranks = [1, 2, 3, 4, 5, 1]
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": f"2024-0{i+1}-01", "final_rank": r, "horse_weight": 480}
        for i, r in enumerate(ranks)
    ])
    result = compute_history(df)
    last = result[result["horse_id"] == "H1"].sort_values("as_of_date").iloc[-1]
    # 직전 5경주: [2, 3, 4, 5, 1] → 평균 3.0
    assert abs(last["prev_5_avg_rank"] - 3.0) < 0.01


# ──────────────────────────────────────────────
# 90번대(비착순 상태 코드) 필터
# ──────────────────────────────────────────────
def test_compute_history_excludes_non_finish_ranks_from_stats():
    """91~99(취소/제외/주행중지 등)와 0(미기입)은 평균 착순·출전횟수에 들어가지 않는다."""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 2, "horse_weight": 480},
        {"horse_id": "H1", "race_date": "2024-01-08", "final_rank": 99, "horse_weight": 480},
        {"horse_id": "H1", "race_date": "2024-01-15", "final_rank": 93, "horse_weight": 480},
        {"horse_id": "H1", "race_date": "2024-01-22", "final_rank": 0, "horse_weight": 480},
        {"horse_id": "H1", "race_date": "2024-01-29", "final_rank": 4, "horse_weight": 480},
    ])
    result = compute_history(df).sort_values("as_of_date").reset_index(drop=True)
    # 유효 행은 2024-01-01, 2024-01-29 두 개뿐
    assert list(result["as_of_date"]) == ["2024-01-01", "2024-01-29"]
    last = result.iloc[-1]
    assert last["prev_total_races"] == 1
    assert last["prev_5_avg_rank"] == 2.0
    assert last["prev_total_avg_rank"] == 2.0
    assert last["prev_top3"] == 1


def test_compute_history_all_non_finish_returns_empty_without_error():
    """경주 전체 취소(99)처럼 유효 행이 하나도 없으면 예외 없이 빈 결과."""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2026-03-20", "final_rank": 99, "horse_weight": 480},
        {"horse_id": "H2", "race_date": "2026-03-20", "final_rank": 99, "horse_weight": 500},
    ])
    result = compute_history(df)
    assert len(result) == 0


def test_load_all_entries_filters_valid_rank_range_in_query():
    """로더 쿼리가 final_rank > 0 이고 < NON_FINISH_RANK_MIN 조건을 함께 건다."""
    from unittest.mock import MagicMock
    from config import NON_FINISH_RANK_MIN
    from core.history_rebuilder import load_all_entries

    client = MagicMock()
    q = client.table.return_value.select.return_value
    q.gt.return_value.lt.return_value.order.return_value.order.return_value \
        .range.return_value.execute.return_value.data = []

    load_all_entries(client)

    q.gt.assert_called_once_with("final_rank", 0)
    q.gt.return_value.lt.assert_called_once_with("final_rank", NON_FINISH_RANK_MIN)
