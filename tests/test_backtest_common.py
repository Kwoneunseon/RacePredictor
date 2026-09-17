import numpy as np
import pandas as pd
import pytest

from scripts._backtest_common import attach_harville_top3


def _make_result_df():
    """race_id별 그룹: R1(5두, 정상), R2(2두, 3두 미만이라 스킵 대상),
    R3(4두, 정상, final_rank는 순위 뒤섞임)."""
    return pd.DataFrame({
        "race_id":    ["R1"] * 5 + ["R2"] * 2 + ["R3"] * 4,
        "race_date":  ["2026-01-01"] * 5 + ["2026-01-02"] * 2 + ["2026-01-03"] * 4,
        "win_prob":   [0.4, 0.3, 0.15, 0.1, 0.05] + [0.6, 0.4] + [0.4, 0.3, 0.2, 0.1],
        "final_rank": [1, 2, 3, 4, 5] + [1, 2] + [2, 1, 4, 3],
    })


def test_attach_harville_top3_sums_to_three_per_race():
    df = _make_result_df()
    out = attach_harville_top3(df)

    r1 = out[out["race_id"] == "R1"]
    assert r1["harville_top3_prob"].sum() == pytest.approx(3.0, abs=1e-9)
    assert (r1["harville_top3_prob"] >= 0).all()


def test_attach_harville_top3_skips_races_with_fewer_than_three_horses():
    df = _make_result_df()
    out = attach_harville_top3(df)

    r2 = out[out["race_id"] == "R2"]
    assert r2["harville_top3_prob"].isna().all()


def test_attach_harville_top3_preserves_row_count():
    df = _make_result_df()
    out = attach_harville_top3(df)
    assert len(out) == len(df)


def test_attach_harville_top3_handles_zero_sum_strengths():
    df = pd.DataFrame({
        "race_id":    ["R4"] * 3,
        "race_date":  ["2026-01-04"] * 3,
        "win_prob":   [0.0, 0.0, 0.0],
        "final_rank": [1, 2, 3],
    })
    out = attach_harville_top3(df)
    assert out["harville_top3_prob"].isna().all()


from scripts._backtest_common import build_harville_pair_table
from core.rank_distribution import pair_top3_probabilities


def test_build_harville_pair_table_excludes_small_races():
    df = _make_result_df()
    table = build_harville_pair_table(df)
    assert "R2" not in set(table["race_id"])
    assert set(table["race_id"]) == {"R1", "R3"}


def test_build_harville_pair_table_picks_highest_probability_pair():
    df = _make_result_df()
    table = build_harville_pair_table(df)

    r1_row = table[table["race_id"] == "R1"].iloc[0]
    strengths = df[df["race_id"] == "R1"]["win_prob"].to_numpy()
    pair = pair_top3_probabilities(strengths)

    assert r1_row["harville_pair_prob"] == pytest.approx(pair.max())


def test_build_harville_pair_table_hit_reflects_final_rank():
    df = _make_result_df()
    table = build_harville_pair_table(df)

    strengths = df[df["race_id"] == "R3"]["win_prob"].to_numpy()
    final_ranks = df[df["race_id"] == "R3"]["final_rank"].to_numpy()
    pair = pair_top3_probabilities(strengths)
    n = len(strengths)
    iu = np.triu_indices(n, k=1)
    best_idx = np.argmax(pair[iu])
    a, b = iu[0][best_idx], iu[1][best_idx]
    expected_hit = bool(final_ranks[a] <= 3 and final_ranks[b] <= 3)

    r3_row = table[table["race_id"] == "R3"].iloc[0]
    assert bool(r3_row["harville_pair_hit"]) == expected_hit


def test_build_harville_pair_table_random_baseline_formula():
    df = _make_result_df()
    table = build_harville_pair_table(df)
    r1_row = table[table["race_id"] == "R1"].iloc[0]
    assert r1_row["random_baseline"] == pytest.approx(6.0 / (5 * 4))


def test_build_harville_pair_table_skips_zero_sum_strengths():
    df = pd.DataFrame({
        "race_id":    ["R5"] * 4,
        "race_date":  ["2026-01-05"] * 4,
        "win_prob":   [0.0, 0.0, 0.0, 0.0],
        "final_rank": [1, 2, 3, 4],
    })
    table = build_harville_pair_table(df)
    assert table.empty
