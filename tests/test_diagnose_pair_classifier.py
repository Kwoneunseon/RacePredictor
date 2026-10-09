import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_pair_classifier import build_classifier_pair_table


def _make_df():
    """R1: 4두 전부 유효 착순(1,2,3,4위). R2: 3두 중 1두는 비착순 코드(99)라
    유효 말이 2두뿐 -> 3두 미만이라 페어 불가."""
    return pd.DataFrame({
        "race_id":      ["R1"] * 4 + ["R2"] * 3,
        "race_date":    ["2026-01-01"] * 4 + ["2026-01-02"] * 3,
        "entry_number": [1, 2, 3, 4, 1, 2, 3],
        "win_prob":     [0.5, 0.3, 0.15, 0.05, 0.4, 0.4, 0.2],
        "top3_prob":    [0.7, 0.6, 0.3, 0.2, 0.5, 0.5, 0.5],
        "final_rank":   [1, 2, 3, 4, 1, 2, 99],
    })


def test_build_classifier_pair_table_pair_count_matches_combinations():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    assert len(pairs[pairs["race_id"] == "R1"]) == 6  # C(4,2)
    assert len(pairs[pairs["race_id"] == "R2"]) == 0  # 유효 말 2두뿐(3두 미만)


def test_build_classifier_pair_table_label_is_both_top3():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["label"] == 1
    row14 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row14["label"] == 0


def test_build_classifier_pair_table_win_prob_diff_is_absolute():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row["win_prob_diff"] == pytest.approx(abs(0.5 - 0.05))


def test_build_classifier_pair_table_win_prob_ratio_is_symmetric_and_bounded():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    expected = min(0.5, 0.05) / max(0.5, 0.05)
    assert row["win_prob_ratio"] == pytest.approx(expected)
    assert 0.0 <= row["win_prob_ratio"] <= 1.0


def test_build_classifier_pair_table_p_indep_is_product_of_top3_prob():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row["p_indep"] == pytest.approx(0.7 * 0.6)


def test_build_classifier_pair_table_both_early_leader_and_pace_pressure():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    r1 = pairs[pairs.race_id == "R1"]
    both_leader = r1[(r1.entry_i == 1) & (r1.entry_j == 2)].iloc[0]
    assert both_leader["both_early_leader"] == True
    assert (r1["pace_pressure"] == 2).all()


def test_build_classifier_pair_table_nan_avg_early_ord_treated_as_not_leader():
    df = pd.DataFrame({
        "race_id":      ["R3"] * 3,
        "race_date":    ["2026-01-03"] * 3,
        "entry_number": [1, 2, 3],
        "win_prob":     [0.5, 0.3, 0.2],
        "top3_prob":    [0.5, 0.4, 0.3],
        "final_rank":   [1, 2, 3],
    })
    avg_early_ord = pd.Series([1.0, np.nan, 1.0], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["both_early_leader"] == False
    row13 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 3)].iloc[0]
    assert row13["both_early_leader"] == True


def test_build_classifier_pair_table_entry_gap_is_absolute_difference():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row["entry_gap"] == pytest.approx(3.0)


def test_build_classifier_pair_table_win_prob_ratio_handles_zero_win_prob():
    # lo/hi는 lo<=hi이므로 hi(분모)가 0이 되는 경우는 두 말의 win_prob가
    # 모두 0일 때뿐이다. entry 1,2가 그 경우이고, entry 3은 유효 말 3두
    # 조건(페어 생성 최소 조건)을 채우기 위해 넣는다.
    df = pd.DataFrame({
        "race_id":      ["R5"] * 3,
        "race_date":    ["2026-01-05"] * 3,
        "entry_number": [1, 2, 3],
        "win_prob":     [0.0, 0.0, 0.2],
        "top3_prob":    [0.5, 0.4, 0.3],
        "final_rank":   [1, 2, 3],
    })
    avg_early_ord = pd.Series([1.0, 1.0, 1.0], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row["win_prob_ratio"] == pytest.approx(0.0)
    assert np.isfinite(row["win_prob_ratio"])
