import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_pair_interaction import build_pair_table


def _make_df():
    """R1: 4두 전부 유효 착순(1,2,3,4위). R2: 2두인데 1두는 비착순 코드(99)라
    유효 말이 1두뿐 -> 페어 불가."""
    return pd.DataFrame({
        "race_id":      ["R1"] * 4 + ["R2"] * 2,
        "race_date":    ["2026-01-01"] * 4 + ["2026-01-02"] * 2,
        "entry_number": [1, 2, 3, 4, 1, 2],
        "top3_prob":    [0.7, 0.6, 0.3, 0.2, 0.5, 0.5],
        "final_rank":   [1, 2, 3, 4, 1, 99],
    })


def test_build_pair_table_pair_count_matches_combinations():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    assert len(pairs[pairs["race_id"] == "R1"]) == 6  # C(4,2)
    assert len(pairs[pairs["race_id"] == "R2"]) == 0  # 유효 말 1두뿐


def test_build_pair_table_label_is_both_top3():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["label"] == 1  # 둘 다 rank<=3
    row14 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row14["label"] == 0  # entry4는 rank=4


def test_build_pair_table_p_indep_is_product_of_top3_prob():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["p_indep"] == pytest.approx(0.7 * 0.6)


def test_build_pair_table_both_early_leader_flag():
    df = _make_df()
    # entry1,2 -> 초반순위 평균 1.0(<=3, 선행형), entry3,4 -> 5.0(선행형 아님)
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    both_leader = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert both_leader["both_early_leader"] == True
    mixed = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 3)].iloc[0]
    assert mixed["both_early_leader"] == False
    neither = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 3) & (pairs.entry_j == 4)].iloc[0]
    assert neither["both_early_leader"] == False


def test_build_pair_table_pace_pressure_counts_early_leaders_in_race():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    r1_pairs = pairs[pairs.race_id == "R1"]
    assert (r1_pairs["pace_pressure"] == 2).all()  # entry1,2 두 마리가 선행형


def test_build_pair_table_nan_avg_early_ord_treated_as_not_leader():
    df = pd.DataFrame({
        "race_id":      ["R3"] * 3,
        "race_date":    ["2026-01-03"] * 3,
        "entry_number": [1, 2, 3],
        "top3_prob":    [0.5, 0.4, 0.3],
        "final_rank":   [1, 2, 3],
    })
    avg_early_ord = pd.Series([1.0, np.nan, 1.0], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["both_early_leader"] == False  # entry2는 기록 없음(NaN) -> 선행형 아님 취급
    row13 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 3)].iloc[0]
    assert row13["both_early_leader"] == True


def test_build_pair_table_entry_gap_is_absolute_difference():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row["entry_gap"] == pytest.approx(3.0)
