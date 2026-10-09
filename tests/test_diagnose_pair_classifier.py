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


from scripts.diagnose_pair_classifier import add_race_relative_features


def _make_race_relative_df():
    """R1: 3개 페어, p_indep이 전부 다름(0.5, 0.5, 0.2 중 둘이 동일 -> 동률
    케이스 포함). R2: 2개 페어, p_indep 전부 다름."""
    return pd.DataFrame({
        "race_id":  ["R1", "R1", "R1", "R2", "R2"],
        "entry_i":  [1, 1, 2, 1, 1],
        "entry_j":  [2, 3, 3, 2, 3],
        "p_indep":  [0.5, 0.5, 0.2, 0.6, 0.3],
    })


def test_add_race_relative_features_rank_highest_p_indep_is_1():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    row = result[(result.race_id == "R2") & (result.entry_j == 2)].iloc[0]
    assert row["p_indep_rank_in_race"] == pytest.approx(1.0)


def test_add_race_relative_features_tied_p_indep_gets_average_rank():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    r1 = result[result.race_id == "R1"]
    tied_rows = r1[r1["p_indep"] == 0.5]
    assert len(tied_rows) == 2
    assert tied_rows["p_indep_rank_in_race"].tolist() == pytest.approx([1.5, 1.5])
    lowest = r1[r1["p_indep"] == 0.2].iloc[0]
    assert lowest["p_indep_rank_in_race"] == pytest.approx(3.0)


def test_add_race_relative_features_ratio_to_max_is_one_for_top_pair():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    r2 = result[result.race_id == "R2"]
    top = r2[r2["p_indep"] == 0.6].iloc[0]
    assert top["p_indep_ratio_to_max"] == pytest.approx(1.0)
    other = r2[r2["p_indep"] == 0.3].iloc[0]
    assert other["p_indep_ratio_to_max"] == pytest.approx(0.5)
    assert 0.0 <= other["p_indep_ratio_to_max"] <= 1.0


def test_add_race_relative_features_n_pairs_in_race_matches_group_size():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    assert (result[result.race_id == "R1"]["n_pairs_in_race"] == 3).all()
    assert (result[result.race_id == "R2"]["n_pairs_in_race"] == 2).all()


def test_add_race_relative_features_races_do_not_mix():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    # R2는 페어가 2개뿐이지만 R1(3개)의 n_pairs_in_race에 영향을 주지 않는다
    assert (result[result.race_id == "R1"]["n_pairs_in_race"] != 2).all()
    # R1 순위가 R2 p_indep 값(0.6, 0.3)에 의해 바뀌지 않는다
    r1_ranks = sorted(result[result.race_id == "R1"]["p_indep_rank_in_race"].tolist())
    assert r1_ranks == [1.5, 1.5, 3.0]


def test_add_race_relative_features_order_independent():
    df = _make_race_relative_df()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    result = add_race_relative_features(shuffled)
    r2 = result[result.race_id == "R2"]
    top = r2[r2["p_indep"] == 0.6].iloc[0]
    assert top["p_indep_rank_in_race"] == pytest.approx(1.0)
    assert top["p_indep_ratio_to_max"] == pytest.approx(1.0)
