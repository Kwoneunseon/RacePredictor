import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────
# _training_grade 단위 테스트 (DB 불필요)
# ──────────────────────────────────────────────
def test_training_grade_양호():
    from core.feature_builder import _training_grade
    assert _training_grade("양호") == 2


def test_training_grade_불량():
    from core.feature_builder import _training_grade
    assert _training_grade("불량") == 0


def test_training_grade_빈문자():
    from core.feature_builder import _training_grade
    assert _training_grade("") == 1


def test_training_grade_일반():
    from core.feature_builder import _training_grade
    assert _training_grade("보통") == 1


# ──────────────────────────────────────────────
# get_flatten_entries 단위 테스트 (DB 불필요)
# ──────────────────────────────────────────────
def make_raw_entry(horse_id="H1", race_date="2026-06-07"):
    return {
        "id": 1,
        "race_id": f"{race_date}_서울_1",
        "race_date": race_date,
        "horse_id": horse_id,
        "entry_number": 1,
        "horse_weight": 480,
        "final_rank": None,
        "jk_no": "J001",
        "trainer_id": "T001",
        "win_odds": 0,
        "finish_time": None,
        "horses": {"name": "테스트마", "age": 4, "gender": "수컷", "rank": "국5"},
        "races": {
            "race_no": 1, "meet_code": 1, "race_distance": 1200,
            "total_horses": 8, "race_grade": "일반",
            "track_condition": "양호", "weather": "맑음",
        },
        "jockeys": {"total_wins": 100, "total_races": 1000},
        "trainers": {"ord1_cnt_t": 50, "rc_cnt_t": 500},
    }


def test_flatten_entries_extracts_nested_fields():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb.get_flatten_entries([make_raw_entry()])
    assert df["horse_name"].iloc[0] == "테스트마"
    assert df["horse_age"].iloc[0] == 4
    assert df["meet_code"].iloc[0] == 1
    assert df["race_distance"].iloc[0] == 1200
    assert df["jockey_total_wins"].iloc[0] == 100
    assert df["trainer_total_wins"].iloc[0] == 50


def test_flatten_entries_multiple_rows():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb.get_flatten_entries([make_raw_entry("H1"), make_raw_entry("H2")])
    assert len(df) == 2
    assert set(df["horse_id"]) == {"H1", "H2"}


# ──────────────────────────────────────────────
# build_for_prediction 통합 테스트 (DB mock)
# ──────────────────────────────────────────────
def make_mock_client(entry_data=None):
    """race_entries 첫 조회에만 데이터 반환, 나머지는 빈 리스트"""
    client = MagicMock()
    mock_chain = client.table.return_value.select.return_value

    # race_entries 첫 조회 (.eq("race_date", ...))
    mock_chain.eq.return_value.execute.return_value.data = entry_data or []

    # 이후 피처 조회들 → 빈 리스트 (MagicMock이 체인을 자동 생성하므로
    # execute().data만 설정하면 됨)
    mock_chain.in_.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.gte.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.gte.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    # _fetch_assigned_weight: .select(...).in_("race_id", ...).execute() (중간 필터 없음)
    mock_chain.in_.return_value.execute.return_value.data = []

    return client


def test_build_for_prediction_empty_db():
    """DB에 데이터 없으면 빈 DataFrame 반환"""
    with patch("core.feature_builder.get_client", return_value=make_mock_client([])):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_build_for_prediction_returns_required_columns():
    """race_entries가 있으면 필수 컬럼을 포함한 DataFrame 반환"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={}):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    required = [
        "horse_id", "horse_name", "meet_code", "race_distance",
        "prev_win_odds", "current_win_odds", "training_grade_enc",
        "days_since_last_race", "prev_speed_figure", "speed_figure_trend",
        "assigned_weight",
    ]
    for col in required:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_build_for_prediction_fallback_values():
    """피처 DB 조회 결과 없을 때 폴백값이 설정된다"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={}):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    assert result["prev_win_odds"].iloc[0] == 15.0
    # 스크래핑 실패 시 current_win_odds = prev_win_odds
    assert result["current_win_odds"].iloc[0] == 15.0
    assert result["training_grade_enc"].iloc[0] == 1.0
    assert result["days_since_last_race"].iloc[0] == 60
    assert result["prev_speed_figure"].iloc[0] == 0.0
    # race_entry_detail에 매칭되는 행 없음 → 기본 폴백값 56.0
    assert result["assigned_weight"].iloc[0] == 56.0


# ──────────────────────────────────────────────
# _fetch_distance_rates 단위 테스트 (DB mock)
# ──────────────────────────────────────────────
def test_fetch_distance_rates_computes_speed_at_distance():
    """같은 거리 과거 경주들의 평균 속도(거리/기록시간)를 계산한다."""
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "race_distance": [1200]})

    client = MagicMock()
    chain = client.table.return_value.select.return_value.in_.return_value.lt.return_value.gt.return_value
    chain.execute.return_value.data = [
        {"horse_id": "H1", "final_rank": 2, "finish_time": 80.0, "races": {"race_distance": 1200}},
        {"horse_id": "H1", "final_rank": 1, "finish_time": 75.0, "races": {"race_distance": 1200}},
    ]

    result = fb._fetch_distance_rates(df, "2026-06-07", client)
    assert "horse_speed_at_distance" in result.columns
    expected = (1200 / 80.0 + 1200 / 75.0) / 2
    assert result["horse_speed_at_distance"].iloc[0] == pytest.approx(expected)


def test_fetch_distance_rates_speed_defaults_to_zero_when_no_history():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "race_distance": [1200]})

    client = MagicMock()
    chain = client.table.return_value.select.return_value.in_.return_value.lt.return_value.gt.return_value
    chain.execute.return_value.data = []

    result = fb._fetch_distance_rates(df, "2026-06-07", client)
    assert result["horse_speed_at_distance"].iloc[0] == 0.0


def test_fetch_assigned_weight_for_training_fills_median_when_missing():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "race_id": ["R1", "R2", "R3"],
        "horse_id": ["H1", "H2", "H3"],
    })
    detail_rows = [
        {"race_id": "R1", "horse_id": "H1", "assigned_weight": 54.0},
        {"race_id": "R2", "horse_id": "H2", "assigned_weight": 58.0},
        # R3/H3는 race_entry_detail에 없음 (수집 실패/미백필 케이스)
    ]
    result = fb._fetch_assigned_weight_for_training(df, detail_rows)
    assert result.loc[result["race_id"] == "R1", "assigned_weight"].iloc[0] == 54.0
    assert result.loc[result["race_id"] == "R2", "assigned_weight"].iloc[0] == 58.0
    # 결측은 중앙값(54, 58의 중앙값=56)으로 대체
    assert result.loc[result["race_id"] == "R3", "assigned_weight"].iloc[0] == 56.0


def test_build_for_prediction_uses_scraped_odds():
    """스크래핑 성공 시 current_win_odds에 실시간 배당이 반영된다"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    # entry_number=1인 말에 배당 5.3 반환
    with patch("core.feature_builder.get_client", return_value=client), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={1: 5.3}):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    assert result["current_win_odds"].iloc[0] == pytest.approx(5.3)


# ──────────────────────────────────────────────
# 2단계: 구간기록·통산/최근1년 성적
# ──────────────────────────────────────────────
def test_fetch_section_features_treats_zero_corner_as_not_applicable():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"],
        "race_date": ["2026-07-12"],
        "race_distance": [1000],  # 임계값 미만 → 1C/2C 미해당
    })
    section_rows = [{
        "horse_id": "H1", "race_distance": 1000, "snapshot_race_date": "2026-07-01",
        "c1_avg": 0.0, "c2_avg": 0.0, "c3_avg": 30.0, "c4_avg": 49.0,
        "s1f_avg": 13.9, "g3f_avg": 37.8, "g1f_avg": 13.5,
    }]
    result = fb._join_section_records(df, section_rows)
    assert result.loc[0, "c3_avg"] == 30.0
    assert result.loc[0, "g1f_avg"] == 13.5


def test_horse_career_win_rate_uses_starts_as_denominator():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "race_id": ["R1"], "horse_id": ["H1"],
    })
    detail_rows = [{
        "race_id": "R1", "horse_id": "H1",
        "horse_career_wins": 3, "horse_career_starts": 10,
        "jockey_career_wins": 50, "jockey_career_starts": 500,
        "jockey_recent1y_wins": 5, "jockey_recent1y_starts": 40,
        "trainer_career_wins": 20, "trainer_career_starts": 300,
        "trainer_recent1y_wins": 2, "trainer_recent1y_starts": 30,
        "assigned_weight": 56.0,
    }]
    result = fb._join_career_stats(df, detail_rows)
    assert result.loc[0, "horse_career_win_rate"] == 0.3
    assert result.loc[0, "jockey_recent1y_win_rate"] == 0.125


# ──────────────────────────────────────────────
# 3단계: 러닝스타일 (코너별 통과순위)
# ──────────────────────────────────────────────
def test_running_style_score_positive_means_closer():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"], "race_date": ["2026-07-12"],
    })
    # 과거 2경주: 1코너 5위→최종 2위(3계단 추입), 1코너 6위→최종 1위(5계단 추입)
    past_rows = [
        {"horse_id": "H1", "race_date": "2026-06-01", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 2},
        {"horse_id": "H1", "race_date": "2026-06-15", "sj_1c_ord": 6, "sj_3c_ord": 5, "final_rank": 1},
    ]
    result = fb._compute_running_style(df, past_rows)
    assert result.loc[0, "running_style_score"] == pytest.approx((5 - 2 + 6 - 1) / 2)


def test_running_style_score_falls_back_to_3c_when_1c_not_applicable():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"], "race_date": ["2026-07-12"],
    })
    # 짧은 거리라 1C 없음(None) → 3C(4위) 기준으로 계산해야 함: 4 - 2 = 2
    past_rows = [
        {"horse_id": "H1", "race_date": "2026-06-01", "sj_1c_ord": None, "sj_3c_ord": 4, "final_rank": 2},
    ]
    result = fb._compute_running_style(df, past_rows)
    assert result.loc[0, "running_style_score"] == pytest.approx(2.0)
