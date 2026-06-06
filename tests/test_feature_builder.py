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
# _flatten_entries 단위 테스트 (DB 불필요)
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
    df = fb._flatten_entries([make_raw_entry()])
    assert df["horse_name"].iloc[0] == "테스트마"
    assert df["horse_age"].iloc[0] == 4
    assert df["meet_code"].iloc[0] == 1
    assert df["race_distance"].iloc[0] == 1200
    assert df["jockey_total_wins"].iloc[0] == 100
    assert df["trainer_total_wins"].iloc[0] == 50


def test_flatten_entries_multiple_rows():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb._flatten_entries([make_raw_entry("H1"), make_raw_entry("H2")])
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

    with patch("core.feature_builder.get_client", return_value=client):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    required = [
        "horse_id", "horse_name", "meet_code", "race_distance",
        "prev_win_odds", "training_grade_enc", "days_since_last_race",
        "prev_speed_figure", "speed_figure_trend",
    ]
    for col in required:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_build_for_prediction_fallback_values():
    """피처 DB 조회 결과 없을 때 폴백값이 설정된다"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    assert result["prev_win_odds"].iloc[0] == 15.0
    assert result["training_grade_enc"].iloc[0] == 1.0
    assert result["days_since_last_race"].iloc[0] == 60
    assert result["prev_speed_figure"].iloc[0] == 0.0
