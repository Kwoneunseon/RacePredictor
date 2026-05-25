import pytest
import pandas as pd
import numpy as np
from core.preprocessor import RacePreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "horse_id": ["H1", "H2"],
        "race_id": ["R1", "R1"],
        "race_date": ["2025-01-01", "2025-01-01"],
        "final_rank": [1, 2],
        "horse_age": [4, 5],
        "gender": ["수컷", "암컷"],
        "horse_class": ["나", "다"],
        "race_distance": [1200, 1200],
        "total_horses": [8, 8],
        "horse_weight": [480, 470],
        "race_grade": ["일반", "일반"],
        "track_condition": ["양호", "양호"],
        "weather": ["맑음", "맑음"],
        "jockey_total_races": [100, 200],
        "jockey_total_wins": [20, 40],
        "trainer_total_races": [150, 300],
        "trainer_total_wins": [30, 60],
        "prev_total_races": [5, 10],
        "prev_wins": [1, 3],
        "prev_top3": [3, 7],
        "prev_5_avg_rank": [3.0, 2.0],
        "prev_total_avg_rank": [3.5, 2.2],
    })


def test_preprocessor_adds_is_winner(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert "is_winner" in result.columns
    assert result["is_winner"].iloc[0] == 1
    assert result["is_winner"].iloc[1] == 0


def test_preprocessor_adds_derived_features(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    for col in ["jockey_win_rate", "trainer_win_rate", "horse_win_rate",
                "horse_top3_rate", "experience_score", "recent_form", "is_male"]:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_preprocessor_encodes_categoricals(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    for col in ["horse_class_enc", "race_grade_enc", "track_condition_enc", "weather_enc"]:
        assert col in result.columns
        assert result[col].dtype in [np.int64, np.int32, np.float64]


def test_preprocessor_transform_uses_fitted_encoders(sample_df):
    p = RacePreprocessor()
    p.fit_transform(sample_df)
    new_df = sample_df.copy()
    result = p.transform(new_df)
    assert "horse_class_enc" in result.columns


def test_preprocessor_win_rate_range(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert (result["jockey_win_rate"] >= 0).all()
    assert (result["jockey_win_rate"] <= 1).all()
