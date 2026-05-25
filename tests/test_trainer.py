import pytest
import pandas as pd
import numpy as np
from core.trainer import RaceModelTrainer


@pytest.fixture
def preprocessed_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "horse_age": np.random.randint(3, 8, n),
        "is_male": np.random.randint(0, 2, n),
        "horse_class_enc": np.random.randint(0, 5, n),
        "race_distance": np.random.choice([1000, 1200, 1400, 1800], n),
        "total_horses": np.random.randint(5, 14, n),
        "horse_weight": np.random.randint(440, 520, n),
        "race_grade_enc": np.random.randint(0, 3, n),
        "track_condition_enc": np.random.randint(0, 3, n),
        "weather_enc": np.random.randint(0, 3, n),
        "prev_total_races": np.random.randint(3, 30, n),
        "prev_5_avg_rank": np.random.uniform(1, 8, n),
        "prev_total_avg_rank": np.random.uniform(1, 8, n),
        "jockey_win_rate": np.random.uniform(0, 0.3, n),
        "trainer_win_rate": np.random.uniform(0, 0.3, n),
        "horse_win_rate": np.random.uniform(0, 0.3, n),
        "horse_top3_rate": np.random.uniform(0, 0.5, n),
        "experience_score": np.random.uniform(0, 4, n),
        "recent_form": np.random.uniform(-2, 5, n),
        "is_winner": np.random.randint(0, 2, n),
        "race_date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })
    return df


def test_trainer_creates_models(preprocessed_df):
    trainer = RaceModelTrainer()
    result = trainer.train(preprocessed_df)
    assert "models" in result
    assert len(result["models"]) >= 3


def test_trainer_returns_metrics(preprocessed_df):
    trainer = RaceModelTrainer()
    result = trainer.train(preprocessed_df)
    assert "metrics" in result
    for name, metrics in result["metrics"].items():
        assert "auc" in metrics
        assert 0 <= metrics["auc"] <= 1


def test_trainer_predict_proba_shape(preprocessed_df):
    trainer = RaceModelTrainer()
    trainer.train(preprocessed_df)
    from config import FEATURE_COLUMNS
    available = [c for c in FEATURE_COLUMNS if c in preprocessed_df.columns]
    X = preprocessed_df[available].head(10)
    proba = trainer.predict_proba(X)
    assert proba.shape == (10,)
    assert (proba >= 0).all() and (proba <= 1).all()
