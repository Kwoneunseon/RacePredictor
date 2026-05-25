import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from core.predictor import RacePredictor


@pytest.fixture
def mock_trainer_and_preprocessor():
    trainer = MagicMock()
    trainer.feature_columns = ["horse_age", "is_male", "horse_win_rate"]
    trainer.predict_proba.return_value = np.array([0.8, 0.5, 0.3])
    preprocessor = MagicMock()
    preprocessor.transform.return_value = pd.DataFrame({
        "horse_age": [4, 5, 3],
        "is_male": [1, 0, 1],
        "horse_win_rate": [0.2, 0.1, 0.05],
        "horse_name": ["말A", "말B", "말C"],
        "entry_number": [1, 2, 3],
    })
    return trainer, preprocessor


def test_predictor_returns_sorted_predictions(mock_trainer_and_preprocessor):
    trainer, preprocessor = mock_trainer_and_preprocessor
    p = RacePredictor(trainer, preprocessor)
    df = preprocessor.transform.return_value
    result = p.predict(df)
    assert result["win_probability"].iloc[0] >= result["win_probability"].iloc[1]


def test_predictor_result_has_required_columns(mock_trainer_and_preprocessor):
    trainer, preprocessor = mock_trainer_and_preprocessor
    p = RacePredictor(trainer, preprocessor)
    df = preprocessor.transform.return_value
    result = p.predict(df)
    for col in ["entry_number", "win_probability", "predicted_rank"]:
        assert col in result.columns


def test_predictor_probabilities_in_range(mock_trainer_and_preprocessor):
    trainer, preprocessor = mock_trainer_and_preprocessor
    p = RacePredictor(trainer, preprocessor)
    df = preprocessor.transform.return_value
    result = p.predict(df)
    assert (result["win_probability"] >= 0).all()
    assert (result["win_probability"] <= 1).all()
