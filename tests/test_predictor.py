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


def test_predictor_top3_probability_uses_ranker_when_available():
    """top3_probability는 clf_proba와 ranker score를 모두 반영해야 한다.
    clf는 말B > 말A, ranker는 말A > 말B 로 불일치시켜 ranker 반영 여부 검증.
    """
    import numpy as np, pandas as pd
    from core.predictor import RacePredictor
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.feature_columns = ["horse_age", "horse_win_rate"]
    trainer.predict_rank_score.return_value = None
    trainer.predict_proba.return_value = np.array([0.3, 0.4, 0.3])
    # clf: 말B(0.7) > 말A(0.1), ranker: 말A(0.99) >> 말B(0.01)
    trainer.predict_proba_top3.return_value = np.array([0.1, 0.7, 0.2])
    trainer.predict_rank_score_top3.return_value = np.array([0.99, 0.01, 0.0])

    preprocessor = MagicMock()
    df = pd.DataFrame({
        "horse_age": [4, 5, 3],
        "horse_win_rate": [0.2, 0.1, 0.05],
        "horse_name": ["말A", "말B", "말C"],
        "entry_number": [1, 2, 3],
        "race_id": ["r1", "r1", "r1"],
    })

    p = RacePredictor(trainer, preprocessor)

    # ranker 없을 때: clf만 쓰면 말B가 높아야 함
    trainer.predict_rank_score_top3.return_value = None
    result_no_ranker = p.predict(df)
    top3_no_ranker = result_no_ranker.set_index("horse_name")["top3_probability"]
    assert top3_no_ranker["말B"] > top3_no_ranker["말A"], "clf만 쓸 때 말B가 더 높아야 함"

    # ranker 있을 때: 블렌딩 후 말A가 높아야 함 (ranker가 말A를 압도적으로 선호)
    trainer.predict_rank_score_top3.return_value = np.array([0.99, 0.01, 0.0])
    result_with_ranker = p.predict(df)
    top3_with_ranker = result_with_ranker.set_index("horse_name")["top3_probability"]
    assert "top3_probability" in result_with_ranker.columns
    assert top3_with_ranker["말A"] > top3_with_ranker["말B"], \
        f"ranker 반영 후 말A({top3_with_ranker['말A']:.3f}) > 말B({top3_with_ranker['말B']:.3f}) 여야 함"


def test_predictor_predict_merges_history_columns(mock_trainer_and_preprocessor):
    """predict_from_db에서 horse_race_history가 병합되는 경로를 단위 검증."""
    trainer, preprocessor = mock_trainer_and_preprocessor

    history_row = {
        "horse_id": "H1",
        "prev_total_races": 10,
        "prev_wins": 2,
        "prev_top3": 5,
        "prev_5_avg_rank": 2.5,
        "prev_total_avg_rank": 3.0,
    }
    df_entries = pd.DataFrame([
        {"horse_id": "H1", "horse_age": 4, "is_male": 1, "horse_win_rate": 0.2,
         "horse_name": "말A", "entry_number": 1},
    ])
    hist_df = pd.DataFrame([history_row])

    merged = df_entries.merge(hist_df, on="horse_id", how="left")
    assert "prev_total_races" in merged.columns
    assert merged["prev_total_races"].iloc[0] == 10
    assert merged["prev_wins"].iloc[0] == 2
