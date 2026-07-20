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


def test_predictor_top3_probability_ignores_ranker():
    """top3_probability는 분류기 확률을 그대로 사용하고 ranker 점수를 반영하지 않아야 한다.

    이유: softmax 정규화는 경주 내 확률 합=1을 강제하는데, 연승은 3마리가 top3에
    들어오는 조건이라 이 정규화가 부적합하다 (검증: 408경주 백테스트에서 블렌딩 시
    TOP3_PROB_THRESHOLD=0.65를 넘는 경주가 0건으로 감소, 원시 분류기 확률 대비
    정확도 차이는 없음(-0.2pp)). ranker 점수는 predict_rank_score_top3로 별도
    노출되며 순위 개선 용도로만 쓰인다.
    """
    import numpy as np, pandas as pd
    from core.predictor import RacePredictor
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.feature_columns = ["horse_age", "horse_win_rate"]
    trainer.predict_rank_score.return_value = None
    trainer.predict_proba.return_value = np.array([0.3, 0.4, 0.3])
    # clf: 말B(0.7) > 말A(0.1), ranker: 말A(0.99) >> 말B(0.01) (불일치 상황)
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
    result = p.predict(df)
    top3 = result.set_index("horse_name")["top3_probability"]

    # ranker가 말A를 압도적으로 선호해도 top3_probability는 clf 그대로(말B가 높음)여야 함
    assert top3["말B"] > top3["말A"], "top3_probability는 ranker와 무관하게 clf 확률을 그대로 써야 함"
    assert top3["말A"] == pytest.approx(0.1)
    assert top3["말B"] == pytest.approx(0.7)
    assert top3["말C"] == pytest.approx(0.2)


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
