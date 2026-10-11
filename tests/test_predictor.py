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


def test_predictor_top3_probability_uses_raw_not_calibrated():
    """top3_probability는 trainer.predict_proba_top3(X, raw=True)로 호출해 isotonic
    보정이 적용되지 않은 원본 분류기 확률을 써야 한다.

    이유(진단: scripts/diagnose_calibration.py, 2026-07-31): isotonic 보정은 top3
    쪽에서 Brier/ECE 개선폭이 미미한 반면(-1.4%/-33%, win 쪽 -23%/-75%보다 훨씬 약함),
    408경주 백테스트로 검증된 TOP3_PROB_THRESHOLD=0.65를 그대로 적용하면 통과 경주가
    61.6%→28.7%로 급감한다(2026-05-01~07-31, 596경주). 임계값 재검증 없이 이런 변화가
    조용히 반영되는 것을 막기 위해 raw=True를 명시적으로 강제한다.
    win_probability는 EV 계산에 절대값이 직접 쓰이고 개선폭도 크므로 계속 보정된 값을
    쓴다(raw 인자 없이 기본 호출) — 이 비대칭은 의도된 것.
    """
    import numpy as np, pandas as pd
    from core.predictor import RacePredictor
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.feature_columns = ["horse_age"]
    trainer.predict_rank_score.return_value = None
    trainer.predict_rank_score_top3.return_value = None
    trainer.predict_proba.return_value = np.array([0.5, 0.3, 0.2])
    trainer.predict_proba_top3.return_value = np.array([0.4, 0.4, 0.2])

    preprocessor = MagicMock()
    df = pd.DataFrame({
        "horse_age": [4, 5, 3],
        "horse_name": ["말A", "말B", "말C"],
        "entry_number": [1, 2, 3],
        "race_id": ["r1", "r1", "r1"],
    })

    p = RacePredictor(trainer, preprocessor)
    p.predict(df)

    assert trainer.predict_proba_top3.call_args.kwargs.get("raw") is True, \
        "predict_proba_top3는 raw=True로 호출되어야 함 (top3_probability는 미보정 원본 확률 유지)"
    # win_probability 쪽은 계속 보정된 값을 쓰므로 raw 인자를 넘기지 않아야 함(기본값=보정 적용)
    assert "raw" not in trainer.predict_proba.call_args.kwargs


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
