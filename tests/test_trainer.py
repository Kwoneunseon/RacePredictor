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
    # race_id 없으면 ranker skip → RF·LR 최소 2개, race_id 있으면 +2(XGB·LGB)
    assert len(result["models"]) >= 2


def test_trainer_returns_metrics(preprocessed_df):
    # preprocessed_df에 race_id·final_rank 없어 ranker skip → RF·LR 메트릭만 검증
    trainer = RaceModelTrainer()
    result = trainer.train(preprocessed_df)
    assert "metrics" in result
    for name, m in result["metrics"].items():
        if "precision_at_1" in m:
            assert 0 <= m["precision_at_1"] <= 1
        elif "auc" in m:
            assert 0 <= m["auc"] <= 1


def test_trainer_predict_proba_shape(preprocessed_df):
    trainer = RaceModelTrainer()
    trainer.train(preprocessed_df)
    from config import FEATURE_COLUMNS
    available = [c for c in FEATURE_COLUMNS if c in preprocessed_df.columns]
    X = preprocessed_df[available].head(10)
    proba = trainer.predict_proba(X)
    assert proba.shape == (10,)
    assert (proba >= 0).all() and (proba <= 1).all()


def _make_race_df(n_races=30, horses_per=8, seed=0):
    """ranker 테스트용 경주 데이터 생성."""
    np.random.seed(seed)
    n = n_races * horses_per
    race_ids = [f"r{i}" for i in range(n_races) for _ in range(horses_per)]
    final_ranks = [j + 1 for _ in range(n_races) for j in range(horses_per)]
    dates = [f"2024-{(i // 28) % 12 + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
    return pd.DataFrame({
        "race_id": race_ids,
        "race_date": pd.to_datetime(dates),
        "final_rank": final_ranks,
        "is_winner": [1 if r == 1 else 0 for r in final_ranks],
        "horse_age": np.random.randint(3, 8, n),
        "horse_class_num": np.random.randint(1, 6, n),
        "horse_weight": np.random.randint(440, 520, n),
        "race_distance": np.random.choice([1200, 1400], n),
        "race_grade_enc": np.random.randint(0, 3, n),
        "jockey_win_rate": np.random.uniform(0, 0.3, n),
        "trainer_win_rate": np.random.uniform(0, 0.3, n),
        "horse_win_rate": np.random.uniform(0, 0.3, n),
        "horse_top3_rate": np.random.uniform(0, 0.5, n),
        "prev_total_races": np.random.randint(3, 30, n),
        "prev_5_avg_rank": np.random.uniform(1, 8, n),
        "prev_total_avg_rank": np.random.uniform(1, 8, n),
        "horse_class_vs_mean": np.random.uniform(-2, 2, n),
        "win_rate_vs_mean": np.random.uniform(-0.1, 0.1, n),
        "jockey_recent_win_rate": np.random.uniform(0, 0.3, n),
        "trainer_recent_win_rate": np.random.uniform(0, 0.3, n),
        "training_grade_enc": np.random.uniform(0, 3, n),
        "days_since_last_race": np.random.randint(7, 60, n),
        "horse_weight_diff": np.random.uniform(-10, 10, n),
        "horse_win_rate_at_distance": np.random.uniform(0, 0.3, n),
        "horse_track_win_rate": np.random.uniform(0, 0.3, n),
        "jockey_horse_win_rate": np.random.uniform(0, 0.3, n),
        "jockey_track_win_rate": np.random.uniform(0, 0.3, n),
        "recent_rank_trend": np.random.uniform(-1, 1, n),
        "weight_trend": np.random.uniform(-5, 5, n),
        "entry_number": np.tile(np.arange(1, horses_per + 1), n_races),
        "prev_win_odds": np.random.uniform(2, 30, n),
    })


def test_xgb_lgb_are_rankers():
    """학습 후 XGBoost·LightGBM 모델이 XGBRanker·LGBMRanker여야 한다."""
    from xgboost import XGBRanker
    from lightgbm import LGBMRanker as LGBMRankerCls

    df = _make_race_df()
    trainer = RaceModelTrainer()
    trainer.train(df)

    assert isinstance(trainer.models["XGBoost"], XGBRanker), \
        "XGBoost 모델이 XGBRanker가 아님"
    assert isinstance(trainer.models["LightGBM"], LGBMRankerCls), \
        "LightGBM 모델이 LGBMRanker가 아님"
    assert not hasattr(trainer.models["XGBoost"], "predict_proba"), \
        "XGBRanker에 predict_proba가 있으면 안 됨"


def test_ranker_scores_winner_higher():
    """predict_rank_score에서 승자의 평균 점수가 패자보다 높아야 한다."""
    np.random.seed(1)
    n_races, horses_per = 30, 8
    n = n_races * horses_per
    race_ids = [f"r{i}" for i in range(n_races) for _ in range(horses_per)]
    final_ranks = [j + 1 for _ in range(n_races) for j in range(horses_per)]
    # 승자는 jockey_win_rate·horse_win_rate·prev_5_avg_rank가 유리하게 설정
    jockey_wr, horse_wr, avg_rank = [], [], []
    for rank in final_ranks:
        base = 0.3 - rank * 0.025
        jockey_wr.append(max(0.0, base + np.random.uniform(-0.005, 0.005)))
        horse_wr.append(max(0.0, base + np.random.uniform(-0.005, 0.005)))
        avg_rank.append(rank * 0.9 + np.random.uniform(0, 0.3))
    dates = [f"2024-{(i // 28) % 12 + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
    df = pd.DataFrame({
        "race_id": race_ids,
        "race_date": pd.to_datetime(dates),
        "final_rank": final_ranks,
        "is_winner": [1 if r == 1 else 0 for r in final_ranks],
        "horse_age": np.random.randint(3, 8, n),
        "horse_class_num": np.random.randint(1, 6, n),
        "horse_weight": np.random.randint(440, 520, n),
        "race_distance": np.random.choice([1200, 1400], n),
        "race_grade_enc": np.random.randint(0, 3, n),
        "jockey_win_rate": jockey_wr,
        "trainer_win_rate": np.random.uniform(0, 0.3, n),
        "horse_win_rate": horse_wr,
        "horse_top3_rate": np.random.uniform(0, 0.5, n),
        "prev_total_races": np.random.randint(3, 30, n),
        "prev_5_avg_rank": avg_rank,
        "prev_total_avg_rank": np.random.uniform(1, 8, n),
        "horse_class_vs_mean": np.random.uniform(-2, 2, n),
        "win_rate_vs_mean": np.random.uniform(-0.1, 0.1, n),
        "jockey_recent_win_rate": np.random.uniform(0, 0.3, n),
        "trainer_recent_win_rate": np.random.uniform(0, 0.3, n),
        "training_grade_enc": np.random.uniform(0, 3, n),
        "days_since_last_race": np.random.randint(7, 60, n),
        "horse_weight_diff": np.random.uniform(-10, 10, n),
        "horse_win_rate_at_distance": np.random.uniform(0, 0.3, n),
        "horse_track_win_rate": np.random.uniform(0, 0.3, n),
        "jockey_horse_win_rate": np.random.uniform(0, 0.3, n),
        "jockey_track_win_rate": np.random.uniform(0, 0.3, n),
        "recent_rank_trend": np.random.uniform(-1, 1, n),
        "weight_trend": np.random.uniform(-5, 5, n),
        "entry_number": np.tile(np.arange(1, horses_per + 1), n_races),
        "prev_win_odds": np.random.uniform(2, 30, n),
    })
    trainer = RaceModelTrainer()
    trainer.train(df)
    available = [c for c in trainer.feature_columns if c in df.columns]
    X = df[available].fillna(0)
    scores = trainer.predict_rank_score(X)
    assert scores is not None, "predict_rank_score가 None 반환"
    winner_scores = scores[np.array(final_ranks) == 1]
    others_scores = scores[np.array(final_ranks) > 1]
    assert winner_scores.mean() > others_scores.mean(), \
        f"승자 평균 점수({winner_scores.mean():.3f}) ≤ 패자 평균({others_scores.mean():.3f})"


def test_precision_at_1_perfect():
    """모든 경주에서 1위를 정확히 예측하면 1.0"""
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1", "r1", "r1", "r2", "r2", "r2"],
        "final_rank": [1, 2, 3, 1, 2, 3],
        "score":      [0.9, 0.5, 0.2, 0.8, 0.4, 0.1],
    })
    result = trainer._precision_at_1_score(df, score_col="score")
    assert result == pytest.approx(1.0)


def test_precision_at_1_zero():
    """모든 경주에서 1위를 틀리면 0.0"""
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1", "r1", "r1", "r2", "r2", "r2"],
        "final_rank": [1, 2, 3, 1, 2, 3],
        "score":      [0.1, 0.5, 0.9, 0.1, 0.4, 0.8],
    })
    result = trainer._precision_at_1_score(df, score_col="score")
    assert result == pytest.approx(0.0)


def test_precision_at_1_half():
    """절반 적중하면 0.5"""
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1", "r1", "r2", "r2"],
        "final_rank": [1, 2, 1, 2],
        "score":      [0.9, 0.1, 0.1, 0.9],
    })
    result = trainer._precision_at_1_score(df, score_col="score")
    assert result == pytest.approx(0.5)


def test_predict_rank_score_top3_returns_none_when_empty():
    trainer = RaceModelTrainer()
    X = pd.DataFrame({"horse_age": [4, 5], "horse_win_rate": [0.1, 0.2]})
    result = trainer.predict_rank_score_top3(X)
    assert result is None


def test_precision_at_3_perfect():
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1"] * 6,
        "final_rank": [1, 2, 3, 4, 5, 6],
        "score":      [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
    })
    result = trainer._precision_at_3_score(df, score_col="score")
    assert result == pytest.approx(1.0)


def test_precision_at_3_zero():
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1"] * 6,
        "final_rank": [1, 2, 3, 4, 5, 6],
        "score":      [0.1, 0.2, 0.3, 0.9, 0.8, 0.7],
    })
    result = trainer._precision_at_3_score(df, score_col="score")
    assert result == pytest.approx(0.0)
