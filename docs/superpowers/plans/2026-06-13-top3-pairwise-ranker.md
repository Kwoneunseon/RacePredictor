# 연승 Pairwise Ranker 추가 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 연승(top3) 모델에 XGBRanker + LGBMRanker를 추가해 top3_probability 품질을 높인다.

**Architecture:** 현재 연승 모델(RF+LR)에 단승과 동일한 패턴의 pairwise ranker 두 개를 추가하고, predictor에서 ranker score와 분류 확률을 블렌딩한다. 수정 파일은 3개(trainer, predictor, model_manager)이며 신규 파일은 없다.

**Tech Stack:** XGBoost(XGBRanker), LightGBM(LGBMRanker), pandas, numpy, pytest

**Before/After 기준선:**
- 연승 Precision@1 현재: 65.1% (목표 68%+)
- top3_prob >= 0.65 적중률 현재: 66.0% (목표 69%+)

---

## Task 1: model_manager.py — top3_ranker_models 저장/로드

**Files:**
- Modify: `core/model_manager.py:19-25` (save), `core/model_manager.py:57-60` (load)
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_model_manager.py` 끝에 추가:

```python
def test_save_load_preserves_top3_ranker_models(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    trainer = MagicMock()
    trainer.models = {}
    trainer.top3_models = {}
    trainer.top3_ranker_models = {"XGBoost": "fake_xgb", "LightGBM": "fake_lgb"}
    trainer.scaler = "fake_scaler"
    trainer.feature_columns = ["col1"]
    preprocessor = MagicMock()
    preprocessor.label_encoders = {}

    mm = ModelManager()
    version = mm.save(trainer, preprocessor)

    new_trainer = MagicMock()
    new_preprocessor = MagicMock()
    mm.load(new_trainer, new_preprocessor, version=version)

    assert new_trainer.top3_ranker_models == {"XGBoost": "fake_xgb", "LightGBM": "fake_lgb"}
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_model_manager.py::test_save_load_preserves_top3_ranker_models -v
```

Expected: `FAILED` — `AttributeError: Mock object has no attribute 'top3_ranker_models'` 또는 키 없음

- [ ] **Step 3: model_manager.py 수정**

`save()` 메서드의 `data` dict (line 19):

```python
data = {
    "models": trainer.models,
    "top3_models": trainer.top3_models,
    "top3_ranker_models": trainer.top3_ranker_models,   # ← 추가
    "scaler": trainer.scaler,
    "feature_columns": trainer.feature_columns,
    "label_encoders": preprocessor.label_encoders,
}
```

`load()` 메서드 (line 57~60):

```python
trainer.models = data["models"]
trainer.top3_models = data.get("top3_models", {})
trainer.top3_ranker_models = data.get("top3_ranker_models", {})   # ← 추가
trainer.scaler = data["scaler"]
trainer.feature_columns = data["feature_columns"]
preprocessor.label_encoders = data["label_encoders"]
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_model_manager.py -v
```

Expected: 전체 PASSED

- [ ] **Step 5: 커밋**

```
git add core/model_manager.py tests/test_model_manager.py
git commit -m "feat: model_manager save/load top3_ranker_models"
```

---

## Task 2: trainer.py — top3_ranker_models 초기화 + predict_rank_score_top3() + _precision_at_3_score()

**Files:**
- Modify: `core/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_trainer.py` 끝에 추가:

```python
def test_predict_rank_score_top3_returns_none_when_empty():
    trainer = RaceModelTrainer()
    # top3_ranker_models 비어있으면 None
    X = pd.DataFrame({"horse_age": [4, 5], "horse_win_rate": [0.1, 0.2]})
    result = trainer.predict_rank_score_top3(X)
    assert result is None


def test_precision_at_3_perfect():
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1"] * 6,
        "final_rank": [1, 2, 3, 4, 5, 6],
        "score":      [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],  # 상위 3개 = rank 1,2,3
    })
    result = trainer._precision_at_3_score(df, score_col="score")
    assert result == pytest.approx(1.0)


def test_precision_at_3_zero():
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":    ["r1"] * 6,
        "final_rank": [1, 2, 3, 4, 5, 6],
        "score":      [0.1, 0.2, 0.3, 0.9, 0.8, 0.7],  # 상위 3개 = rank 4,5,6
    })
    result = trainer._precision_at_3_score(df, score_col="score")
    assert result == pytest.approx(0.0)
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_trainer.py::test_predict_rank_score_top3_returns_none_when_empty tests/test_trainer.py::test_precision_at_3_perfect tests/test_trainer.py::test_precision_at_3_zero -v
```

Expected: `FAILED` — `AttributeError`

- [ ] **Step 3: trainer.py 수정**

`__init__` 에 `top3_ranker_models` 추가 (line 18 이후):

```python
def __init__(self):
    self.models: dict = {}              # 단승(1등) 분류 + ranker 모델
    self.top3_models: dict = {}         # 연승(3등 이내) 분류 모델 (RF, LR)
    self.top3_ranker_models: dict = {}  # 연승 ranker 모델 (XGBRanker, LGBMRanker)
    self.scaler = StandardScaler()
    self.feature_columns: list = []
```

`predict_rank_score()` 메서드 뒤에 `predict_rank_score_top3()` 추가:

```python
def predict_rank_score_top3(self, X: pd.DataFrame) -> np.ndarray | None:
    """연승 XGBRanker·LGBMRanker 점수를 min-max 정규화 후 평균하여 반환."""
    X = X.fillna(0)
    scores_list = []
    for name in ["XGBoost", "LightGBM"]:
        ranker = self.top3_ranker_models.get(name)
        if ranker is None:
            continue
        raw = ranker.predict(X)
        s_min, s_max = raw.min(), raw.max()
        normalized = (raw - s_min) / (s_max - s_min) if s_max > s_min \
            else np.full(len(raw), 0.5)
        scores_list.append(normalized)
    if not scores_list:
        return None
    return np.mean(scores_list, axis=0)
```

`_precision_at_1_score()` 메서드 뒤에 `_precision_at_3_score()` 추가:

```python
def _precision_at_3_score(self, df: pd.DataFrame, score_col: str = "score") -> float:
    """각 경주에서 score 상위 3마리 중 실제 final_rank <= 3인 말의 비율 평균."""
    scores = []
    for _, grp in df.groupby("race_id"):
        if len(grp) < 3:
            continue
        top3_predicted = grp.nlargest(3, score_col)
        hit = (top3_predicted["final_rank"] <= 3).sum()
        scores.append(hit / 3)
    return float(np.mean(scores)) if scores else 0.0
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_trainer.py::test_predict_rank_score_top3_returns_none_when_empty tests/test_trainer.py::test_precision_at_3_perfect tests/test_trainer.py::test_precision_at_3_zero -v
```

Expected: 전체 PASSED

- [ ] **Step 5: 커밋**

```
git add core/trainer.py tests/test_trainer.py
git commit -m "feat: trainer top3_ranker_models, predict_rank_score_top3, _precision_at_3_score"
```

---

## Task 3: trainer.py — _train_pairwise_rankers_top3() + train() 연결

**Files:**
- Modify: `core/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_trainer.py` 끝에 추가 (`_make_race_df` 헬퍼는 이미 파일에 있음):

```python
def test_top3_xgb_lgb_are_rankers():
    """학습 후 top3_ranker_models에 XGBRanker·LGBMRanker가 있어야 한다."""
    from xgboost import XGBRanker
    from lightgbm import LGBMRanker as LGBMRankerCls

    df = _make_race_df()
    trainer = RaceModelTrainer()
    trainer.train(df)

    assert "XGBoost" in trainer.top3_ranker_models, "top3 XGBoost ranker 없음"
    assert "LightGBM" in trainer.top3_ranker_models, "top3 LightGBM ranker 없음"
    assert isinstance(trainer.top3_ranker_models["XGBoost"], XGBRanker)
    assert isinstance(trainer.top3_ranker_models["LightGBM"], LGBMRankerCls)


def test_top3_ranker_scores_top3_horses_higher():
    """predict_rank_score_top3에서 top3 말의 평균 점수가 나머지보다 높아야 한다."""
    np.random.seed(42)
    n_races, horses_per = 30, 8
    n = n_races * horses_per
    race_ids = [f"r{i}" for i in range(n_races) for _ in range(horses_per)]
    final_ranks = [j + 1 for _ in range(n_races) for j in range(horses_per)]
    # top3 말에 유리한 피처 설정
    jockey_wr, horse_wr = [], []
    for rank in final_ranks:
        base = 0.3 - rank * 0.02
        jockey_wr.append(max(0.0, base + np.random.uniform(-0.01, 0.01)))
        horse_wr.append(max(0.0, base + np.random.uniform(-0.01, 0.01)))
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
        "prev_5_avg_rank": [r * 0.9 + np.random.uniform(0, 0.3) for r in final_ranks],
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
    scores = trainer.predict_rank_score_top3(X)
    assert scores is not None
    is_top3 = np.array(final_ranks) <= 3
    assert scores[is_top3].mean() > scores[~is_top3].mean(), \
        f"top3 평균({scores[is_top3].mean():.3f}) ≤ 나머지 평균({scores[~is_top3].mean():.3f})"
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_trainer.py::test_top3_xgb_lgb_are_rankers tests/test_trainer.py::test_top3_ranker_scores_top3_horses_higher -v
```

Expected: `FAILED` — `AssertionError: top3 XGBoost ranker 없음`

- [ ] **Step 3: trainer.py에 _train_pairwise_rankers_top3() 추가**

`_train_pairwise_rankers()` 메서드 바로 뒤에 추가:

```python
def _train_pairwise_rankers_top3(
    self, df_sorted: pd.DataFrame, feature_cols: list, best_params: dict
) -> dict:
    """연승(top3) 전용 XGBRanker(pairwise)·LGBMRanker(lambdarank) 학습."""
    if "race_id" not in df_sorted.columns or "final_rank" not in df_sorted.columns:
        logger.warning("Top3 Ranker 학습 불가: race_id 또는 final_rank 없음")
        return {}

    race_dates = df_sorted.groupby("race_id")["race_date"].first().sort_values()
    split_idx  = int(len(race_dates) * (1 - MODEL_CONFIG["test_size"]))
    train_races = set(race_dates.index[:split_idx])
    test_races  = set(race_dates.index[split_idx:])

    df_train = df_sorted[df_sorted["race_id"].isin(train_races)].sort_values("race_id")
    df_test  = df_sorted[df_sorted["race_id"].isin(test_races)].sort_values("race_id")

    if df_train.empty or df_test.empty:
        logger.warning("Top3 Ranker: train/test split 후 데이터 부족")
        return {}

    def make_top3_label(rank: int) -> int:
        if rank == 1: return 2
        if rank <= 3: return 1
        return 0

    X_train = df_train[feature_cols].fillna(0)
    X_test  = df_test[feature_cols].fillna(0)
    y_train = df_train["final_rank"].apply(make_top3_label)
    train_groups = df_train.groupby("race_id", sort=False).size().values
    xgb_p = best_params.get("XGBoost", {})
    lgb_p = best_params.get("LightGBM", {})
    metrics = {}

    logger.info("Top3 XGBRanker(pairwise) 학습 중...")
    xgb_ranker = XGBRanker(
        objective="rank:pairwise",
        n_estimators=     xgb_p.get("n_estimators", 200),
        max_depth=        xgb_p.get("max_depth", 6),
        learning_rate=    xgb_p.get("learning_rate", 0.1),
        subsample=        xgb_p.get("subsample", 1.0),
        colsample_bytree= xgb_p.get("colsample_bytree", 1.0),
        random_state=MODEL_CONFIG["random_state"],
        verbosity=0,
    )
    qid_train = np.repeat(np.arange(len(train_groups)), train_groups)
    xgb_ranker.fit(X_train, y_train, qid=qid_train)
    xgb_scores = xgb_ranker.predict(X_test)
    xgb_p3   = self._precision_at_3_score(df_test.assign(score=xgb_scores), "score")
    xgb_ndcg = self._eval_ndcg(df_test, xgb_scores, feature_cols)
    self.top3_ranker_models["XGBoost"] = xgb_ranker
    metrics["top3_XGBoost"] = {"precision_at_3": xgb_p3, "ndcg3": xgb_ndcg}
    logger.info(f"  Top3 XGBRanker: Precision@3={xgb_p3:.3f}, NDCG@3={xgb_ndcg:.3f}")

    logger.info("Top3 LGBMRanker(lambdarank) 학습 중...")
    lgb_ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=      lgb_p.get("n_estimators", 300),
        max_depth=         lgb_p.get("max_depth", 6),
        learning_rate=     lgb_p.get("learning_rate", 0.1),
        subsample=         lgb_p.get("subsample", 1.0),
        colsample_bytree=  lgb_p.get("colsample_bytree", 1.0),
        num_leaves=        lgb_p.get("num_leaves", 31),
        min_child_samples= lgb_p.get("min_child_samples", 20),
        random_state=MODEL_CONFIG["random_state"], verbose=-1,
    )
    lgb_ranker.fit(X_train, y_train, group=train_groups)
    lgb_scores = lgb_ranker.predict(X_test)
    lgb_p3   = self._precision_at_3_score(df_test.assign(score=lgb_scores), "score")
    lgb_ndcg = self._eval_ndcg(df_test, lgb_scores, feature_cols)
    self.top3_ranker_models["LightGBM"] = lgb_ranker
    metrics["top3_LightGBM"] = {"precision_at_3": lgb_p3, "ndcg3": lgb_ndcg}
    logger.info(f"  Top3 LGBMRanker: Precision@3={lgb_p3:.3f}, NDCG@3={lgb_ndcg:.3f}")

    return metrics
```

- [ ] **Step 4: train()에서 호출 추가**

`train()` 메서드 내 `ranker_metrics = self._train_pairwise_rankers(...)` 줄 바로 뒤에:

```python
ranker_metrics    = self._train_pairwise_rankers(df_sorted, feature_cols, best_params)
ranker_top3_metrics = self._train_pairwise_rankers_top3(df_sorted, feature_cols, best_params)

all_metrics = {
    **win_metrics,
    **{f"top3_{k}": v for k, v in top3_metrics.items()},
    **ranker_metrics,
    **ranker_top3_metrics,   # ← 추가
}
```

- [ ] **Step 5: 테스트 통과 확인**

```
pytest tests/test_trainer.py -v
```

Expected: 전체 PASSED

- [ ] **Step 6: 커밋**

```
git add core/trainer.py tests/test_trainer.py
git commit -m "feat: _train_pairwise_rankers_top3 추가 - 연승 전용 XGBRanker·LGBMRanker"
```

---

## Task 4: predictor.py — _blend_group() 추출 + top3 블렌딩 적용

**Files:**
- Modify: `core/predictor.py`
- Test: `tests/test_predictor.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_predictor.py` 끝에 추가:

```python
def test_predictor_top3_probability_uses_ranker_when_available():
    """top3_ranker_models가 있으면 top3_probability가 ranker를 반영해야 한다."""
    import numpy as np, pandas as pd
    from core.predictor import RacePredictor
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.feature_columns = ["horse_age", "horse_win_rate"]
    # win ranker 없음, top3 ranker 있음
    trainer.predict_rank_score.return_value = None
    trainer.predict_proba.return_value = np.array([0.4, 0.3, 0.3])
    trainer.predict_proba_top3.return_value = np.array([0.5, 0.3, 0.2])
    trainer.predict_rank_score_top3.return_value = np.array([0.9, 0.4, 0.1])

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

    assert "top3_probability" in result.columns
    # ranker가 말A를 1위로 봤으므로 top3_prob도 말A가 가장 높아야 함
    top3_probs = result.set_index("horse_name")["top3_probability"]
    assert top3_probs["말A"] > top3_probs["말B"]
    assert top3_probs["말A"] > top3_probs["말C"]
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_predictor.py::test_predictor_top3_probability_uses_ranker_when_available -v
```

Expected: `FAILED` — `AttributeError: predict_rank_score_top3` 또는 assertion 실패

- [ ] **Step 3: predictor.py 수정**

`_softmax()` 함수 정의 바로 뒤, `RacePredictor` 클래스 선언 전에 모듈 수준 헬퍼 추가:

```python
def _blend_group(grp: pd.DataFrame, lr_col: str, clf_col: str) -> pd.Series:
    """경주 내 ranker 점수와 분류 확률을 LAMBDARANK_BLEND 비율로 블렌딩 후 softmax."""
    lr = grp[lr_col].values
    clf = grp[clf_col].values
    lr_range = lr.max() - lr.min()
    lr_norm = (lr - lr.min()) / lr_range if lr_range > 1e-9 else np.full(len(lr), 0.5)
    blended = LAMBDARANK_BLEND * lr_norm + (1 - LAMBDARANK_BLEND) * clf
    return pd.Series(_softmax(blended), index=grp.index)
```

`predict()` 메서드 전체를 아래로 교체:

```python
def predict(self, df: pd.DataFrame) -> pd.DataFrame:
    features = self.trainer.feature_columns
    available = [c for c in features if c in df.columns]
    X = df[available].fillna(0)

    rank_scores      = self.trainer.predict_rank_score(X)
    rank_scores_top3 = self.trainer.predict_rank_score_top3(X)
    clf_probs        = self.trainer.predict_proba(X)
    clf_probs_top3   = self.trainer.predict_proba_top3(X)

    if not isinstance(rank_scores, np.ndarray):
        rank_scores = None
    if not isinstance(rank_scores_top3, np.ndarray):
        rank_scores_top3 = None
    if not isinstance(clf_probs_top3, np.ndarray):
        clf_probs_top3 = None

    result = df.copy()

    # ── 단승(win_probability) ──────────────────────────────────────
    if rank_scores is not None and "race_id" in result.columns:
        result["_lr"]  = rank_scores
        result["_clf"] = clf_probs
        result["win_probability"] = result.groupby("race_id", group_keys=False).apply(
            lambda grp: _blend_group(grp, "_lr", "_clf")
        )
        result = result.drop(columns=["_lr", "_clf"])
        logger.debug(f"앙상블 블렌딩: LambdaRank {LAMBDARANK_BLEND:.0%} + 분류모델 {1-LAMBDARANK_BLEND:.0%}")
    elif rank_scores is not None:
        result["win_probability"] = _softmax(rank_scores)
    else:
        result["win_probability"] = clf_probs

    # ── 연승(top3_probability) ────────────────────────────────────
    if rank_scores_top3 is not None and clf_probs_top3 is not None and "race_id" in result.columns:
        result["_top3_lr"]  = rank_scores_top3
        result["_top3_clf"] = clf_probs_top3
        result["top3_probability"] = result.groupby("race_id", group_keys=False).apply(
            lambda grp: _blend_group(grp, "_top3_lr", "_top3_clf")
        )
        result = result.drop(columns=["_top3_lr", "_top3_clf"])
    elif clf_probs_top3 is not None:
        result["top3_probability"] = clf_probs_top3
    else:
        result["top3_probability"] = result["win_probability"]

    result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
    result["predicted_rank"] = range(1, len(result) + 1)

    logger.info(f"예측 완료: {len(result)}마리, 최고 확률={result['win_probability'].max():.3f}")
    return result
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_predictor.py -v
```

Expected: 전체 PASSED

- [ ] **Step 5: 전체 테스트 통과 확인**

```
pytest tests/ -v --ignore=tests/test_database.py --ignore=tests/test_collectors.py --ignore=tests/test_scheduler.py
```

Expected: 전체 PASSED (DB 연결 불필요 테스트 기준)

- [ ] **Step 6: 커밋**

```
git add core/predictor.py tests/test_predictor.py
git commit -m "feat: predictor top3_probability에 ranker 블렌딩 적용"
```

---

## Task 5: 재학습 + 백테스트

**Files:** 없음 (실행만)

- [ ] **Step 1: 모델 재학습**

```
python -X utf8 main.py --mode train --start-date 2024-01-01
```

로그에서 확인:
```
Top3 XGBRanker: Precision@3=... NDCG@3=...
Top3 LGBMRanker: Precision@3=... NDCG@3=...
```

두 줄이 출력되면 정상.

- [ ] **Step 2: 백테스트 실행**

```python
# python -X utf8 -c "..." 로 실행 (아래 스크립트)
import pandas as pd, numpy as np
from Utils.database import get_client
from core.model_manager import ModelManager
from core.trainer import RaceModelTrainer
from core.preprocessor import RacePreprocessor
from core.predictor import RacePredictor

trainer = RaceModelTrainer()
preprocessor = RacePreprocessor()
ModelManager().load(trainer, preprocessor)
predictor = RacePredictor(trainer, preprocessor)
client = get_client()

dates = [
    '2026-01-31','2026-01-25','2026-01-24',
    '2025-06-29','2025-06-28','2025-06-27',
    '2025-06-22','2025-06-21','2025-06-20',
    '2025-06-15','2025-06-07','2025-06-06','2025-06-01',
]

win_hits, top3_hits, total = 0, 0, 0
rows = []

for date in dates:
    preds = predictor.predict_from_db(date)
    if preds.empty: continue
    res = client.table('race_entries').select('race_id,horse_id,final_rank') \
        .eq('race_date', date).not_.is_('final_rank','null').execute()
    if not res.data: continue
    actuals = pd.DataFrame(res.data)
    df = preds.merge(actuals, on=['race_id','horse_id'], how='inner', suffixes=('_pred',''))
    if df.empty: continue
    df['actual_top3'] = (df['final_rank'] <= 3).astype(int)
    for race_id, grp in df.groupby('race_id'):
        if len(grp) < 2: continue
        win_pick  = grp.loc[grp['win_probability'].idxmax(), 'final_rank']
        top3_pick = grp.loc[grp['top3_probability'].idxmax(), 'final_rank']
        win_hits  += (win_pick == 1)
        top3_hits += (top3_pick <= 3)
        total += 1
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
print(f'총 {total}경주')
print(f'단승 Precision@1: {win_hits/total:.1%}')
print(f'연승 Precision@1: {top3_hits/total:.1%}  (기존 65.1%)')
print()
for thr in [0.60, 0.65, 0.70, 0.75]:
    s = all_df[all_df['top3_probability'] >= thr]
    if len(s) == 0: continue
    print(f'top3_prob >= {thr}: {len(s)}건 / {s["race_id"].nunique()}경주 → 적중 {s["actual_top3"].mean():.1%}')
```

- [ ] **Step 3: 결과 기록**

개선됐으면 완료. 기준:
- 연승 Precision@1 > 65.1%
- top3_prob >= 0.65 적중률 > 66.0%

개선되지 않았으면 `LAMBDARANK_BLEND` 값 조정 후 재학습 (config.py에서 현재 값 확인 후 0.3→0.5 또는 0.5→0.7 시도).

- [ ] **Step 4: 커밋**

```
git add -A
git commit -m "feat: 연승 Pairwise Ranker 추가 - 재학습 완료"
```
