# Pairwise Ranking 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** XGBoost·LightGBM을 binary classifier에서 pairwise ranker로 전환하고, 평가 지표를 AUC → Precision@1·NDCG@3로 교체한다.

**Architecture:** `_train_classifiers`에서 XGB·LGB를 제거하고 `_train_pairwise_rankers`로 분리한다. 두 ranker는 race-level 그룹 정보를 사용하여 경주 내 상대 순위를 직접 학습한다. `predict_rank_score`는 XGBRanker·LGBMRanker·LambdaRank 세 모델의 점수를 min-max 정규화 후 평균한다.

**Tech Stack:** `xgboost.XGBRanker`, `lightgbm.LGBMRanker`, `sklearn.metrics.ndcg_score`, pytest

---

## 파일 목록

| 파일 | 변경 |
|---|---|
| `core/trainer.py` | XGB·LGB classifier → ranker 전환, `_train_pairwise_rankers` 추가, Precision@1 추가 |
| `tests/test_trainer.py` | 기존 AUC 테스트 수정, Precision@1·NDCG@3 테스트 추가 |

---

## Task 1: Precision@1 헬퍼 메서드 추가 및 테스트

**Files:**
- Modify: `core/trainer.py`
- Modify: `tests/test_trainer.py`

### 배경

현재 평가 지표: AUC, Accuracy (binary 기준)  
필요한 지표: **Precision@1** — 각 경주에서 1위로 예측한 말이 실제 1위인 비율  
`Precision@1 = (예측 1위 == 실제 1위인 경주 수) / 전체 경주 수`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_trainer.py` 끝에 추가:

```python
def test_precision_at_1_perfect():
    """모든 경주에서 1위를 정확히 예측하면 1.0"""
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":   ["r1", "r1", "r1", "r2", "r2", "r2"],
        "final_rank": [1, 2, 3, 1, 2, 3],
        "score":      [0.9, 0.5, 0.2, 0.8, 0.4, 0.1],
    })
    result = trainer._precision_at_1_score(df, score_col="score")
    assert result == pytest.approx(1.0)


def test_precision_at_1_zero():
    """모든 경주에서 1위를 틀리면 0.0"""
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":   ["r1", "r1", "r1", "r2", "r2", "r2"],
        "final_rank": [1, 2, 3, 1, 2, 3],
        "score":      [0.1, 0.5, 0.9, 0.1, 0.4, 0.8],  # 3위 말이 가장 높음
    })
    result = trainer._precision_at_1_score(df, score_col="score")
    assert result == pytest.approx(0.0)


def test_precision_at_1_half():
    """절반 적중하면 0.5"""
    trainer = RaceModelTrainer()
    df = pd.DataFrame({
        "race_id":   ["r1", "r1", "r2", "r2"],
        "final_rank": [1, 2, 1, 2],
        "score":      [0.9, 0.1, 0.1, 0.9],  # r1 맞음, r2 틀림
    })
    result = trainer._precision_at_1_score(df, score_col="score")
    assert result == pytest.approx(0.5)
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd "F:\Project\경마 자동 투자 봇"
python -m pytest tests/test_trainer.py::test_precision_at_1_perfect -v
```

예상: `AttributeError: 'RaceModelTrainer' object has no attribute '_precision_at_1_score'`

- [ ] **Step 3: `_precision_at_1_score` 구현**

`core/trainer.py`의 `_load_best_params` 메서드 **앞에** 추가:

```python
def _precision_at_1_score(self, df: pd.DataFrame, score_col: str = "score") -> float:
    """
    각 경주에서 score가 가장 높은 말이 실제 final_rank==1인 비율.
    df 컬럼 필수: race_id, final_rank, {score_col}
    """
    hits = 0
    races = 0
    for _, grp in df.groupby("race_id"):
        if grp.empty:
            continue
        pred_winner_idx = grp[score_col].idxmax()
        actual_rank = grp.loc[pred_winner_idx, "final_rank"]
        if actual_rank == 1:
            hits += 1
        races += 1
    return hits / races if races > 0 else 0.0
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
python -m pytest tests/test_trainer.py::test_precision_at_1_perfect tests/test_trainer.py::test_precision_at_1_zero tests/test_trainer.py::test_precision_at_1_half -v
```

예상: 3개 모두 PASSED

- [ ] **Step 5: 커밋**

```bash
git add core/trainer.py tests/test_trainer.py
git commit -m "feat: add _precision_at_1_score evaluation metric"
```

---

## Task 2: XGBoost·LightGBM → Pairwise Ranker 전환

**Files:**
- Modify: `core/trainer.py`

### 배경

현재 `_train_classifiers`는 XGBClassifier·LGBMClassifier를 행(row) 단위 이진분류로 학습한다.  
문제: 같은 경주의 말들이 독립 샘플로 취급됨 → 경주 내 상대 경쟁 구조 무시.

변경 후:
- `XGBRanker(objective='rank:pairwise')` — 같은 경주 내 쌍(pair)을 비교하여 순위를 학습
- `LGBMRanker(objective='lambdarank')` — LambdaRank 목적함수로 NDCG 최적화
- 두 모델 모두 race-level 그룹 정보(`group=`) 사용

관련성 레이블(relevance label): 1위=4, 2위=3, 3위=2, 나머지=1

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_trainer.py`에 추가:

```python
def test_xgb_lgb_are_rankers():
    """학습 후 XGBoost·LightGBM 모델이 ranker(predict_proba 없음)여야 한다."""
    from xgboost import XGBRanker
    from lightgbm import LGBMRanker

    np.random.seed(0)
    n_races, horses_per = 30, 8
    n = n_races * horses_per
    race_ids = [f"r{i}" for i in range(n_races) for _ in range(horses_per)]
    final_ranks = [j + 1 for _ in range(n_races) for j in range(horses_per)]

    df = pd.DataFrame({
        "race_id": race_ids,
        "race_date": pd.to_datetime(
            [f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
        ),
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

    trainer = RaceModelTrainer()
    trainer.train(df)

    assert isinstance(trainer.models["XGBoost"], XGBRanker), \
        "XGBoost 모델이 XGBRanker가 아님"
    assert isinstance(trainer.models["LightGBM"], LGBMRanker), \
        "LightGBM 모델이 LGBMRanker가 아님"
    assert not hasattr(trainer.models["XGBoost"], "predict_proba"), \
        "XGBRanker에 predict_proba가 있으면 안 됨"


def test_ranker_scores_winner_higher():
    """predict_rank_score에서 승자의 점수가 평균보다 높아야 한다."""
    np.random.seed(1)
    n_races, horses_per = 20, 8
    n = n_races * horses_per
    race_ids = [f"r{i}" for i in range(n_races) for _ in range(horses_per)]
    final_ranks = [j + 1 for _ in range(n_races) for j in range(horses_per)]

    # 승자는 jockey_win_rate·horse_win_rate를 높게 설정
    jockey_wr = []
    horse_wr = []
    for rank in final_ranks:
        base = 0.3 - rank * 0.02
        jockey_wr.append(max(0.0, base + np.random.uniform(-0.01, 0.01)))
        horse_wr.append(max(0.0, base + np.random.uniform(-0.01, 0.01)))

    df = pd.DataFrame({
        "race_id": race_ids,
        "race_date": pd.to_datetime(
            [f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
        ),
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
        "prev_5_avg_rank": [rank * 0.8 + np.random.uniform(0, 0.5)
                            for rank in final_ranks],
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
    assert scores is not None

    winner_scores = scores[df["final_rank"] == 1]
    others_scores = scores[df["final_rank"] > 1]
    assert winner_scores.mean() > others_scores.mean(), \
        "승자의 평균 점수가 패자보다 낮음"
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
python -m pytest tests/test_trainer.py::test_xgb_lgb_are_rankers -v
```

예상: `AssertionError: XGBoost 모델이 XGBRanker가 아님`

- [ ] **Step 3: `core/trainer.py` 수정**

**(3-a) import 수정** — 파일 상단:

```python
# 기존
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRanker

# 변경
from xgboost import XGBRanker
from lightgbm import LGBMRanker
```

**(3-b) `train` 메서드** — `_train_classifiers` 두 번 호출 부분 수정:

```python
def train(self, df: pd.DataFrame) -> dict:
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    self.feature_columns = feature_cols
    logger.info(f"학습 특성 {len(feature_cols)}개")

    df_sorted = df.sort_values("race_date")
    split_idx = int(len(df_sorted) * (1 - MODEL_CONFIG["test_size"]))

    X_train = df_sorted.iloc[:split_idx][feature_cols].fillna(0)
    X_test  = df_sorted.iloc[split_idx:][feature_cols].fillna(0)

    y_train_win  = df_sorted.iloc[:split_idx]["is_winner"]
    y_test_win   = df_sorted.iloc[split_idx:]["is_winner"]
    y_train_top3 = self._top3_target(df_sorted.iloc[:split_idx], y_train_win)
    y_test_top3  = self._top3_target(df_sorted.iloc[split_idx:], y_test_win)

    X_train_s = self.scaler.fit_transform(X_train)
    X_test_s  = self.scaler.transform(X_test)

    best_params = self._load_best_params()

    # RandomForest·LogisticRegression만 분류 학습
    logger.info("=== 단승(1등) 분류 모델 학습 (RF·LR) ===")
    win_metrics = self._train_classifiers(
        X_train, X_test, y_train_win, y_test_win,
        X_train_s, X_test_s, target="winner", best_params=best_params,
    )

    logger.info("=== 연승(3등 이내) 분류 모델 학습 (RF·LR) ===")
    top3_metrics = self._train_classifiers(
        X_train, X_test, y_train_top3, y_test_top3,
        X_train_s, X_test_s, target="top3", best_params=best_params,
    )

    # XGBoost·LightGBM은 pairwise ranker로 학습
    ranker_metrics = self._train_pairwise_rankers(df_sorted, feature_cols, best_params)

    all_metrics = {
        **win_metrics,
        **{f"top3_{k}": v for k, v in top3_metrics.items()},
        **ranker_metrics,
    }
    return {"models": self.models, "metrics": all_metrics}
```

**(3-c) `_train_classifiers` 수정** — XGB·LGB 제거, RF·LR만 남김:

```python
def _train_classifiers(
    self,
    X_train, X_test, y_train, y_test,
    X_train_s, X_test_s,
    target: str,
    best_params: dict,
) -> dict:
    model_defs = {
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=MODEL_CONFIG["random_state"],
                class_weight="balanced",
            ),
            X_train, X_test,
        ),
        "LogisticRegression": (
            LogisticRegression(
                random_state=MODEL_CONFIG["random_state"],
                class_weight="balanced", max_iter=1000,
            ),
            X_train_s, X_test_s,
        ),
    }

    results, metrics = {}, {}
    for name, (model, Xtr, Xte) in model_defs.items():
        logger.info(f"  {name} 학습 중...")
        model.fit(Xtr, y_train)
        y_prob = model.predict_proba(Xte)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.0
        acc = accuracy_score(y_test, y_pred)
        results[name] = model
        metrics[name] = {"auc": auc, "accuracy": acc}
        logger.info(f"    {name}: AUC={auc:.3f}, Acc={acc:.3f}")

    if target == "winner":
        self.models.update(results)
    else:
        self.top3_models.update(results)
    return metrics
```

**(3-d) `_train_pairwise_rankers` 추가** — `_train_ranker` 메서드 **대체**:

```python
def _train_pairwise_rankers(
    self, df_sorted: pd.DataFrame, feature_cols: list, best_params: dict
) -> dict:
    """XGBRanker(pairwise)·LGBMRanker(lambdarank) 학습. race-level 그룹 사용."""
    if "race_id" not in df_sorted.columns or "final_rank" not in df_sorted.columns:
        logger.warning("Ranker 학습 불가: race_id 또는 final_rank 없음")
        return {}

    # race 단위 시간 기준 split
    race_dates = df_sorted.groupby("race_id")["race_date"].first().sort_values()
    split_idx  = int(len(race_dates) * (1 - MODEL_CONFIG["test_size"]))
    train_races = set(race_dates.index[:split_idx])
    test_races  = set(race_dates.index[split_idx:])

    df_train = df_sorted[df_sorted["race_id"].isin(train_races)].sort_values("race_id")
    df_test  = df_sorted[df_sorted["race_id"].isin(test_races)].sort_values("race_id")

    if df_train.empty or df_test.empty:
        logger.warning("Ranker: train/test split 후 데이터 부족")
        return {}

    def make_label(rank: int) -> int:
        if rank == 1: return 4
        if rank == 2: return 3
        if rank == 3: return 2
        return 1

    X_train = df_train[feature_cols].fillna(0)
    X_test  = df_test[feature_cols].fillna(0)
    y_train = df_train["final_rank"].apply(make_label)
    y_test  = df_test["final_rank"].apply(make_label)
    train_groups = df_train.groupby("race_id", sort=False).size().values
    test_groups  = df_test.groupby("race_id", sort=False).size().values

    lgb_p = best_params.get("LightGBM", {})
    metrics = {}

    # ── XGBRanker (pairwise) ──────────────────────────────
    logger.info("XGBRanker(pairwise) 학습 중...")
    xgb_p = best_params.get("XGBoost", {})
    xgb_ranker = XGBRanker(
        objective="rank:pairwise",
        n_estimators=    xgb_p.get("n_estimators", 200),
        max_depth=       xgb_p.get("max_depth", 6),
        learning_rate=   xgb_p.get("learning_rate", 0.1),
        subsample=       xgb_p.get("subsample", 1.0),
        colsample_bytree=xgb_p.get("colsample_bytree", 1.0),
        random_state=MODEL_CONFIG["random_state"],
        verbosity=0,
    )
    xgb_ranker.fit(X_train, y_train, qid=np.repeat(
        np.arange(len(train_groups)), train_groups
    ))
    xgb_scores = xgb_ranker.predict(X_test)
    xgb_p1 = self._precision_at_1_score(
        df_test.assign(score=xgb_scores), score_col="score"
    )
    xgb_ndcg = self._eval_ndcg(df_test, xgb_scores, feature_cols)
    self.models["XGBoost"] = xgb_ranker
    metrics["XGBoost"] = {"precision_at_1": xgb_p1, "ndcg3": xgb_ndcg}
    logger.info(f"  XGBRanker: Precision@1={xgb_p1:.3f}, NDCG@3={xgb_ndcg:.3f}")

    # ── LGBMRanker (lambdarank) ───────────────────────────
    logger.info("LGBMRanker(lambdarank) 학습 중...")
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
    lgb_p1 = self._precision_at_1_score(
        df_test.assign(score=lgb_scores), score_col="score"
    )
    lgb_ndcg = self._eval_ndcg(df_test, lgb_scores, feature_cols)
    self.models["LightGBM"] = lgb_ranker
    metrics["LightGBM"] = {"precision_at_1": lgb_p1, "ndcg3": lgb_ndcg}
    logger.info(f"  LGBMRanker: Precision@1={lgb_p1:.3f}, NDCG@3={lgb_ndcg:.3f}")

    return metrics


def _eval_ndcg(
    self, df_test: pd.DataFrame, scores: np.ndarray, feature_cols: list
) -> float:
    """경주별 NDCG@3 평균."""
    def make_label(rank):
        if rank == 1: return 4
        if rank == 2: return 3
        if rank == 3: return 2
        return 1

    df_eval = df_test.copy()
    df_eval["_score"] = scores
    ndcg_scores = []
    for _, grp in df_eval.groupby("race_id"):
        if len(grp) < 2:
            continue
        true_labels = grp["final_rank"].apply(make_label).values.reshape(1, -1)
        pred_scores = grp["_score"].values.reshape(1, -1)
        k = min(3, len(grp))
        ndcg_scores.append(ndcg_score(true_labels, pred_scores, k=k))
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
```

**(3-e) `_train_ranker` 메서드 삭제** — 전체 메서드를 파일에서 제거 (위의 `_train_pairwise_rankers`가 대체)

**(3-f) `predict_rank_score` 수정** — XGBoost·LightGBM 점수도 통합:

```python
def predict_rank_score(self, X: pd.DataFrame) -> np.ndarray | None:
    """XGBRanker·LGBMRanker 점수를 min-max 정규화 후 평균."""
    X = X.fillna(0)
    ranker_names = ["XGBoost", "LightGBM", "LambdaRank"]
    scores_list = []

    for name in ranker_names:
        ranker = self.models.get(name)
        if ranker is None:
            continue
        raw = ranker.predict(X)
        s_min, s_max = raw.min(), raw.max()
        if s_max > s_min:
            normalized = (raw - s_min) / (s_max - s_min)
        else:
            normalized = np.full(len(raw), 0.5)
        scores_list.append(normalized)

    if not scores_list:
        return None
    return np.mean(scores_list, axis=0)
```

**(3-g) `_ensemble_proba` 수정** — XGBoost·LightGBM 건너뜀 (이제 ranker):

```python
def _ensemble_proba(self, model_dict: dict, X: pd.DataFrame) -> np.ndarray:
    X = X.fillna(0)
    X_scaled = self.scaler.transform(X)
    # ranker 모델은 predict_proba 없으므로 건너뜀
    _RANKER_KEYS = {"XGBoost", "LightGBM", "LambdaRank"}
    probas = []
    for name, model in model_dict.items():
        if name in _RANKER_KEYS:
            continue
        Xin = X_scaled if name == "LogisticRegression" else X
        probas.append(model.predict_proba(Xin)[:, 1])
    if not probas:
        return np.zeros(len(X))
    return np.mean(probas, axis=0)
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
python -m pytest tests/test_trainer.py::test_xgb_lgb_are_rankers tests/test_trainer.py::test_ranker_scores_winner_higher -v
```

예상: 2개 모두 PASSED

- [ ] **Step 5: 기존 테스트 수정**

`test_trainer_returns_metrics`는 `"auc"` 키를 가정하지만 XGB·LGB는 이제 `"precision_at_1"` 키를 가짐. 수정:

```python
def test_trainer_returns_metrics(preprocessed_df):
    # preprocessed_df에는 race_id·final_rank 없어서 ranker가 skip됨
    # RF·LR 메트릭만 검증
    trainer = RaceModelTrainer()
    result = trainer.train(preprocessed_df)
    assert "metrics" in result
    for name, metrics in result["metrics"].items():
        if "precision_at_1" in metrics:
            assert 0 <= metrics["precision_at_1"] <= 1
        elif "auc" in metrics:
            assert 0 <= metrics["auc"] <= 1
```

- [ ] **Step 6: 전체 테스트 실행**

```bash
python -m pytest tests/test_trainer.py -v
```

예상: 전체 PASSED (기존 테스트 포함)

- [ ] **Step 7: 커밋**

```bash
git add core/trainer.py tests/test_trainer.py
git commit -m "feat: convert XGBoost/LightGBM to pairwise rankers, add Precision@1 metric"
```

---

## Task 3: 재학습 및 비교

**Files:**
- 실행만 (코드 변경 없음)

- [ ] **Step 1: 재학습 실행**

```bash
cd "F:\Project\경마 자동 투자 봇"
python main.py --mode train --start-date 2024-01-05 --end-date 2026-06-05
```

예상 출력:
```
=== 단승(1등) 분류 모델 학습 (RF·LR) ===
  RandomForest: AUC=0.8xx, Acc=0.8xx
  LogisticRegression: AUC=0.8xx, Acc=0.8xx
XGBRanker(pairwise) 학습 중...
  XGBRanker: Precision@1=0.2xx, NDCG@3=0.7xx
LGBMRanker(lambdarank) 학습 중...
  LGBMRanker: Precision@1=0.2xx, NDCG@3=0.7xx
모델 저장: models/model_YYYYMMDD_HHMMSS.pkl
```

- [ ] **Step 2: 오늘 데이터로 예측 후 적중률 확인**

```bash
python -c "
import sys; sys.stdout.reconfigure(encoding='utf-8')
import joblib
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from config import MODEL_DIR
from pathlib import Path

# 방금 학습한 최신 모델 로드
pkl_path = sorted(MODEL_DIR.glob('model_*.pkl'))[-1]
data = joblib.load(pkl_path)
preprocessor = RacePreprocessor()
trainer = RaceModelTrainer()
trainer.models = data['models']
trainer.top3_models = data.get('top3_models', {})
trainer.scaler = data['scaler']
trainer.feature_columns = data['feature_columns']
preprocessor.label_encoders = data['label_encoders']
print(f'모델: {pkl_path.name}')

predictor = RacePredictor(trainer, preprocessor)
result_df = predictor.predict_from_db('2026-06-07')
for race_id in sorted(result_df['race_id'].unique()):
    race = result_df[result_df['race_id'] == race_id]
    top = race.sort_values('win_probability', ascending=False).iloc[0]
    print(f'{race_id} | {top.get(\"horse_name\",\"?\")} | {top[\"win_probability\"]:.1%}')
"
```

- [ ] **Step 3: 커밋 (재학습 결과는 모델 파일이므로 .gitignore 대상 — 코드만 커밋)**

```bash
git status
git add -u
git commit -m "refactor: remove _train_ranker, unify pairwise rankers in _train_pairwise_rankers"
```

---

## 자가 검토

**Spec 커버리지:**
- [x] XGBoost → `rank:pairwise` 전환 (Task 2)
- [x] LightGBM → `lambdarank` 전환 (Task 2)
- [x] Precision@1 평가 지표 추가 (Task 1, 2)
- [x] NDCG@3 평가 지표 유지 (Task 2)
- [x] 경주별 그룹 정보 사용 (Task 2, `_train_pairwise_rankers`)
- [x] `predict_rank_score` 통합 (Task 2 Step 3-f)
- [x] `_ensemble_proba`에서 ranker 제외 (Task 2 Step 3-g)

**타입 일관성:**
- `_precision_at_1_score(df, score_col)` → Task 1에서 정의, Task 2에서 `df_test.assign(score=...)` 형태로 호출 — 일치
- `_eval_ndcg(df_test, scores, feature_cols)` → Task 2에서 정의·사용 — 일치
- `predict_rank_score` 반환: `np.ndarray | None` — predictor의 `if not isinstance(rank_scores, np.ndarray)` 체크와 호환
