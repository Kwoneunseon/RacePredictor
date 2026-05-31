# Prediction Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 피처 5개 추가(경마장별 승률, 기수-말 조합 승률, 순위 트렌드 등) + LambdaRank 랭킹 모델로 예측 정확도 개선

**Architecture:** preprocessor에 학습용 피처 계산 추가, predictor의 predict_from_db에 예측용 DB 쿼리 추가, trainer에 LGBMRanker 학습 추가. 기존 분류 모델은 유지하고 LambdaRank를 앙상블에 추가해 predict 시 rank score를 win_probability로 사용.

**Tech Stack:** LightGBM(LGBMRanker), pandas, numpy, Supabase

---

### Task 1: config.py — 피처 5개 추가

**Files:**
- Modify: `config.py:48-58`

- [ ] **Step 1: FEATURE_COLUMNS에 신규 피처 추가**

```python
# config.py:48
FEATURE_COLUMNS = [
    "horse_age", "is_male", "horse_class_num", "race_distance", "total_horses",
    "horse_weight", "race_grade_enc",
    "prev_total_races", "prev_5_avg_rank", "prev_total_avg_rank",
    "jockey_win_rate", "trainer_win_rate", "horse_win_rate",
    "horse_top3_rate", "experience_score", "recent_form",
    "horse_class_vs_mean", "win_rate_vs_mean",
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    "training_grade_enc",
    "days_since_last_race", "horse_weight_diff", "horse_win_rate_at_distance",
    # 신규
    "horse_track_win_rate", "jockey_horse_win_rate", "recent_rank_trend",
    "jockey_track_win_rate", "weight_trend",
]
```

- [ ] **Step 2: 커밋**

```bash
git add config.py
git commit -m "feat: FEATURE_COLUMNS에 신규 피처 5개 추가"
```

---

### Task 2: preprocessor.py — 학습용 피처 계산 추가

**Files:**
- Modify: `core/preprocessor.py`

- [ ] **Step 1: `_add_new_features` 메서드 추가**

`_add_historical_performance_features` 메서드 바로 아래에 추가:

```python
def _add_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
    needed = ["race_date", "horse_id", "final_rank", "meet_code", "jk_no", "horse_weight"]
    if not all(c in df.columns for c in needed):
        for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                    "recent_rank_trend", "jockey_track_win_rate", "weight_trend"]:
            if col not in df.columns:
                df[col] = 0.0
        return df

    df = df.copy()
    df["_rdt"] = pd.to_datetime(df["race_date"])
    df["_is_win"] = (df["final_rank"] == 1).astype(float)

    # horse_track_win_rate: 말의 경마장별 누적 승률 (당일 경주 제외)
    df = df.sort_values(["horse_id", "meet_code", "_rdt"])
    df["_cumwins_track"] = (
        df.groupby(["horse_id", "meet_code"])["_is_win"].cumsum() - df["_is_win"]
    )
    df["_cumraces_track"] = df.groupby(["horse_id", "meet_code"]).cumcount()
    df["horse_track_win_rate"] = df["_cumwins_track"] / (df["_cumraces_track"] + 1)

    # jockey_horse_win_rate: 기수-말 조합 누적 승률
    df = df.sort_values(["jk_no", "horse_id", "_rdt"])
    df["_cumwins_jh"] = (
        df.groupby(["jk_no", "horse_id"])["_is_win"].cumsum() - df["_is_win"]
    )
    df["_cumraces_jh"] = df.groupby(["jk_no", "horse_id"]).cumcount()
    df["jockey_horse_win_rate"] = df["_cumwins_jh"] / (df["_cumraces_jh"] + 1)

    # jockey_track_win_rate: 기수의 경마장별 누적 승률
    df = df.sort_values(["jk_no", "meet_code", "_rdt"])
    df["_cumwins_jt"] = (
        df.groupby(["jk_no", "meet_code"])["_is_win"].cumsum() - df["_is_win"]
    )
    df["_cumraces_jt"] = df.groupby(["jk_no", "meet_code"]).cumcount()
    df["jockey_track_win_rate"] = df["_cumwins_jt"] / (df["_cumraces_jt"] + 1)

    # recent_rank_trend: 최근 3경주 순위 기울기 (음수=개선)
    df = df.sort_values(["horse_id", "_rdt"])

    def _slope(series):
        arr = series.values
        out = np.zeros(len(arr))
        for i in range(len(arr)):
            past = arr[max(0, i - 3):i]
            if len(past) >= 2:
                out[i] = float(np.polyfit(np.arange(len(past)), past, 1)[0])
        return pd.Series(out, index=series.index)

    df["recent_rank_trend"] = df.groupby("horse_id")["final_rank"].transform(_slope)

    # weight_trend: 최근 3경주 마체중 증감 평균
    df["_wdiff"] = df.groupby("horse_id")["horse_weight"].diff()

    def _wt_mean(series):
        arr = series.fillna(0).values
        out = np.zeros(len(arr))
        for i in range(len(arr)):
            past = arr[max(0, i - 3):i]
            out[i] = float(past.mean()) if len(past) > 0 else 0.0
        return pd.Series(out, index=series.index)

    df["weight_trend"] = df.groupby("horse_id")["_wdiff"].transform(_wt_mean)

    drop_cols = [
        "_rdt", "_is_win",
        "_cumwins_track", "_cumraces_track",
        "_cumwins_jh", "_cumraces_jh",
        "_cumwins_jt", "_cumraces_jt",
        "_wdiff",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df
```

- [ ] **Step 2: `fit_transform`에 `_add_new_features` 호출 추가**

`_add_historical_performance_features` 호출 바로 다음 줄에 추가:

```python
def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = self._normalize_raw_fields(df)
    df = self._add_target(df)
    df = self._add_derived_features(df)
    df = self._add_rolling_recent_rates(df)
    df = self._add_historical_performance_features(df)
    df = self._add_new_features(df)          # ← 추가
    if "training_grade_enc" not in df.columns:
        df["training_grade_enc"] = 0.0
    df = self._add_race_relative_features(df)
    df = self._encode_categoricals(df, fit=True)
    self._fitted = True
    return df
```

- [ ] **Step 3: `transform`에 신규 피처 기본값 추가**

기존 기본값 블록 아래에 추가:

```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = self._normalize_raw_fields(df)
    df = self._add_derived_features(df)
    # 기존 기본값들
    if "jockey_recent_win_rate" not in df.columns:
        df["jockey_recent_win_rate"] = 0.0
    if "trainer_recent_win_rate" not in df.columns:
        df["trainer_recent_win_rate"] = 0.0
    if "training_grade_enc" not in df.columns:
        df["training_grade_enc"] = 0.0
    if "days_since_last_race" not in df.columns:
        df["days_since_last_race"] = 60
    if "horse_weight_diff" not in df.columns:
        df["horse_weight_diff"] = 0.0
    if "horse_win_rate_at_distance" not in df.columns:
        df["horse_win_rate_at_distance"] = 0.0
    # 신규 기본값
    for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                "recent_rank_trend", "jockey_track_win_rate", "weight_trend"]:
        if col not in df.columns:
            df[col] = 0.0
    df = self._add_race_relative_features(df)
    df = self._encode_categoricals(df, fit=False)
    return df
```

- [ ] **Step 4: 커밋**

```bash
git add core/preprocessor.py
git commit -m "feat: preprocessor에 신규 피처 5개 계산 추가"
```

---

### Task 3: predictor.py — 예측 시 신규 피처 DB 쿼리 추가

**Files:**
- Modify: `core/predictor.py`

- [ ] **Step 1: `predict_from_db` 내 `horse_win_rate_at_distance` 쿼리 블록 바로 다음에 추가**

`horse_win_rate_at_distance` 블록 끝(`df["horse_win_rate_at_distance"] = df["horse_win_rate_at_distance"].fillna(0.0)`) 다음에 삽입:

```python
        # 신규 피처: horse_track_win_rate, jockey_horse_win_rate,
        #           jockey_track_win_rate, recent_rank_trend, weight_trend
        if horse_ids and "meet_code" in df.columns:
            combo_res = client.table("race_entries")\
                .select("horse_id, jk_no, final_rank, race_date, horse_weight, races(meet_code)")\
                .in_("horse_id", horse_ids)\
                .lt("race_date", race_date)\
                .gt("final_rank", 0)\
                .execute()
            if combo_res.data:
                hist = pd.DataFrame([{
                    "horse_id": r["horse_id"],
                    "jk_no": r.get("jk_no"),
                    "final_rank": r["final_rank"],
                    "race_date": r["race_date"],
                    "horse_weight": r.get("horse_weight") or 0,
                    "meet_code": (r.get("races") or {}).get("meet_code"),
                } for r in combo_res.data])
                hist["_is_win"] = (hist["final_rank"] == 1).astype(float)

                # horse_track_win_rate
                ht = hist.groupby(["horse_id", "meet_code"]).agg(
                    _w=("_is_win", "sum"), _r=("_is_win", "count")
                ).reset_index()
                ht["horse_track_win_rate"] = ht["_w"] / (ht["_r"] + 1)
                df = df.merge(
                    ht[["horse_id", "meet_code", "horse_track_win_rate"]],
                    on=["horse_id", "meet_code"], how="left"
                )
                df["horse_track_win_rate"] = df["horse_track_win_rate"].fillna(0.0)

                # jockey_horse_win_rate
                jh = hist.groupby(["horse_id", "jk_no"]).agg(
                    _w=("_is_win", "sum"), _r=("_is_win", "count")
                ).reset_index()
                jh["jockey_horse_win_rate"] = jh["_w"] / (jh["_r"] + 1)
                if "jk_no" in df.columns:
                    df = df.merge(
                        jh[["horse_id", "jk_no", "jockey_horse_win_rate"]],
                        on=["horse_id", "jk_no"], how="left"
                    )
                df["jockey_horse_win_rate"] = df.get("jockey_horse_win_rate", pd.Series(0.0, index=df.index)).fillna(0.0)

                # jockey_track_win_rate
                jt = hist.groupby(["jk_no", "meet_code"]).agg(
                    _w=("_is_win", "sum"), _r=("_is_win", "count")
                ).reset_index()
                jt["jockey_track_win_rate"] = jt["_w"] / (jt["_r"] + 1)
                if "jk_no" in df.columns:
                    df = df.merge(
                        jt[["jk_no", "meet_code", "jockey_track_win_rate"]],
                        on=["jk_no", "meet_code"], how="left"
                    )
                df["jockey_track_win_rate"] = df.get("jockey_track_win_rate", pd.Series(0.0, index=df.index)).fillna(0.0)

                # recent_rank_trend
                hist_s = hist.sort_values(["horse_id", "race_date"])
                trend_map = {}
                for hid, grp in hist_s.groupby("horse_id"):
                    last3 = grp["final_rank"].values[-3:]
                    if len(last3) >= 2:
                        slope = float(np.polyfit(np.arange(len(last3)), last3, 1)[0])
                    else:
                        slope = 0.0
                    trend_map[hid] = slope
                df["recent_rank_trend"] = df["horse_id"].map(trend_map).fillna(0.0)

                # weight_trend
                wt_map = {}
                for hid, grp in hist_s.groupby("horse_id"):
                    wts = grp["horse_weight"].values
                    diffs = np.diff(wts)
                    last3d = diffs[-3:] if len(diffs) >= 1 else np.array([0.0])
                    wt_map[hid] = float(last3d.mean())
                df["weight_trend"] = df["horse_id"].map(wt_map).fillna(0.0)
            else:
                for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                            "jockey_track_win_rate", "recent_rank_trend", "weight_trend"]:
                    df[col] = 0.0
        else:
            for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                        "jockey_track_win_rate", "recent_rank_trend", "weight_trend"]:
                df[col] = 0.0
```

- [ ] **Step 2: 파일 상단 `import numpy as np` 확인**

`predictor.py` 상단에 `import numpy as np`가 없으면 추가:

```python
import numpy as np
```

- [ ] **Step 3: 커밋**

```bash
git add core/predictor.py
git commit -m "feat: predict_from_db에 신규 피처 DB 쿼리 추가"
```

---

### Task 4: trainer.py — LambdaRank 모델 추가

**Files:**
- Modify: `core/trainer.py`

- [ ] **Step 1: import 추가**

파일 상단에 추가:

```python
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score
```

- [ ] **Step 2: `train` 메서드 끝에 `train_ranker` 호출 추가**

`self.models = results` 다음에:

```python
        self.models = results
        # LambdaRank 학습
        ranker_metrics = self._train_ranker(df_sorted, feature_cols)
        metrics.update(ranker_metrics)
        return {"models": self.models, "metrics": metrics}
```

- [ ] **Step 3: `_train_ranker` 메서드 추가**

`predict_proba` 메서드 위에 추가:

```python
    def _train_ranker(self, df_sorted: pd.DataFrame, feature_cols: list) -> dict:
        if "race_id" not in df_sorted.columns or "final_rank" not in df_sorted.columns:
            logger.warning("LambdaRank 학습 불가: race_id 또는 final_rank 없음")
            return {}

        # 경주 단위 80/20 분할 (완전한 경주 단위 유지)
        race_dates = df_sorted.groupby("race_id")["race_date"].first().sort_values()
        split_idx = int(len(race_dates) * (1 - MODEL_CONFIG["test_size"]))
        train_races = set(race_dates.index[:split_idx])
        test_races = set(race_dates.index[split_idx:])

        df_train = df_sorted[df_sorted["race_id"].isin(train_races)].sort_values("race_id")
        df_test = df_sorted[df_sorted["race_id"].isin(test_races)].sort_values("race_id")

        if df_train.empty or df_test.empty:
            logger.warning("LambdaRank: train/test split 후 데이터 부족")
            return {}

        def make_label(rank):
            if rank == 1: return 4
            if rank == 2: return 3
            if rank == 3: return 2
            return 1

        X_train = df_train[feature_cols].fillna(0)
        X_test = df_test[feature_cols].fillna(0)
        y_train = df_train["final_rank"].apply(make_label)
        y_test = df_test["final_rank"].apply(make_label)

        train_groups = df_train.groupby("race_id", sort=False).size().values
        test_groups = df_test.groupby("race_id", sort=False).size().values

        ranker = LGBMRanker(
            objective="lambdarank",
            n_estimators=300,
            max_depth=6,
            random_state=MODEL_CONFIG["random_state"],
            verbose=-1,
        )
        logger.info("LambdaRank 학습 중...")
        ranker.fit(X_train, y_train, group=train_groups)

        # NDCG@3 평가
        ndcg_scores = []
        for race_id, grp in df_test.groupby("race_id"):
            if len(grp) < 2:
                continue
            true_labels = grp["final_rank"].apply(make_label).values.reshape(1, -1)
            pred_scores = ranker.predict(grp[feature_cols].fillna(0)).reshape(1, -1)
            k = min(3, len(grp))
            ndcg_scores.append(ndcg_score(true_labels, pred_scores, k=k))

        ndcg3 = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
        logger.info(f"  LambdaRank: NDCG@3={ndcg3:.3f}")

        self.models["LambdaRank"] = ranker
        return {"LambdaRank": {"ndcg3": ndcg3}}

    def predict_rank_score(self, X: pd.DataFrame) -> np.ndarray | None:
        ranker = self.models.get("LambdaRank")
        if ranker is None:
            return None
        return ranker.predict(X.fillna(0))
```

- [ ] **Step 4: `predict_proba`에서 LambdaRank 제외**

`predict_proba`는 classifier 전용이므로 LambdaRank를 건너뜀:

```python
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("모델이 학습되지 않았습니다.")
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        probas = []
        for name, model in self.models.items():
            if name == "LambdaRank":
                continue  # ranker는 predict_rank_score로 따로 사용
            if name == "LogisticRegression":
                probas.append(model.predict_proba(X_scaled)[:, 1])
            else:
                probas.append(model.predict_proba(X)[:, 1])
        return np.mean(probas, axis=0)
```

- [ ] **Step 5: 커밋**

```bash
git add core/trainer.py
git commit -m "feat: LambdaRank 랭킹 모델 추가 (NDCG@3 평가)"
```

---

### Task 5: predictor.py — LambdaRank score를 win_probability로 사용

**Files:**
- Modify: `core/predictor.py`

- [ ] **Step 1: `predict` 메서드 수정**

기존 `predict` 메서드를 아래로 교체:

```python
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.trainer.feature_columns
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)

        # LambdaRank score 우선 사용, 없으면 classifier 앙상블
        rank_scores = self.trainer.predict_rank_score(X)
        if rank_scores is not None:
            probas = rank_scores
        else:
            probas = self.trainer.predict_proba(X)

        result = df.copy()
        result["win_probability"] = probas
        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        logger.info(f"예측 완료: {len(result)}마리, 최고 점수={probas.max():.3f}")
        return result
```

- [ ] **Step 2: 커밋**

```bash
git add core/predictor.py
git commit -m "feat: LambdaRank score를 win_probability로 사용"
```

---

### Task 6: 재학습 및 검증

**Files:**
- 없음 (실행 단계)

- [ ] **Step 1: 재학습 실행**

```bash
python main.py --mode train --start-date 2025-06-01 --end-date 2026-05-30
```

Expected 출력:
```
학습 특성 29개
LambdaRank 학습 중...
  LambdaRank: NDCG@3=0.XXX
모델 저장 완료: 2026XXXX_XXXXXX
```

- [ ] **Step 2: 오늘 예측 실행 및 점수 분포 확인**

```bash
python main.py --mode predict --date 2026-05-31
```

Expected: 경주별로 1위 score가 다른 말들보다 명확히 높게 나옴

- [ ] **Step 3: 점수 분포 상세 확인**

```python
# python -c 실행
from core.predictor import RacePredictor
from core.model_manager import ModelManager
from core.trainer import RaceModelTrainer
from core.preprocessor import RacePreprocessor

mm = ModelManager()
trainer = RaceModelTrainer()
preprocessor = RacePreprocessor()
mm.load(trainer, preprocessor)
predictor = RacePredictor(trainer, preprocessor)

preds = predictor.predict_from_db('2026-05-31')
for race_id, grp in preds.groupby('race_id'):
    grp = grp.sort_values('win_probability', ascending=False)
    scores = grp['win_probability'].values
    print(f'{race_id}: top={scores[0]:.2f}, 2nd={scores[1]:.2f}, gap={scores[0]-scores[1]:.2f}')
```

Expected: 경주마다 1위와 2위 사이 gap이 명확히 존재

- [ ] **Step 4: 최종 커밋**

```bash
git add -A
git commit -m "feat: 예측 개선 완료 - 신규 피처 5개 + LambdaRank 랭킹 모델"
```
