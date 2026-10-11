import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, accuracy_score, ndcg_score
from xgboost import XGBRanker, XGBClassifier
from lightgbm import LGBMRanker, LGBMClassifier
from config import FEATURE_COLUMNS, MODEL_CONFIG, MODEL_DIR
from Utils.logger import get_logger

logger = get_logger(__name__)

# 확률 보정(isotonic regression) 학습용 홀드아웃은 이 날짜 이전 데이터만 쓴다.
# scripts/backtest_win.py·backtest_top3.py의 기본 백테스트 구간(2026-05-01~)과
# 겹치면 백테스트 평가 지표가 오염되므로, 반드시 그 시작일보다 이전으로 유지할 것.
CALIBRATION_EVAL_CUTOFF = "2026-05-01"


class RaceModelTrainer:
    def __init__(self):
        self.models: dict = {}              # 단승(1등) 분류 + ranker 모델
        self.top3_models: dict = {}         # 연승(3등 이내) 분류 모델 (RF, LR)
        self.top3_ranker_models: dict = {}  # 연승 ranker 모델 (XGBRanker, LGBMRanker)
        self.scaler = StandardScaler()
        self.feature_columns: list = []
        self.win_calibrator: IsotonicRegression | None = None
        self.top3_calibrator: IsotonicRegression | None = None

    # ──────────────────────────────────
    # Public API
    # ──────────────────────────────────

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

        # RandomForest·LogisticRegression: 이진분류 (행 단위)
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

        self._fit_calibrators(df_sorted, split_idx, X_test, y_test_win, y_test_top3)

        # XGBoost·LightGBM: pairwise ranker (경주 내 상대 순위 직접 학습)
        ranker_metrics      = self._train_pairwise_rankers(df_sorted, feature_cols, best_params)
        ranker_top3_metrics = self._train_pairwise_rankers_top3(df_sorted, feature_cols, best_params)

        all_metrics = {
            **win_metrics,
            **{f"top3_{k}": v for k, v in top3_metrics.items()},
            **ranker_metrics,
            **ranker_top3_metrics,
        }
        return {"models": self.models, "metrics": all_metrics}

    def tune(self, df: pd.DataFrame, n_trials: int = 60) -> dict:
        """Optuna로 XGBRanker·LGBMRanker 하이퍼파라미터 탐색 후 best_params.json 저장.

        train()과 동일한 랭커 모델·평가 지표(NDCG@3)를 사용해 튜닝 결과가 실제 학습에 유효하도록 한다.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("optuna 미설치. pip install optuna 실행 후 다시 시도하세요.")
            return {}

        if "race_id" not in df.columns or "final_rank" not in df.columns:
            logger.error("tune() 실패: race_id 또는 final_rank 컬럼 없음")
            return {}

        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        df_sorted = df.sort_values("race_date")

        # race-level 시계열 분할 (train()과 동일한 방식)
        race_dates = df_sorted.groupby("race_id")["race_date"].first().sort_values()
        split_idx_r = int(len(race_dates) * (1 - MODEL_CONFIG["test_size"]))
        train_races = set(race_dates.index[:split_idx_r])
        test_races  = set(race_dates.index[split_idx_r:])
        df_tr = df_sorted[df_sorted["race_id"].isin(train_races)].sort_values("race_id")
        df_te = df_sorted[df_sorted["race_id"].isin(test_races)].sort_values("race_id")

        if df_tr.empty or df_te.empty:
            logger.error("tune() 실패: train/test 분할 후 데이터 부족")
            return {}

        def make_label(rank: int) -> int:
            if rank == 1: return 4
            if rank == 2: return 3
            if rank == 3: return 2
            return 1

        X_tr = df_tr[feature_cols].fillna(0)
        X_te = df_te[feature_cols].fillna(0)
        y_tr = df_tr["final_rank"].apply(make_label)
        tr_groups = df_tr.groupby("race_id", sort=False).size().values
        qid_tr = np.repeat(np.arange(len(tr_groups)), tr_groups)

        def _ndcg_for_tune(scores: np.ndarray) -> float:
            df_eval = df_te.copy()
            df_eval["_score"] = scores
            vals = []
            for _, grp in df_eval.groupby("race_id"):
                if len(grp) < 2:
                    continue
                true_labels = grp["final_rank"].apply(make_label).values.reshape(1, -1)
                pred_scores = grp["_score"].values.reshape(1, -1)
                vals.append(ndcg_score(true_labels, pred_scores, k=min(3, len(grp))))
            return float(np.mean(vals)) if vals else 0.0

        logger.info(f"XGBoost 튜닝 시작 ({n_trials} trials, 평가=NDCG@3)...")

        def xgb_objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "max_depth":         trial.suggest_int("max_depth", 3, 9),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
                "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            }
            m = XGBRanker(
                objective="rank:pairwise",
                **params,
                random_state=MODEL_CONFIG["random_state"],
                verbosity=0,
            )
            m.fit(X_tr, y_tr, qid=qid_tr)
            return _ndcg_for_tune(m.predict(X_te))

        study_xgb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"XGBoost 최적 NDCG@3={study_xgb.best_value:.4f}")

        logger.info(f"LightGBM 튜닝 시작 ({n_trials} trials, 평가=NDCG@3)...")

        def lgb_objective(trial):
            params = {
                "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
                "max_depth":          trial.suggest_int("max_depth", 3, 9),
                "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "num_leaves":         trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples":  trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 1.0),
            }
            m = LGBMRanker(
                objective="lambdarank",
                **params,
                random_state=MODEL_CONFIG["random_state"], verbose=-1,
            )
            m.fit(X_tr, y_tr, group=tr_groups)
            return _ndcg_for_tune(m.predict(X_te))

        study_lgb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_lgb.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"LightGBM 최적 NDCG@3={study_lgb.best_value:.4f}")

        best_params = {
            "XGBoost":  study_xgb.best_params,
            "LightGBM": study_lgb.best_params,
        }
        params_path = MODEL_DIR / "best_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        logger.info(f"최적 파라미터 저장: {params_path}")
        return best_params

    def predict_rank_score(self, X: pd.DataFrame) -> np.ndarray | None:
        """XGBRanker·LGBMRanker 점수를 min-max 정규화 후 평균하여 반환."""
        X = X.fillna(0)
        _RANKER_KEYS = ["XGBoost", "LightGBM", "LambdaRank"]
        scores_list = []
        for name in _RANKER_KEYS:
            ranker = self.models.get(name)
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

    def predict_proba(self, X: pd.DataFrame, raw: bool = False) -> np.ndarray:
        """단승(1등) 확률 앙상블. win_calibrator가 있으면 보정된 확률을 반환한다
        (raw=True로 강제 원본 확률 — 보정 자체를 학습할 때 등 내부용)."""
        if not self.models:
            raise ValueError("모델이 학습되지 않았습니다.")
        proba = self._ensemble_proba(self.models, X)
        if not raw and self.win_calibrator is not None:
            proba = self.win_calibrator.predict(proba)
        return proba

    def predict_proba_top3(self, X: pd.DataFrame, raw: bool = False) -> np.ndarray | None:
        """연승(3등 이내) 확률 앙상블. top3_models 없으면 None.
        top3_calibrator가 있으면 보정된 확률을 반환한다(raw=True로 원본 강제)."""
        if not self.top3_models:
            return None
        proba = self._ensemble_proba(self.top3_models, X)
        if not raw and self.top3_calibrator is not None:
            proba = self.top3_calibrator.predict(proba)
        return proba

    # ──────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────

    def _top3_target(self, df_slice: pd.DataFrame, fallback) -> pd.Series:
        if "is_top3" in df_slice.columns:
            return df_slice["is_top3"]
        if "final_rank" in df_slice.columns:
            return (df_slice["final_rank"] <= 3).astype(int)
        return fallback

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

        if target == "top3":
            # XGBoost/LightGBM은 ranker용("XGBoost"/"LightGBM" 키)으로도 쓰이므로
            # _ensemble_proba의 _RANKER_KEYS 필터와 충돌하지 않도록 별도 키 사용.
            xgb_p = best_params.get("XGBoost", {})
            lgb_p = best_params.get("LightGBM", {})
            pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
            model_defs["XGBoostClf"] = (
                XGBClassifier(
                    n_estimators=     xgb_p.get("n_estimators", 200),
                    max_depth=        xgb_p.get("max_depth", 6),
                    learning_rate=    xgb_p.get("learning_rate", 0.1),
                    subsample=        xgb_p.get("subsample", 1.0),
                    colsample_bytree= xgb_p.get("colsample_bytree", 1.0),
                    min_child_weight= xgb_p.get("min_child_weight", 1),
                    gamma=            xgb_p.get("gamma", 0.0),
                    scale_pos_weight=pos_weight,
                    random_state=MODEL_CONFIG["random_state"],
                    eval_metric="logloss",
                    verbosity=0,
                ),
                X_train, X_test,
            )
            model_defs["LightGBMClf"] = (
                LGBMClassifier(
                    n_estimators=      lgb_p.get("n_estimators", 300),
                    max_depth=         lgb_p.get("max_depth", 6),
                    learning_rate=     lgb_p.get("learning_rate", 0.1),
                    subsample=         lgb_p.get("subsample", 1.0),
                    colsample_bytree=  lgb_p.get("colsample_bytree", 1.0),
                    num_leaves=        lgb_p.get("num_leaves", 31),
                    min_child_samples= lgb_p.get("min_child_samples", 20),
                    reg_alpha=         lgb_p.get("reg_alpha", 0.0),
                    reg_lambda=        lgb_p.get("reg_lambda", 0.0),
                    class_weight="balanced",
                    random_state=MODEL_CONFIG["random_state"], verbose=-1,
                ),
                X_train, X_test,
            )

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

    def _fit_calibrators(
        self, df_sorted: pd.DataFrame, split_idx: int,
        X_test: pd.DataFrame, y_test_win: pd.Series, y_test_top3: pd.Series,
    ) -> None:
        """단승/연승 앙상블 확률을 isotonic regression으로 보정한다.

        core/investor.py가 model_prob을 배당과 곱해 EV·베팅액을 직접 계산하므로,
        임계값 정확도뿐 아니라 확률값 자체의 신뢰도가 중요하다. CALIBRATION_EVAL_CUTOFF
        이전 데이터만 써서 scripts/backtest_win.py·backtest_top3.py의 평가 구간을
        오염시키지 않는다. train()의 내부 홀드아웃(X_test)은 모델이 학습 시 보지 않은
        구간이라 보정 학습에 그대로 써도 안전하다.
        """
        test_dates = df_sorted.iloc[split_idx:]["race_date"]
        calib_mask = test_dates < CALIBRATION_EVAL_CUTOFF
        n_calib = int(calib_mask.sum())
        if n_calib < 30:
            logger.warning(f"확률 보정 홀드아웃 표본 부족({n_calib}건) — 보정 생략")
            return

        X_calib = X_test[calib_mask]
        raw_win = self.predict_proba(X_calib, raw=True)
        self.win_calibrator = IsotonicRegression(out_of_bounds="clip")
        self.win_calibrator.fit(raw_win, y_test_win[calib_mask])

        raw_top3 = self.predict_proba_top3(X_calib, raw=True)
        if raw_top3 is not None:
            self.top3_calibrator = IsotonicRegression(out_of_bounds="clip")
            self.top3_calibrator.fit(raw_top3, y_test_top3[calib_mask])

        logger.info(f"확률 보정(isotonic) 완료 — 홀드아웃 {n_calib}건 ({CALIBRATION_EVAL_CUTOFF} 이전)")

    def _train_pairwise_rankers(
        self, df_sorted: pd.DataFrame, feature_cols: list, best_params: dict
    ) -> dict:
        """XGBRanker(pairwise)·LGBMRanker(lambdarank) 학습. race-level 그룹 사용."""
        if "race_id" not in df_sorted.columns or "final_rank" not in df_sorted.columns:
            logger.warning("Ranker 학습 불가: race_id 또는 final_rank 없음")
            return {}

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
        train_groups = df_train.groupby("race_id", sort=False).size().values
        xgb_p = best_params.get("XGBoost", {})
        lgb_p = best_params.get("LightGBM", {})
        metrics = {}

        # ── XGBRanker (pairwise) ──────────────────────────────────────────
        logger.info("XGBRanker(pairwise) 학습 중...")
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
        xgb_p1   = self._precision_at_1_score(df_test.assign(score=xgb_scores), "score")
        xgb_ndcg = self._eval_ndcg(df_test, xgb_scores, feature_cols)
        self.models["XGBoost"] = xgb_ranker
        metrics["XGBoost"] = {"precision_at_1": xgb_p1, "ndcg3": xgb_ndcg}
        logger.info(f"  XGBRanker: Precision@1={xgb_p1:.3f}, NDCG@3={xgb_ndcg:.3f}")

        # ── LGBMRanker (lambdarank) ──────────────────────────────────────
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
        lgb_p1   = self._precision_at_1_score(df_test.assign(score=lgb_scores), "score")
        lgb_ndcg = self._eval_ndcg(df_test, lgb_scores, feature_cols)
        self.models["LightGBM"] = lgb_ranker
        metrics["LightGBM"] = {"precision_at_1": lgb_p1, "ndcg3": lgb_ndcg}
        logger.info(f"  LGBMRanker: Precision@1={lgb_p1:.3f}, NDCG@3={lgb_ndcg:.3f}")

        return metrics

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
            ndcg_scores.append(ndcg_score(true_labels, pred_scores, k=min(3, len(grp))))
        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def _ensemble_proba(self, model_dict: dict, X: pd.DataFrame) -> np.ndarray:
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
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

    def _load_best_params(self) -> dict:
        params_path = MODEL_DIR / "best_params.json"
        if params_path.exists():
            with open(params_path, encoding="utf-8") as f:
                params = json.load(f)
            logger.info(f"저장된 최적 파라미터 로드: {list(params.keys())}")
            return params
        return {}
