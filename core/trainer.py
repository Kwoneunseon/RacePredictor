import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, ndcg_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRanker
from config import FEATURE_COLUMNS, MODEL_CONFIG, MODEL_DIR
from Utils.logger import get_logger

logger = get_logger(__name__)


class RaceModelTrainer:
    def __init__(self):
        self.models: dict = {}         # 단승(1등) 분류 모델
        self.top3_models: dict = {}    # 연승(3등 이내) 분류 모델
        self.scaler = StandardScaler()
        self.feature_columns: list = []

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

        logger.info("=== 단승(1등) 모델 학습 ===")
        win_metrics = self._train_classifiers(
            X_train, X_test, y_train_win, y_test_win,
            X_train_s, X_test_s, target="winner", best_params=best_params,
        )

        logger.info("=== 연승(3등 이내) 모델 학습 ===")
        top3_metrics = self._train_classifiers(
            X_train, X_test, y_train_top3, y_test_top3,
            X_train_s, X_test_s, target="top3", best_params=best_params,
        )

        ranker_metrics = self._train_ranker(df_sorted, feature_cols)

        all_metrics = {
            **win_metrics,
            **{f"top3_{k}": v for k, v in top3_metrics.items()},
            **ranker_metrics,
        }
        return {"models": self.models, "metrics": all_metrics}

    def tune(self, df: pd.DataFrame, n_trials: int = 60) -> dict:
        """Optuna로 XGBoost·LightGBM 하이퍼파라미터 탐색 후 best_params.json 저장."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("optuna 미설치. pip install optuna 실행 후 다시 시도하세요.")
            return {}

        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        df_sorted = df.sort_values("race_date")
        split_idx = int(len(df_sorted) * (1 - MODEL_CONFIG["test_size"]))

        X_train = df_sorted.iloc[:split_idx][feature_cols].fillna(0)
        X_test  = df_sorted.iloc[split_idx:][feature_cols].fillna(0)
        y_train = df_sorted.iloc[:split_idx]["is_winner"]
        y_test  = df_sorted.iloc[split_idx:]["is_winner"]
        pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

        logger.info(f"XGBoost 튜닝 시작 ({n_trials} trials)...")

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
            m = XGBClassifier(
                **params, scale_pos_weight=pos_weight,
                random_state=MODEL_CONFIG["random_state"],
                eval_metric="logloss", verbosity=0,
            )
            m.fit(X_train, y_train)
            return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])

        study_xgb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"XGBoost 최적 AUC={study_xgb.best_value:.4f}")

        logger.info(f"LightGBM 튜닝 시작 ({n_trials} trials)...")

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
            m = LGBMClassifier(
                **params, class_weight="balanced",
                random_state=MODEL_CONFIG["random_state"], verbose=-1,
            )
            m.fit(X_train, y_train)
            return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])

        study_lgb = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_lgb.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"LightGBM 최적 AUC={study_lgb.best_value:.4f}")

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
        ranker = self.models.get("LambdaRank")
        if ranker is None:
            return None
        return ranker.predict(X.fillna(0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """단승(1등) 확률 앙상블"""
        if not self.models:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self._ensemble_proba(self.models, X)

    def predict_proba_top3(self, X: pd.DataFrame) -> np.ndarray | None:
        """연승(3등 이내) 확률 앙상블. top3_models 없으면 None."""
        if not self.top3_models:
            return None
        return self._ensemble_proba(self.top3_models, X)

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
        pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        xgb_p = best_params.get("XGBoost", {})
        lgb_p = best_params.get("LightGBM", {})

        model_defs = {
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced",
                ),
                X_train, X_test,
            ),
            "XGBoost": (
                XGBClassifier(
                    n_estimators=       xgb_p.get("n_estimators", 200),
                    max_depth=          xgb_p.get("max_depth", 6),
                    learning_rate=      xgb_p.get("learning_rate", 0.1),
                    subsample=          xgb_p.get("subsample", 1.0),
                    colsample_bytree=   xgb_p.get("colsample_bytree", 1.0),
                    min_child_weight=   xgb_p.get("min_child_weight", 1),
                    gamma=              xgb_p.get("gamma", 0.0),
                    random_state=MODEL_CONFIG["random_state"],
                    eval_metric="logloss", verbosity=0,
                    scale_pos_weight=pos_weight,
                ),
                X_train, X_test,
            ),
            "LightGBM": (
                LGBMClassifier(
                    n_estimators=       lgb_p.get("n_estimators", 200),
                    max_depth=          lgb_p.get("max_depth", 6),
                    learning_rate=      lgb_p.get("learning_rate", 0.1),
                    subsample=          lgb_p.get("subsample", 1.0),
                    colsample_bytree=   lgb_p.get("colsample_bytree", 1.0),
                    num_leaves=         lgb_p.get("num_leaves", 31),
                    min_child_samples=  lgb_p.get("min_child_samples", 20),
                    reg_alpha=          lgb_p.get("reg_alpha", 0.0),
                    reg_lambda=         lgb_p.get("reg_lambda", 0.0),
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced", verbose=-1,
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
            self.models = results
        else:
            self.top3_models = results
        return metrics

    def _train_ranker(self, df_sorted: pd.DataFrame, feature_cols: list) -> dict:
        if "race_id" not in df_sorted.columns or "final_rank" not in df_sorted.columns:
            logger.warning("LambdaRank 학습 불가: race_id 또는 final_rank 없음")
            return {}

        race_dates = df_sorted.groupby("race_id")["race_date"].first().sort_values()
        split_idx = int(len(race_dates) * (1 - MODEL_CONFIG["test_size"]))
        train_races = set(race_dates.index[:split_idx])
        test_races  = set(race_dates.index[split_idx:])

        df_train = df_sorted[df_sorted["race_id"].isin(train_races)].sort_values("race_id")
        df_test  = df_sorted[df_sorted["race_id"].isin(test_races)].sort_values("race_id")

        if df_train.empty or df_test.empty:
            logger.warning("LambdaRank: train/test split 후 데이터 부족")
            return {}

        def make_label(rank):
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

        ranker = LGBMRanker(
            objective="lambdarank",
            n_estimators=300, max_depth=6,
            random_state=MODEL_CONFIG["random_state"], verbose=-1,
        )
        logger.info("LambdaRank 학습 중...")
        ranker.fit(X_train, y_train, group=train_groups)

        ndcg_scores = []
        for _, grp in df_test.groupby("race_id"):
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

    def _ensemble_proba(self, model_dict: dict, X: pd.DataFrame) -> np.ndarray:
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        probas = []
        for name, model in model_dict.items():
            if name == "LambdaRank":
                continue
            Xin = X_scaled if name == "LogisticRegression" else X
            probas.append(model.predict_proba(Xin)[:, 1])
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

    def _load_best_params(self) -> dict:
        params_path = MODEL_DIR / "best_params.json"
        if params_path.exists():
            with open(params_path, encoding="utf-8") as f:
                params = json.load(f)
            logger.info(f"저장된 최적 파라미터 로드: {list(params.keys())}")
            return params
        return {}
