import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, ndcg_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRanker
from config import FEATURE_COLUMNS, MODEL_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)


class RaceModelTrainer:
    def __init__(self):
        self.models: dict = {}
        self.scaler = StandardScaler()
        self.feature_columns: list = []

    def train(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        self.feature_columns = feature_cols
        logger.info(f"학습 특성 {len(feature_cols)}개")

        df_sorted = df.sort_values("race_date")
        split_idx = int(len(df_sorted) * (1 - MODEL_CONFIG["test_size"]))

        X_train = df_sorted.iloc[:split_idx][feature_cols].fillna(0)
        X_test = df_sorted.iloc[split_idx:][feature_cols].fillna(0)
        y_train = df_sorted.iloc[:split_idx]["is_winner"]
        y_test = df_sorted.iloc[split_idx:]["is_winner"]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

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
                    n_estimators=200, max_depth=6,
                    random_state=MODEL_CONFIG["random_state"],
                    eval_metric="logloss",
                    verbosity=0,
                    scale_pos_weight=pos_weight,
                ),
                X_train, X_test,
            ),
            "LightGBM": (
                LGBMClassifier(
                    n_estimators=200, max_depth=6,
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced",
                    verbose=-1,
                ),
                X_train, X_test,
            ),
            "LogisticRegression": (
                LogisticRegression(
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced",
                    max_iter=1000,
                ),
                X_train_s, X_test_s,
            ),
        }

        results, metrics = {}, {}
        for name, (model, Xtr, Xte) in model_defs.items():
            logger.info(f"{name} 학습 중...")
            model.fit(Xtr, y_train)
            y_prob = model.predict_proba(Xte)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.0
            acc = accuracy_score(y_test, y_pred)
            results[name] = model
            metrics[name] = {"auc": auc, "accuracy": acc}
            logger.info(f"  {name}: AUC={auc:.3f}, Acc={acc:.3f}")

        self.models = results
        ranker_metrics = self._train_ranker(df_sorted, feature_cols)
        metrics.update(ranker_metrics)
        return {"models": self.models, "metrics": metrics}

    def _train_ranker(self, df_sorted: pd.DataFrame, feature_cols: list) -> dict:
        if "race_id" not in df_sorted.columns or "final_rank" not in df_sorted.columns:
            logger.warning("LambdaRank 학습 불가: race_id 또는 final_rank 없음")
            return {}

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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("모델이 학습되지 않았습니다.")
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        probas = []
        for name, model in self.models.items():
            if name == "LambdaRank":
                continue
            if name == "LogisticRegression":
                probas.append(model.predict_proba(X_scaled)[:, 1])
            else:
                probas.append(model.predict_proba(X)[:, 1])
        return np.mean(probas, axis=0)
