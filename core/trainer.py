import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
        return {"models": results, "metrics": metrics}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("모델이 학습되지 않았습니다.")
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        probas = []
        for name, model in self.models.items():
            if name == "LogisticRegression":
                probas.append(model.predict_proba(X_scaled)[:, 1])
            else:
                probas.append(model.predict_proba(X)[:, 1])
        return np.mean(probas, axis=0)
