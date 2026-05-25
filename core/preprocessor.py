import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["horse_class", "race_grade", "track_condition", "weather"]


class RacePreprocessor:
    def __init__(self):
        self.label_encoders: dict = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_target(df)
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df, fit=False)
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if "final_rank" in df.columns:
            df["is_winner"] = (df["final_rank"] == 1).astype(int)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        gender = df.get("gender", pd.Series([""] * len(df), index=df.index))
        df["is_male"] = (gender == "수컷").astype(int)
        df["jockey_win_rate"] = df["jockey_total_wins"] / (df["jockey_total_races"] + 1)
        df["trainer_win_rate"] = df["trainer_total_wins"] / (df["trainer_total_races"] + 1)
        df["horse_win_rate"] = df["prev_wins"] / (df["prev_total_races"] + 1)
        df["horse_top3_rate"] = df["prev_top3"] / (df["prev_total_races"] + 1)
        df["experience_score"] = np.log1p(df["prev_total_races"])
        df["recent_form"] = 6 - df["prev_5_avg_rank"].fillna(6)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[f"{col}_enc"] = 0
                continue
            series = df[col].fillna("unknown").astype(str)
            if fit:
                enc = LabelEncoder()
                df[f"{col}_enc"] = enc.fit_transform(series)
                self.label_encoders[col] = enc
            else:
                if col in self.label_encoders:
                    enc = self.label_encoders[col]
                    known = set(enc.classes_)
                    series = series.apply(lambda x: x if x in known else enc.classes_[0])
                    df[f"{col}_enc"] = enc.transform(series)
                else:
                    df[f"{col}_enc"] = 0
        return df
