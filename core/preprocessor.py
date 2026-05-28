import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["race_grade", "track_condition", "weather"]


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
        gender = df["gender"] if "gender" in df.columns else pd.Series([""] * len(df), index=df.index)
        df["is_male"] = (gender == "수컷").astype(int)

        df["jockey_win_rate"] = df.get("jockey_total_wins", pd.Series(0, index=df.index)).fillna(0) / \
                                (df.get("jockey_total_races", pd.Series(0, index=df.index)).fillna(0) + 1)
        df["trainer_win_rate"] = df.get("trainer_total_wins", pd.Series(0, index=df.index)).fillna(0) / \
                                 (df.get("trainer_total_races", pd.Series(0, index=df.index)).fillna(0) + 1)

        prev_wins  = df["prev_wins"].fillna(0)           if "prev_wins"           in df.columns else pd.Series(0,   index=df.index)
        prev_top3  = df["prev_top3"].fillna(0)           if "prev_top3"           in df.columns else pd.Series(0,   index=df.index)
        prev_total = df["prev_total_races"].fillna(0)    if "prev_total_races"    in df.columns else pd.Series(0,   index=df.index)
        prev_5_avg = df["prev_5_avg_rank"].fillna(6)     if "prev_5_avg_rank"     in df.columns else pd.Series(6,   index=df.index)
        prev_avg   = df["prev_total_avg_rank"].fillna(6) if "prev_total_avg_rank" in df.columns else pd.Series(6.0, index=df.index)

        df["prev_total_races"]    = prev_total
        df["prev_5_avg_rank"]     = prev_5_avg
        df["prev_total_avg_rank"] = prev_avg
        df["horse_win_rate"]      = prev_wins  / (prev_total + 1)
        df["horse_top3_rate"]     = prev_top3  / (prev_total + 1)
        df["experience_score"]    = np.log1p(prev_total)
        df["recent_form"]         = 6 - prev_5_avg

        # horse_class 서수 인코딩: "국5"→5, "외3"→3, 숫자 없으면 3 (중간 기본값)
        horse_class = df["horse_class"].fillna("") if "horse_class" in df.columns else pd.Series([""] * len(df), index=df.index)
        df["horse_class_num"] = horse_class.apply(_extract_class_num)

        # 시장 배당률 기반 내재확률
        win_odds = df["win_odds"].fillna(10.0) if "win_odds" in df.columns else pd.Series(10.0, index=df.index)
        df["win_odds"]     = win_odds
        df["implied_prob"] = 1.0 / win_odds.clip(lower=1.01)

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


def _extract_class_num(s: str) -> int:
    m = re.search(r'(\d+)$', str(s).strip())
    return int(m.group(1)) if m else 3
