import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Utils.logger import get_logger
from Utils.ranks import valid_rank_mask

logger = get_logger(__name__)

CATEGORICAL_COLS = ["race_grade", "track_condition", "weather"]


class RacePreprocessor:
    def __init__(self):
        self.label_encoders: dict = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_non_finish_rows(df)
        df = self._normalize_raw_fields(df)
        df = self._add_target(df)
        df = self._add_derived_features(df)
        df = self._add_rolling_recent_rates(df)
        df = self._add_historical_performance_features(df)
        df = self._add_new_features(df)
        df = self._add_race_relative_features(df)
        df = self._encode_categoricals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._normalize_raw_fields(df)
        df = self._add_derived_features(df)
        if "jockey_recent_win_rate" not in df.columns:
            df["jockey_recent_win_rate"] = 0.0
        if "trainer_recent_win_rate" not in df.columns:
            df["trainer_recent_win_rate"] = 0.0
        if "days_since_last_race" not in df.columns:
            df["days_since_last_race"] = 60
        if "horse_weight_diff" not in df.columns:
            df["horse_weight_diff"] = 0.0
        if "horse_win_rate_at_distance" not in df.columns:
            df["horse_win_rate_at_distance"] = 0.0
        if "horse_speed_at_distance" not in df.columns:
            df["horse_speed_at_distance"] = 0.0
        if "assigned_weight" not in df.columns:
            df["assigned_weight"] = 56.0
        for col in ["s1f_avg", "c1_avg", "c2_avg", "c3_avg", "c4_avg", "g3f_avg", "g1f_avg",
                    "horse_career_win_rate",
                    "jockey_career_win_rate", "jockey_recent1y_win_rate",
                    "trainer_career_win_rate", "trainer_recent1y_win_rate",
                    "jockey_recent1y_starts", "trainer_recent1y_starts"]:
            if col not in df.columns:
                df[col] = 0.0
        if "running_style_score" not in df.columns:
            df["running_style_score"] = 0.0
        for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                    "recent_rank_trend", "jockey_track_win_rate", "weight_trend"]:
            if col not in df.columns:
                df[col] = 0.0
        if "prev_win_odds" not in df.columns:
            df["prev_win_odds"] = 15.0
        if "entry_number" not in df.columns:
            df["entry_number"] = 0
        if "prev_speed_figure" not in df.columns:
            df["prev_speed_figure"] = 0.0
        if "speed_figure_trend" not in df.columns:
            df["speed_figure_trend"] = 0.0
        if "recent_form" not in df.columns:
            df["recent_form"] = 0.0
        if "experience_score" not in df.columns:
            df["experience_score"] = 0.0
        if "is_male" not in df.columns:
            df["is_male"] = 0
        if "implied_prob_vs_mean" not in df.columns:
            df["implied_prob_vs_mean"] = 1.0
        if "race_implied_prob_rank" not in df.columns:
            df["race_implied_prob_rank"] = 1
        df = self._add_race_relative_features(df)
        df = self._encode_categoricals(df, fit=False)
        return df

    def _normalize_raw_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        # track_condition: "양호 (9%)" → "양호", "" / "(0%)" / "-" → "unknown"
        if "track_condition" in df.columns:
            df["track_condition"] = df["track_condition"].apply(_normalize_track_condition)
        # weather: "-" / "" → "unknown"
        if "weather" in df.columns:
            df["weather"] = df["weather"].apply(_normalize_weather)
        return df

    def _drop_non_finish_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습용: 유효 착순(1 <= final_rank < NON_FINISH_RANK_MIN)이 아닌 행(미기입 0,
        90번대 취소/제외/주행중지 등 상태 코드)을 제거한다. 남겨두면 is_top3 음성 라벨이 되고
        recent_rank_trend·누적 출전 통계에도 섞인다. 유효 행이 없으면 빈 DataFrame이 된다."""
        if "final_rank" not in df.columns:
            return df
        mask = valid_rank_mask(df["final_rank"])
        dropped = int((~mask).sum())
        if dropped:
            logger.warning(f"비착순 상태 코드/미기입 {dropped}건 학습 데이터에서 제외 (final_rank 0 또는 90 이상)")
        return df[mask].copy()

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if "final_rank" in df.columns:
            df["is_winner"] = (df["final_rank"] == 1).astype(int)
            df["is_top3"] = (df["final_rank"] <= 3).astype(int)
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

        return df

    def _add_rolling_recent_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["race_date", "jk_no", "final_rank"]
        if not all(c in df.columns for c in needed):
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
            return df

        df = df.copy()
        df["_rdt"] = pd.to_datetime(df["race_date"])
        df["_is_win"] = (df["final_rank"] == 1).astype(float)

        def _rolling_rate(id_col):
            rates = pd.Series(0.0, index=df.index)
            for gid, grp in df.groupby(id_col):
                grp = grp.sort_values("_rdt")
                dates = grp["_rdt"].values
                wins = grp["_is_win"].values
                vals = []
                for i in range(len(wins)):
                    current_date = dates[i]
                    cutoff = current_date - np.timedelta64(30, "D")
                    mask = (dates < current_date) & (dates >= cutoff)
                    past = wins[mask]
                    vals.append(float(past.mean()) if len(past) > 0 else 0.0)
                rates[grp.index] = vals
            return rates

        df["jockey_recent_win_rate"] = _rolling_rate("jk_no")
        if "trainer_id" in df.columns:
            df["trainer_recent_win_rate"] = _rolling_rate("trainer_id")
        else:
            df["trainer_recent_win_rate"] = 0.0

        df = df.drop(columns=["_rdt", "_is_win"])
        return df

    def _add_historical_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["race_date", "horse_id", "final_rank", "horse_weight", "race_distance"]
        if not all(c in df.columns for c in needed):
            df["days_since_last_race"] = 60
            df["horse_weight_diff"] = 0.0
            df["horse_win_rate_at_distance"] = 0.0
            df["horse_speed_at_distance"] = 0.0
            return df

        df = df.copy()
        df["_rdt"] = pd.to_datetime(df["race_date"])
        df = df.sort_values(["horse_id", "_rdt"])

        df["_prev_date"] = df.groupby("horse_id")["_rdt"].shift(1)
        df["days_since_last_race"] = (df["_rdt"] - df["_prev_date"]).dt.days.fillna(60)

        df["_prev_weight"] = df.groupby("horse_id")["horse_weight"].shift(1)
        df["horse_weight_diff"] = (df["horse_weight"].fillna(0) - df["_prev_weight"].fillna(0)).fillna(0)

        # 거리별 승률: 현재 경주 이전 누적 (leakage 방지)
        df["_is_win"] = (df["final_rank"] == 1).astype(float)
        df = df.sort_values(["horse_id", "race_distance", "_rdt"])
        df["_cumwins_dist"] = (df.groupby(["horse_id", "race_distance"])["_is_win"].cumsum() - df["_is_win"])
        df["_cumraces_dist"] = df.groupby(["horse_id", "race_distance"]).cumcount()
        df["horse_win_rate_at_distance"] = df["_cumwins_dist"] / (df["_cumraces_dist"] + 1)

        # 거리별 평균 속도: 현재 경주 이전 누적 (leakage 방지). 해당 거리 첫 출전은 0.0(정보 없음).
        if "finish_time" in df.columns:
            df["_speed"] = df["race_distance"] / df["finish_time"].replace(0, np.nan)
            df["_speed_valid"] = df["_speed"].notna().astype(float)
            df["_speed_filled"] = df["_speed"].fillna(0.0)
            grp = df.groupby(["horse_id", "race_distance"])
            df["_cum_speed_sum"] = grp["_speed_filled"].cumsum() - df["_speed_filled"]
            df["_cum_speed_cnt"] = grp["_speed_valid"].cumsum() - df["_speed_valid"]
            df["horse_speed_at_distance"] = np.where(
                df["_cum_speed_cnt"] > 0,
                df["_cum_speed_sum"] / df["_cum_speed_cnt"],
                0.0,
            )
            df = df.drop(columns=["_speed", "_speed_valid", "_speed_filled", "_cum_speed_sum", "_cum_speed_cnt"])
        else:
            df["horse_speed_at_distance"] = 0.0

        df = df.drop(columns=["_rdt", "_prev_date", "_prev_weight", "_is_win", "_cumwins_dist", "_cumraces_dist"])
        return df

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

    def _add_race_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # current_win_odds 우선 사용 (예측 시 실시간 배당), 없으면 win_odds(학습 시 결과 배당) 사용
        if "current_win_odds" in df.columns:
            odds_src = pd.to_numeric(df["current_win_odds"], errors="coerce").replace(0, np.nan)
        elif "win_odds" in df.columns:
            odds_src = pd.to_numeric(df["win_odds"], errors="coerce").replace(0, np.nan)
        else:
            odds_src = pd.Series(np.nan, index=df.index)

        df["implied_prob"] = 1.0 / odds_src

        if "race_id" not in df.columns:
            df["horse_class_vs_mean"] = 0.0
            df["win_rate_vs_mean"] = 0.0
            df["race_implied_prob_rank"] = 1
            df["implied_prob_vs_mean"] = 1.0
            return df

        grp = df.groupby("race_id")
        df["horse_class_vs_mean"] = grp["horse_class_num"].transform("mean") - df["horse_class_num"]
        df["win_rate_vs_mean"] = df["horse_win_rate"] - grp["horse_win_rate"].transform("mean")
        df["race_implied_prob_rank"] = grp["implied_prob"].rank(ascending=False, method="min")
        df["implied_prob_vs_mean"] = df["implied_prob"] / grp["implied_prob"].transform("mean")
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


def _normalize_track_condition(val) -> str:
    s = str(val).strip() if val is not None else ""
    if not s or s in ("-",) or re.fullmatch(r'\(\d+%\)', s):
        return "unknown"
    m = re.match(r'^([가-힣]+)', s)
    return m.group(1) if m else "unknown"


def _normalize_weather(val) -> str:
    s = str(val).strip() if val is not None else ""
    return s if s and s != "-" else "unknown"
