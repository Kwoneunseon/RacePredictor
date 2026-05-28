import pandas as pd
import numpy as np
from Utils.logger import get_logger

logger = get_logger(__name__)


class RacePredictor:
    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.trainer.feature_columns
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)

        probas = self.trainer.predict_proba(X)

        result = df.copy()
        result["win_probability"] = probas
        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        logger.info(f"예측 완료: {len(result)}마리, 최고 확률={probas.max():.3f}")
        return result

    def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        from Utils.database import get_client
        client = get_client()

        query = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).eq("race_date", race_date)

        if meet_code:
            query = query.eq("races.meet_code", meet_code)

        result = query.execute()
        if not result.data:
            logger.warning(f"{race_date} 출전 데이터 없음")
            return pd.DataFrame()

        rows = []
        for r in result.data:
            flat = {k: v for k, v in r.items() if not isinstance(v, dict)}

            h = r.get("horses") or {}
            flat["horse_name"] = h.get("name")
            flat["horse_age"] = h.get("age")
            flat["gender"] = h.get("gender")
            flat["horse_class"] = h.get("rank")

            rc = r.get("races") or {}
            flat["race_no"] = rc.get("race_no")
            flat["meet_code"] = rc.get("meet_code")
            flat["race_distance"] = rc.get("race_distance")
            flat["total_horses"] = rc.get("total_horses")
            flat["race_grade"] = rc.get("race_grade")
            flat["track_condition"] = rc.get("track_condition")
            flat["weather"] = rc.get("weather")

            j = r.get("jockeys") or {}
            flat["jockey_total_wins"] = j.get("total_wins", 0) or 0
            flat["jockey_total_races"] = j.get("total_races", 0) or 0

            t = r.get("trainers") or {}
            flat["trainer_total_wins"] = t.get("ord1_cnt_t", 0) or 0
            flat["trainer_total_races"] = t.get("rc_cnt_t", 0) or 0

            rows.append(flat)

        df = pd.DataFrame(rows)

        # horse_race_history 조인: 해당 날짜 이하의 가장 최신 통계
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if horse_ids:
            hist_res = client.table("horse_race_history").select(
                "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, prev_5_avg_rank, prev_total_avg_rank"
            ).in_("horse_id", horse_ids).lte("as_of_date", race_date).execute()

            if hist_res.data:
                hist_df = pd.DataFrame(hist_res.data)
                hist_latest = (
                    hist_df.sort_values("as_of_date", ascending=False)
                           .drop_duplicates(subset=["horse_id"], keep="first")
                           .drop(columns=["as_of_date"])
                )
                df = df.merge(hist_latest, on="horse_id", how="left")

        df = self.preprocessor.transform(df)
        return self.predict(df)
