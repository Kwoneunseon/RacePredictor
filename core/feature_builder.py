"""DB에서 피처를 수집·계산해 DataFrame으로 반환하는 클래스"""
import numpy as np
import pandas as pd
import library.scraper_odds as scraper_odds
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


def _training_grade(remk_txt: str) -> int:
    """조교 평가 텍스트 → 수치 등급. 양호=2, 기록없음=1, 불량계열=0"""
    s = str(remk_txt).strip()
    if not s:
        return 1
    if "불량" in s:
        return 0
    if "양호" in s:
        return 2
    return 1


class RaceFeatureBuilder:
    """
    race_entries 기반 피처 수집 클래스.

    - build_for_prediction(race_date): 예측용. DB에서 race_entries 조회 후 피처 전부 붙임.
    - build_for_training(df, end_date): 학습용. 이미 조회된 df에 피처 조인.
    """

    # ──────────────────────────────────
    # Public API
    # ──────────────────────────────────

    def build_for_prediction(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        client = get_client()

        result = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).eq("race_date", race_date).execute()

        if not result.data:
            logger.warning(f"{race_date} 출전 데이터 없음")
            return pd.DataFrame()

        df = self._flatten_entries(result.data)

        if meet_code is not None:
            df = df[df["meet_code"] == meet_code].reset_index(drop=True)
            if df.empty:
                return df

        df = self._fetch_weights(df, race_date, client)
        df = self._fetch_recent_rates(df, race_date, client)
        df = self._fetch_training_grades(df, race_date, client)
        df = self._fetch_history(df, race_date, client)
        df = self._fetch_last_race_features(df, race_date, client)
        df = self._fetch_prev_odds(df, race_date, client)
        df = self._fetch_current_odds(df, race_date)
        df = self._fetch_speed_features(df, race_date, client)
        df = self._fetch_assigned_weight(df, client)
        df = self._fetch_distance_rates(df, race_date, client)
        df = self._fetch_combo_features(df, race_date, client)
        return df

    def build_for_training(self, df: pd.DataFrame, end_date: str) -> pd.DataFrame:
        client = get_client()

        df = self._fetch_history_for_training(df, end_date, client)
        df = self._fetch_training_grades_for_training(df, client)
        df = self._fetch_prev_odds_for_training(df)
        df = self._fetch_speed_features_for_training(df)
        df = self._fetch_assigned_weight_for_training(df)
        return df

    # ──────────────────────────────────
    # 공통 헬퍼
    # ──────────────────────────────────

    def _flatten_entries(self, data: list) -> pd.DataFrame:
        rows = []
        for r in data:
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
        return pd.DataFrame(rows)

    # ──────────────────────────────────
    # 예측용 피처 메서드
    # ──────────────────────────────────

    def _fetch_weights(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        if "horse_weight" not in df.columns or "entry_number" not in df.columns:
            return df
        zero_races = df[
            df["horse_weight"].isna() | (df["horse_weight"] == 0)
        ][["meet_code", "race_no"]].drop_duplicates()
        if not zero_races.empty:
            from library.scraper_weights import fetch_race_weights
            for _, row in zero_races.iterrows():
                mc, rn = int(row["meet_code"]), int(row["race_no"])
                weights = fetch_race_weights(mc, rn, race_date)
                for entry_no, wt in weights.items():
                    mask = (
                        (df["meet_code"] == mc)
                        & (df["race_no"] == rn)
                        & (df["entry_number"] == entry_no)
                    )
                    if mask.any():
                        df.loc[mask, "horse_weight"] = wt["horse_weight"]
        return df

    def _fetch_recent_rates(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        if "jk_no" not in df.columns:
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
            return df
        cutoff = (pd.to_datetime(race_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        res = client.table("race_entries")\
            .select("jk_no, trainer_id, final_rank")\
            .gte("race_date", cutoff).lt("race_date", race_date)\
            .not_.is_("final_rank", "null").execute()
        if res.data:
            h = pd.DataFrame(res.data)
            h["_is_win"] = (h["final_rank"] == 1).astype(int)
            jk_rates = h.groupby("jk_no")["_is_win"].mean().to_dict()
            df["jockey_recent_win_rate"] = df["jk_no"].map(jk_rates).fillna(0.0)
            if "trainer_id" in df.columns:
                tr_rates = h.groupby("trainer_id")["_is_win"].mean().to_dict()
                df["trainer_recent_win_rate"] = df["trainer_id"].map(tr_rates).fillna(0.0)
            else:
                df["trainer_recent_win_rate"] = 0.0
        else:
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
        return df

    def _fetch_training_grades(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        if "horse_name" not in df.columns:
            df["training_grade_enc"] = 1.0
            return df
        horse_names = df["horse_name"].dropna().unique().tolist()
        cutoff = (pd.to_datetime(race_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        res = client.table("horse_training_sessions")\
            .select("hrnm, remk_txt")\
            .in_("hrnm", horse_names)\
            .gte("trng_dt", cutoff).lt("trng_dt", race_date)\
            .execute()
        if res.data:
            grade_map: dict = {}
            for row in res.data:
                name = row["hrnm"]
                grade = _training_grade(row.get("remk_txt", ""))
                grade_map[name] = min(grade_map.get(name, 2), grade)
            df["training_grade_enc"] = df["horse_name"].apply(
                lambda n: float(grade_map.get(n, 1))
            )
        else:
            df["training_grade_enc"] = 1.0
        return df

    def _fetch_history(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            return df
        res = client.table("horse_race_history").select(
            "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, "
            "prev_5_avg_rank, prev_total_avg_rank"
        ).in_("horse_id", horse_ids).lt("as_of_date", race_date).execute()
        if res.data:
            hist_df = pd.DataFrame(res.data)
            hist_latest = (
                hist_df.sort_values("as_of_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
                       .drop(columns=["as_of_date"])
            )
            df = df.merge(hist_latest, on="horse_id", how="left")
        return df

    def _fetch_last_race_features(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            df["days_since_last_race"] = 60
            df["horse_weight_diff"] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, race_date, horse_weight")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).execute()
        if res.data:
            last_df = pd.DataFrame(res.data)
            last_df = (
                last_df.sort_values("race_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
            )
            race_dt = pd.to_datetime(race_date)
            last_df["_days"] = (race_dt - pd.to_datetime(last_df["race_date"])).dt.days
            days_map = last_df.set_index("horse_id")["_days"].to_dict()
            prev_wt_map = last_df.set_index("horse_id")["horse_weight"].to_dict()
            df["days_since_last_race"] = df["horse_id"].map(days_map).fillna(60)
            df["_prev_weight"] = df["horse_id"].map(prev_wt_map).fillna(0)
            zero_mask = df["horse_weight"].isna() | (df["horse_weight"] == 0)
            df.loc[zero_mask, "horse_weight"] = df.loc[zero_mask, "_prev_weight"]
            df["horse_weight_diff"] = df["horse_weight"].fillna(0) - df["_prev_weight"].fillna(0)
            df = df.drop(columns=["_prev_weight"])
        else:
            df["days_since_last_race"] = 60
            df["horse_weight_diff"] = 0.0
        return df

    def _fetch_prev_odds(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            df["prev_win_odds"] = 15.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, race_date, win_odds")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("win_odds", 0).execute()
        if res.data:
            prev_df = pd.DataFrame(res.data)
            prev_df = (
                prev_df.sort_values("race_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
            )
            odds_map = prev_df.set_index("horse_id")["win_odds"].to_dict()
            median_odds = prev_df["win_odds"].median()
            df["prev_win_odds"] = df["horse_id"].map(odds_map).fillna(median_odds)
        else:
            df["prev_win_odds"] = 15.0
        return df

    def _fetch_current_odds(self, df: pd.DataFrame, race_date: str) -> pd.DataFrame:
        """경주 직전 실시간 단승배당 스크래핑. 실패하면 prev_win_odds로 대체."""
        df["current_win_odds"] = df["prev_win_odds"]
        if not {"meet_code", "race_no", "entry_number"}.issubset(df.columns):
            return df

        for (meet_code, race_no), grp in df.groupby(["meet_code", "race_no"]):
            odds_map = scraper_odds.fetch_race_win_odds(int(meet_code), int(race_no), race_date)
            if not odds_map:
                continue
            for idx in grp.index:
                try:
                    entry_no = int(df.at[idx, "entry_number"])
                except (TypeError, ValueError):
                    continue
                if entry_no in odds_map:
                    df.at[idx, "current_win_odds"] = odds_map[entry_no]
        return df

    def _fetch_speed_features(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            df["prev_speed_figure"] = 0.0
            df["speed_figure_trend"] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, race_date, finish_time, races(race_distance)")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).not_.is_("finish_time", "null").execute()
        if res.data:
            spd_rows = [{
                "horse_id": r["horse_id"],
                "race_date": r["race_date"],
                "finish_time": r["finish_time"],
                "race_distance": (r.get("races") or {}).get("race_distance"),
            } for r in res.data]
            spd_df = pd.DataFrame(spd_rows).dropna(subset=["finish_time", "race_distance"])
            spd_df = spd_df[spd_df["finish_time"] > 0]
            if not spd_df.empty:
                spd_df["_speed"] = spd_df["race_distance"].astype(float) / spd_df["finish_time"]
                spd_sorted = spd_df.sort_values(["horse_id", "race_date"])
                spd_latest = spd_sorted.drop_duplicates(subset=["horse_id"], keep="last")
                spd_map = spd_latest.set_index("horse_id")["_speed"].to_dict()
                df["prev_speed_figure"] = df["horse_id"].map(spd_map).fillna(0.0)
                trend_map = {}
                for hid, grp in spd_sorted.groupby("horse_id"):
                    vals = grp["_speed"].values[-3:]
                    trend_map[hid] = (
                        float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
                        if len(vals) >= 2 else 0.0
                    )
                df["speed_figure_trend"] = df["horse_id"].map(trend_map).fillna(0.0)
            else:
                df["prev_speed_figure"] = 0.0
                df["speed_figure_trend"] = 0.0
        else:
            df["prev_speed_figure"] = 0.0
            df["speed_figure_trend"] = 0.0
        return df

    def _fetch_distance_rates(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids or "race_distance" not in df.columns:
            df["horse_win_rate_at_distance"] = 0.0
            df["horse_speed_at_distance"] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, final_rank, finish_time, races(race_distance)")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).execute()
        if res.data:
            dist_rows = [{
                "horse_id": r["horse_id"],
                "final_rank": r["final_rank"],
                "finish_time": r.get("finish_time"),
                "race_distance": (r.get("races") or {}).get("race_distance"),
            } for r in res.data]
            dist_df = pd.DataFrame(dist_rows).dropna(subset=["race_distance"])
            dist_df["_is_win"] = (dist_df["final_rank"] == 1).astype(float)
            dist_stats = dist_df.groupby(["horse_id", "race_distance"]).agg(
                _wins=("_is_win", "sum"), _races=("_is_win", "count")
            ).reset_index()
            dist_stats["horse_win_rate_at_distance"] = (
                dist_stats["_wins"] / (dist_stats["_races"] + 1)
            )

            dist_df["_speed"] = pd.to_numeric(dist_df["race_distance"], errors="coerce") / \
                pd.to_numeric(dist_df["finish_time"], errors="coerce").replace(0, np.nan)
            speed_stats = dist_df.dropna(subset=["_speed"]).groupby(["horse_id", "race_distance"])["_speed"]\
                .mean().reset_index(name="horse_speed_at_distance")
            dist_stats = dist_stats.merge(speed_stats, on=["horse_id", "race_distance"], how="left")
            dist_stats["horse_speed_at_distance"] = dist_stats["horse_speed_at_distance"].fillna(0.0)

            df = df.merge(
                dist_stats[["horse_id", "race_distance", "horse_win_rate_at_distance", "horse_speed_at_distance"]],
                on=["horse_id", "race_distance"], how="left",
            )
            df["horse_win_rate_at_distance"] = df["horse_win_rate_at_distance"].fillna(0.0)
            df["horse_speed_at_distance"] = df["horse_speed_at_distance"].fillna(0.0)
        else:
            df["horse_win_rate_at_distance"] = 0.0
            df["horse_speed_at_distance"] = 0.0
        return df

    def _fetch_combo_features(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        combo_cols = [
            "horse_track_win_rate", "jockey_horse_win_rate",
            "jockey_track_win_rate", "recent_rank_trend", "weight_trend",
        ]
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids or "meet_code" not in df.columns:
            for col in combo_cols:
                df[col] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, jk_no, final_rank, race_date, horse_weight, races(meet_code)")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).execute()
        if not res.data:
            for col in combo_cols:
                df[col] = 0.0
            return df

        hist = pd.DataFrame([{
            "horse_id": r["horse_id"],
            "jk_no": r.get("jk_no"),
            "final_rank": r["final_rank"],
            "race_date": r["race_date"],
            "horse_weight": r.get("horse_weight") or 0,
            "meet_code": (r.get("races") or {}).get("meet_code"),
        } for r in res.data])
        hist["_is_win"] = (hist["final_rank"] == 1).astype(float)

        # horse_track_win_rate
        ht = hist.groupby(["horse_id", "meet_code"]).agg(
            _w=("_is_win", "sum"), _r=("_is_win", "count")
        ).reset_index()
        ht["horse_track_win_rate"] = ht["_w"] / (ht["_r"] + 1)
        df = df.merge(ht[["horse_id", "meet_code", "horse_track_win_rate"]],
                      on=["horse_id", "meet_code"], how="left")
        df["horse_track_win_rate"] = df["horse_track_win_rate"].fillna(0.0)

        # jockey_horse_win_rate
        jh = hist.groupby(["horse_id", "jk_no"]).agg(
            _w=("_is_win", "sum"), _r=("_is_win", "count")
        ).reset_index()
        jh["jockey_horse_win_rate"] = jh["_w"] / (jh["_r"] + 1)
        if "jk_no" in df.columns:
            df = df.merge(jh[["horse_id", "jk_no", "jockey_horse_win_rate"]],
                          on=["horse_id", "jk_no"], how="left")
        if "jockey_horse_win_rate" not in df.columns:
            df["jockey_horse_win_rate"] = 0.0
        df["jockey_horse_win_rate"] = df["jockey_horse_win_rate"].fillna(0.0)

        # jockey_track_win_rate
        jt = hist.groupby(["jk_no", "meet_code"]).agg(
            _w=("_is_win", "sum"), _r=("_is_win", "count")
        ).reset_index()
        jt["jockey_track_win_rate"] = jt["_w"] / (jt["_r"] + 1)
        if "jk_no" in df.columns:
            df = df.merge(jt[["jk_no", "meet_code", "jockey_track_win_rate"]],
                          on=["jk_no", "meet_code"], how="left")
        if "jockey_track_win_rate" not in df.columns:
            df["jockey_track_win_rate"] = 0.0
        df["jockey_track_win_rate"] = df["jockey_track_win_rate"].fillna(0.0)

        # recent_rank_trend
        hist_s = hist.sort_values(["horse_id", "race_date"])
        trend_map = {}
        for hid, grp in hist_s.groupby("horse_id"):
            last3 = grp["final_rank"].values[-3:]
            trend_map[hid] = (
                float(np.polyfit(np.arange(len(last3)), last3, 1)[0])
                if len(last3) >= 2 else 0.0
            )
        df["recent_rank_trend"] = df["horse_id"].map(trend_map).fillna(0.0)

        # weight_trend
        wt_map = {}
        for hid, grp in hist_s.groupby("horse_id"):
            wts = grp["horse_weight"].values
            diffs = np.diff(wts.astype(float))
            last3d = diffs[-3:] if len(diffs) >= 1 else np.array([0.0])
            wt_map[hid] = float(last3d.mean())
        df["weight_trend"] = df["horse_id"].map(wt_map).fillna(0.0)

        return df

    # ──────────────────────────────────
    # 학습용 피처 메서드
    # ──────────────────────────────────

    def _fetch_history_for_training(self, df: pd.DataFrame, end_date: str, client) -> pd.DataFrame:
        all_hist = []
        page_size, offset = 1000, 0
        while True:
            res = client.table("horse_race_history").select(
                "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, "
                "prev_5_avg_rank, prev_total_avg_rank"
            ).lte("as_of_date", end_date)\
             .order("horse_id").order("as_of_date")\
             .range(offset, offset + page_size - 1).execute()
            if not res.data:
                break
            all_hist.extend(res.data)
            if len(res.data) < page_size:
                break
            offset += page_size

        if not all_hist:
            logger.warning("horse_race_history 비어있음 – rebuild 먼저 실행하세요")
            return df

        hist_df = pd.DataFrame(all_hist)
        hist_df["as_of_date"] = pd.to_datetime(hist_df["as_of_date"])
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        merged = pd.merge_asof(
            df.sort_values("_race_dt").reset_index(drop=True),
            hist_df.sort_values("as_of_date").reset_index(drop=True)
                   .rename(columns={"as_of_date": "_race_dt"}),
            on="_race_dt", by="horse_id",
            direction="backward", allow_exact_matches=False,
        )
        df = merged.drop(columns=["_race_dt"])
        logger.info(f"horse_race_history 조인 완료: {len(all_hist)}개 행")
        return df

    def _fetch_training_grades_for_training(self, df: pd.DataFrame, client) -> pd.DataFrame:
        horse_names = df["horse_name"].dropna().unique().tolist() if "horse_name" in df.columns else []
        if not horse_names:
            df["training_grade_enc"] = 1.0
            return df
        tr_all_data = []
        for i in range(0, len(horse_names), 100):
            res = client.table("horse_training_sessions")\
                .select("hrnm, remk_txt, trng_dt")\
                .in_("hrnm", horse_names[i:i + 100]).execute()
            if res.data:
                tr_all_data.extend(res.data)

        if not tr_all_data:
            df["training_grade_enc"] = 1.0
            return df

        tr_df = pd.DataFrame(tr_all_data)
        tr_df["trng_dt"] = pd.to_datetime(tr_df["trng_dt"])
        tr_df["_grade"] = tr_df["remk_txt"].apply(_training_grade)
        df["_race_dt"] = pd.to_datetime(df["race_date"])

        grade_rows = []
        for _, row in df[["horse_name", "_race_dt"]].drop_duplicates().iterrows():
            name, race_dt = row["horse_name"], row["_race_dt"]
            cutoff = race_dt - pd.Timedelta(days=7)
            sub = tr_df[
                (tr_df["hrnm"] == name)
                & (tr_df["trng_dt"] >= cutoff)
                & (tr_df["trng_dt"] < race_dt)
            ]
            grade_rows.append({
                "horse_name": name,
                "_race_dt": race_dt,
                "training_grade_enc": float(int(sub["_grade"].min()) if len(sub) else 1),
            })

        grade_df = pd.DataFrame(grade_rows)
        df = df.merge(grade_df, on=["horse_name", "_race_dt"], how="left")
        df["training_grade_enc"] = df["training_grade_enc"].fillna(1.0)
        df = df.drop(columns=["_race_dt"])
        logger.info(f"조교 등급 조인 완료: {(df['training_grade_enc'] != 1).sum()}건 반영")
        return df

    def _fetch_prev_odds_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        if "win_odds" not in df.columns or "horse_id" not in df.columns:
            df["prev_win_odds"] = 15.0
            return df
        odds_df = df[["horse_id", "race_date", "win_odds"]].copy()
        odds_df["win_odds"] = pd.to_numeric(odds_df["win_odds"], errors="coerce")
        odds_df = odds_df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
        odds_df["prev_win_odds"] = odds_df.groupby("horse_id")["win_odds"].shift(1)
        df = df.merge(
            odds_df[["horse_id", "race_date", "prev_win_odds"]],
            on=["horse_id", "race_date"], how="left",
        )
        median_odds = odds_df["win_odds"].median()
        df["prev_win_odds"] = df["prev_win_odds"].fillna(median_odds)
        logger.info(f"prev_win_odds 조인 완료 (중앙값 폴백={median_odds:.1f})")
        return df

    def _fetch_speed_features_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        if "finish_time" not in df.columns or "race_distance" not in df.columns:
            df["prev_speed_figure"] = 0.0
            df["speed_figure_trend"] = 0.0
            return df
        spd_df = df[["horse_id", "race_date", "finish_time", "race_distance"]].copy()
        spd_df["finish_time"] = pd.to_numeric(spd_df["finish_time"], errors="coerce")
        spd_df["race_distance"] = pd.to_numeric(spd_df["race_distance"], errors="coerce")
        spd_df = spd_df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
        spd_df["_speed"] = spd_df["race_distance"] / spd_df["finish_time"].replace(0, np.nan)
        spd_df["prev_speed_figure"] = spd_df.groupby("horse_id")["_speed"].shift(1)

        def _speed_slope(s):
            arr = s.values
            out = np.zeros(len(arr))
            for i in range(len(arr)):
                past = arr[max(0, i - 3):i]
                past = past[~np.isnan(past)]
                if len(past) >= 2:
                    out[i] = float(np.polyfit(np.arange(len(past)), past, 1)[0])
            return pd.Series(out, index=s.index)

        spd_df["speed_figure_trend"] = spd_df.groupby("horse_id")["_speed"].transform(_speed_slope)
        df = df.merge(
            spd_df[["horse_id", "race_date", "prev_speed_figure", "speed_figure_trend"]],
            on=["horse_id", "race_date"], how="left",
        )
        df["prev_speed_figure"] = df["prev_speed_figure"].fillna(0.0)
        df["speed_figure_trend"] = df["speed_figure_trend"].fillna(0.0)
        n_valid = (df["prev_speed_figure"] > 0).sum()
        logger.info(f"prev_speed_figure 계산 완료 (유효 {n_valid}건)")
        return df

    def _fetch_assigned_weight(self, df: pd.DataFrame, client) -> pd.DataFrame:
        if "race_id" not in df.columns or "horse_id" not in df.columns:
            df["assigned_weight"] = 56.0
            return df
        race_ids = df["race_id"].dropna().unique().tolist()
        if not race_ids:
            df["assigned_weight"] = 56.0
            return df
        res = client.table("race_entry_detail")\
            .select("race_id, horse_id, assigned_weight")\
            .in_("race_id", race_ids).execute()
        return self._join_assigned_weight(df, res.data or [])

    def _fetch_assigned_weight_for_training(self, df: pd.DataFrame, detail_rows: list = None) -> pd.DataFrame:
        if detail_rows is None:
            if "race_id" not in df.columns:
                df["assigned_weight"] = 56.0
                return df
            client = get_client()
            race_ids = df["race_id"].dropna().unique().tolist()
            all_rows = []
            for i in range(0, len(race_ids), 500):
                res = client.table("race_entry_detail")\
                    .select("race_id, horse_id, assigned_weight")\
                    .in_("race_id", race_ids[i:i + 500]).execute()
                all_rows.extend(res.data or [])
            detail_rows = all_rows
        return self._join_assigned_weight(df, detail_rows)

    def _join_assigned_weight(self, df: pd.DataFrame, detail_rows: list) -> pd.DataFrame:
        if not detail_rows:
            df["assigned_weight"] = 56.0
            return df
        detail_df = pd.DataFrame(detail_rows)[["race_id", "horse_id", "assigned_weight"]]
        detail_df["assigned_weight"] = pd.to_numeric(detail_df["assigned_weight"], errors="coerce")
        df = df.merge(detail_df, on=["race_id", "horse_id"], how="left")
        median_weight = detail_df["assigned_weight"].median()
        if pd.isna(median_weight):
            median_weight = 56.0
        n_missing = df["assigned_weight"].isna().sum()
        df["assigned_weight"] = df["assigned_weight"].fillna(median_weight)
        logger.info(f"assigned_weight 조인 완료 (결측 {n_missing}건 → 중앙값 {median_weight:.1f}로 대체)")
        return df
