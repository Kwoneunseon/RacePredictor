"""DB에서 피처를 수집·계산해 DataFrame으로 반환하는 클래스"""
import numpy as np
import pandas as pd
import library.scraper_odds as scraper_odds
from config import NON_FINISH_RANK_MIN
from Utils.database import get_client
from Utils.logger import get_logger
from Utils.ranks import valid_rank_mask

logger = get_logger(__name__)

_SECTION_CORNER_APPLICABLE_DISTANCE = 1300  # 이 값 미만 거리는 1C/2C 데이터 없음 (라이브 API 실측으로 확정)


def _fetch_in_batches(client, table: str, select_cols: str, in_col: str, values: list,
                       chunk_size: int = 500, page_size: int = 1000, filter_fn=None) -> list:
    """PostgREST는 명시적으로 range()를 걸지 않으면 응답을 기본 최대 행수(보통 1000)로
    자른다. in_col IN values 조회를 값 리스트만 chunk_size로 나누고 range() 없이 쏘면,
    한 청크에 실제로 일치하는 행이 1000건을 넘을 때 나머지가 조용히 잘려나간다
    (2026-07-20 발견 — 신규 피처 학습용 조인이 대부분 결측으로 빠지던 근본 원인).
    청크 분할 + 청크 내부 range() 페이지네이션을 함께 해야 안전하다."""
    rows = []
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i + chunk_size]
        offset = 0
        while True:
            q = client.table(table).select(select_cols).in_(in_col, chunk)
            if filter_fn:
                q = filter_fn(q)
            res = q.range(offset, offset + page_size - 1).execute()
            data = res.data or []
            rows.extend(data)
            if len(data) < page_size:
                break
            offset += page_size
    return rows


class RaceFeatureBuilder:
    """
    race_entries 기반 피처 수집 클래스.

    - build_for_prediction(race_date): 예측용. DB에서 race_entries 조회 후 피처 전부 붙임.
    - build_for_training(df, end_date): 학습용. 이미 조회된 df에 피처 조인.
    - get_flatten_entries(data): 중첩된 Supabase 조인 결과를 평평한 DataFrame으로 변환.
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

        df = self.get_flatten_entries(result.data)

        if meet_code is not None:
            df = df[df["meet_code"] == meet_code].reset_index(drop=True)
            if df.empty:
                return df

        df = self._fetch_weights(df, race_date, client)
        df = self._fetch_recent_rates(df, race_date, client)
        df = self._fetch_history(df, race_date, client)
        df = self._fetch_last_race_features(df, race_date, client)
        df = self._fetch_prev_odds(df, race_date, client)
        df = self._fetch_current_odds(df, race_date)
        df = self._fetch_speed_features(df, race_date, client)
        df = self._fetch_assigned_weight(df, client)
        df = self._fetch_section_and_career_features(df, client)
        df = self._fetch_running_style(df, race_date, client)
        df = self._fetch_distance_rates(df, race_date, client)
        df = self._fetch_combo_features(df, race_date, client)
        return df

    def build_for_training(self, df: pd.DataFrame, end_date: str) -> pd.DataFrame:
        client = get_client()

        df = self._fetch_history_for_training(df, end_date, client)
        df = self._fetch_prev_odds_for_training(df)
        df = self._fetch_speed_features_for_training(df)
        df = self._fetch_assigned_weight_for_training(df)
        df = self._fetch_section_and_career_features_for_training(df)
        df = self._fetch_running_style_for_training(df)
        return df

    @staticmethod
    def get_flatten_entries(data: list) -> pd.DataFrame:
        """중첩된 Supabase 조인 결과(dict)를 평평한 DataFrame으로 변환. 순수 변환이며 인스턴스 상태를 쓰지 않음."""
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
            .gt("final_rank", 0).lt("final_rank", NON_FINISH_RANK_MIN).execute()
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
            .gt("final_rank", 0).lt("final_rank", NON_FINISH_RANK_MIN).execute()
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
            .gt("final_rank", 0).lt("final_rank", NON_FINISH_RANK_MIN)\
            .not_.is_("finish_time", "null").execute()
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
            .gt("final_rank", 0).lt("final_rank", NON_FINISH_RANK_MIN).execute()
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
            .gt("final_rank", 0).lt("final_rank", NON_FINISH_RANK_MIN).execute()
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
        rows = _fetch_in_batches(client, "race_entry_detail", "race_id, horse_id, assigned_weight",
                                  "race_id", race_ids)
        return self._join_assigned_weight(df, rows)

    def _fetch_assigned_weight_for_training(self, df: pd.DataFrame, detail_rows: list = None) -> pd.DataFrame:
        if detail_rows is None:
            if "race_id" not in df.columns:
                df["assigned_weight"] = 56.0
                return df
            client = get_client()
            race_ids = df["race_id"].dropna().unique().tolist()
            detail_rows = _fetch_in_batches(client, "race_entry_detail",
                                             "race_id, horse_id, assigned_weight", "race_id", race_ids)
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

    def _join_section_records(self, df: pd.DataFrame, section_rows: list) -> pd.DataFrame:
        section_cols = [
            "s1f_avg", "c1_avg", "c2_avg", "c3_avg", "c4_avg", "g3f_avg", "g1f_avg",
        ]
        if not section_rows or "horse_id" not in df.columns or "race_date" not in df.columns:
            for col in section_cols:
                df[col] = 0.0
            return df

        sec_df = pd.DataFrame(section_rows)
        sec_df["snapshot_race_date"] = pd.to_datetime(sec_df["snapshot_race_date"])
        # 짧은 거리는 1C/2C가 0으로 나옴 → 실제 기록이 아니라 "미해당" 이므로 NaN 처리
        short_mask = sec_df["race_distance"] < _SECTION_CORNER_APPLICABLE_DISTANCE
        sec_df.loc[short_mask, ["c1_avg", "c2_avg"]] = np.nan

        df = df.copy()
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        df["_orig_idx"] = df.index
        merged_parts = []
        for hid, grp in df.groupby("horse_id"):
            sub = sec_df[sec_df["horse_id"] == hid].sort_values("snapshot_race_date")
            if sub.empty:
                part = grp.copy()
                for col in section_cols:
                    part[col] = np.nan
                merged_parts.append(part)
                continue
            # sub의 horse_id/race_distance는 df에도 동명 컬럼이 있어 merge_asof가
            # _x/_y로 접미사를 붙여버린다(원본 race_distance/horse_id가 사라지는 사고로
            # 이어짐 — 2026-07-19 재학습에서 dropna(race_distance)가 대량으로 행을
            # 지운 원인). 여기서는 이미 horse_id로 그룹핑했고 race_distance는 위에서
            # short_mask 계산에 다 썼으므로 병합 전에 제거한다.
            sub = sub.drop(columns=["horse_id", "race_distance"])
            # merge_asof는 결과에 새 RangeIndex를 부여해 원본 인덱스를 버린다. 말별로
            # 그룹핑해서 각각 merge_asof를 돌리면 말마다 인덱스가 0부터 다시 시작해
            # concat 후 인덱스가 중복되고, 그 뒤 sort_index()가 서로 다른 말의 행을
            # 인덱스 값 기준으로 뒤섞어버린다(_orig_idx 컬럼으로 원래 인덱스를 보존해
            # 마지막에 복원 — 2026-07-19 인기마 기준선 수치가 흔들린 원인).
            part = pd.merge_asof(
                grp.sort_values("_race_dt"),
                sub.rename(columns={"snapshot_race_date": "_race_dt"}),
                on="_race_dt", direction="backward", allow_exact_matches=False,
            )
            merged_parts.append(part)
        df = pd.concat(merged_parts).set_index("_orig_idx").sort_index()
        df.index.name = None
        df = df.drop(columns=["_race_dt"])

        for col in section_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            if n_missing:
                logger.info(f"{col} 결측 {n_missing}건 → 중앙값 {median_val:.2f}로 대체")
        return df

    def _join_career_stats(self, df: pd.DataFrame, detail_rows: list) -> pd.DataFrame:
        # horse_recent1y_wins/starts는 API에 소스 필드가 없어 항상 NULL이므로
        # (library/collector_race_detail.py parse_item 참고) 여기서 다루지 않는다.
        rate_specs = [
            ("horse_career_win_rate", "horse_career_wins", "horse_career_starts"),
            ("jockey_career_win_rate", "jockey_career_wins", "jockey_career_starts"),
            ("jockey_recent1y_win_rate", "jockey_recent1y_wins", "jockey_recent1y_starts"),
            ("trainer_career_win_rate", "trainer_career_wins", "trainer_career_starts"),
            ("trainer_recent1y_win_rate", "trainer_recent1y_wins", "trainer_recent1y_starts"),
        ]
        sample_cols = ["jockey_recent1y_starts", "trainer_recent1y_starts"]
        if not detail_rows or "race_id" not in df.columns or "horse_id" not in df.columns:
            for rate_col, _, _ in rate_specs:
                df[rate_col] = 0.0
            for col in sample_cols:
                df[col] = 0.0
            return df

        detail_df = pd.DataFrame(detail_rows)
        needed_cols = [c for _, w, s in rate_specs for c in (w, s) if c in detail_df.columns]
        df = df.merge(
            detail_df[["race_id", "horse_id"] + needed_cols].drop_duplicates(),
            on=["race_id", "horse_id"], how="left",
        )
        for rate_col, win_col, start_col in rate_specs:
            if start_col not in df.columns or win_col not in df.columns:
                df[rate_col] = 0.0
                continue
            starts = pd.to_numeric(df[start_col], errors="coerce")
            wins = pd.to_numeric(df[win_col], errors="coerce")
            df[rate_col] = (wins / starts.replace(0, np.nan)).fillna(0.0)
        for col in sample_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
        return df

    def _fetch_section_and_career_features(self, df: pd.DataFrame, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist() if "horse_id" in df.columns else []
        race_ids = df["race_id"].dropna().unique().tolist() if "race_id" in df.columns else []
        sec_rows, detail_rows = [], []
        if horse_ids:
            sec_rows = _fetch_in_batches(client, "horse_section_records", "*", "horse_id", horse_ids)
        if race_ids:
            detail_rows = _fetch_in_batches(
                client, "race_entry_detail",
                "race_id, horse_id, horse_career_wins, horse_career_starts, "
                "jockey_career_wins, jockey_career_starts, "
                "jockey_recent1y_wins, jockey_recent1y_starts, trainer_career_wins, trainer_career_starts, "
                "trainer_recent1y_wins, trainer_recent1y_starts",
                "race_id", race_ids)
        df = self._join_section_records(df, sec_rows)
        df = self._join_career_stats(df, detail_rows)
        return df

    def _compute_running_style(self, df: pd.DataFrame, past_rows: list) -> pd.DataFrame:
        if not past_rows or "horse_id" not in df.columns:
            df["running_style_score"] = 0.0
            return df
        past_df = pd.DataFrame(past_rows)
        # 1코너 순위 우선, 없으면(짧은 거리) 3코너 순위로 대체
        past_df["_early_ord"] = past_df["sj_1c_ord"].fillna(past_df["sj_3c_ord"])
        past_df = past_df.dropna(subset=["_early_ord", "final_rank"])
        past_df = past_df[valid_rank_mask(past_df["final_rank"])].copy()  # 0·90번대(비착순) 제외
        past_df["_delta"] = pd.to_numeric(past_df["_early_ord"], errors="coerce") - \
            pd.to_numeric(past_df["final_rank"], errors="coerce")
        style_map = past_df.groupby("horse_id")["_delta"].mean().to_dict()
        df["running_style_score"] = df["horse_id"].map(style_map).fillna(0.0)
        return df

    def _fetch_running_style(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        # race_entry_detail에는 races/race_entries로의 FK 제약이 없어 PostgREST
        # 임베디드 조인(races(...), race_entries(...))이 스키마캐시에서 관계를
        # 찾지 못해 실패한다(PGRST200). 학습용(_fetch_running_style_for_training)과
        # 동일하게 수동으로 두 번 조회해 파이썬에서 조인한다.
        horse_ids = df["horse_id"].dropna().unique().tolist() if "horse_id" in df.columns else []
        if not horse_ids:
            df["running_style_score"] = 0.0
            return df
        detail_rows = _fetch_in_batches(client, "race_entry_detail",
                                         "horse_id, race_id, sj_1c_ord, sj_3c_ord", "horse_id", horse_ids)
        if not detail_rows:
            df["running_style_score"] = 0.0
            return df
        race_ids = list({r["race_id"] for r in detail_rows})
        rank_rows = _fetch_in_batches(
            client, "race_entries", "race_id, horse_id, race_date, final_rank", "race_id", race_ids,
            filter_fn=lambda q: q.lt("race_date", race_date))
        rank_map = {(r["race_id"], r["horse_id"]): r for r in rank_rows}
        rows = []
        for r in detail_rows:
            rank_row = rank_map.get((r["race_id"], r["horse_id"]))
            if rank_row is None:
                continue
            rows.append({
                "horse_id": r["horse_id"],
                "sj_1c_ord": r.get("sj_1c_ord"),
                "sj_3c_ord": r.get("sj_3c_ord"),
                "final_rank": rank_row.get("final_rank"),
            })
        return self._compute_running_style(df, rows)

    def _fetch_running_style_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        if "horse_id" not in df.columns or "race_date" not in df.columns:
            df["running_style_score"] = 0.0
            return df
        client = get_client()
        horse_ids = df["horse_id"].dropna().unique().tolist()
        all_rows = _fetch_in_batches(client, "race_entry_detail",
                                      "horse_id, race_id, sj_1c_ord, sj_3c_ord", "horse_id", horse_ids)
        if not all_rows:
            df["running_style_score"] = 0.0
            return df
        detail_df = pd.DataFrame(all_rows)
        race_ids = detail_df["race_id"].unique().tolist()
        rank_rows = _fetch_in_batches(client, "race_entries",
                                       "race_id, horse_id, race_date, final_rank", "race_id", race_ids)
        rank_df = pd.DataFrame(rank_rows)
        merged = detail_df.merge(rank_df, on=["race_id", "horse_id"], how="inner")
        merged = merged[valid_rank_mask(merged["final_rank"])]  # 0·90번대(비착순) 제외

        df = df.copy()
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        merged["_race_dt"] = pd.to_datetime(merged["race_date"])
        parts = []
        for hid, grp in df.groupby("horse_id"):
            past = merged[merged["horse_id"] == hid]
            part = grp.copy()
            style_vals = []
            for _, row in grp.iterrows():
                past_before = past[past["_race_dt"] < row["_race_dt"]]
                if past_before.empty:
                    style_vals.append(0.0)
                    continue
                sub_df = past_before.copy()
                sub_df["_early_ord"] = sub_df["sj_1c_ord"].fillna(sub_df["sj_3c_ord"])
                sub_df = sub_df.dropna(subset=["_early_ord", "final_rank"])
                if sub_df.empty:
                    style_vals.append(0.0)
                    continue
                delta = pd.to_numeric(sub_df["_early_ord"], errors="coerce") - \
                    pd.to_numeric(sub_df["final_rank"], errors="coerce")
                style_vals.append(float(delta.mean()))
            part["running_style_score"] = style_vals
            parts.append(part)
        df = pd.concat(parts).sort_index().drop(columns=["_race_dt"])
        return df

    def _fetch_section_and_career_features_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        client = get_client()
        horse_ids = df["horse_id"].dropna().unique().tolist() if "horse_id" in df.columns else []
        race_ids = df["race_id"].dropna().unique().tolist() if "race_id" in df.columns else []

        sec_rows = _fetch_in_batches(client, "horse_section_records", "*", "horse_id", horse_ids)
        detail_rows = _fetch_in_batches(
            client, "race_entry_detail",
            "race_id, horse_id, horse_career_wins, horse_career_starts, "
            "jockey_career_wins, jockey_career_starts, "
            "jockey_recent1y_wins, jockey_recent1y_starts, trainer_career_wins, trainer_career_starts, "
            "trainer_recent1y_wins, trainer_recent1y_starts",
            "race_id", race_ids)

        df = self._join_section_records(df, sec_rows)
        df = self._join_career_stats(df, detail_rows)
        return df
