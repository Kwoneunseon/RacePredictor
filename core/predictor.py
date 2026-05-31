import pandas as pd
import numpy as np
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


class RacePredictor:
    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.trainer.feature_columns
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)

        rank_scores = self.trainer.predict_rank_score(X)
        if rank_scores is not None:
            scores = rank_scores
        else:
            scores = self.trainer.predict_proba(X)

        result = df.copy()
        result["win_probability"] = scores

        # 경주별 softmax: rank score → 경주 내 합계 100% 확률
        if "race_id" in result.columns and rank_scores is not None:
            def softmax(x):
                e = np.exp(x - x.max())
                return e / e.sum()
            result["win_probability"] = result.groupby("race_id")["win_probability"].transform(
                lambda x: softmax(x.values)
            )

        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        logger.info(f"예측 완료: {len(result)}마리, 최고 확률={result['win_probability'].max():.3f}")
        return result

    def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        from Utils.database import get_client
        client = get_client()

        query = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).eq("race_date", race_date)

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

        # 마체중 0인 경우 KRA 사이트에서 당일 스크래핑
        if "horse_weight" in df.columns and "entry_number" in df.columns:
            zero_races = df[df["horse_weight"].isna() | (df["horse_weight"] == 0)][["meet_code", "race_no"]].drop_duplicates()
            if not zero_races.empty:
                from library.scraper_weights import fetch_race_weights
                for _, row in zero_races.iterrows():
                    mc, rn = int(row["meet_code"]), int(row["race_no"])
                    weights = fetch_race_weights(mc, rn, race_date)
                    for entry_no, wt in weights.items():
                        mask = (df["meet_code"] == mc) & (df["race_no"] == rn) & (df["entry_number"] == entry_no)
                        if mask.any():
                            df.loc[mask, "horse_weight"] = wt["horse_weight"]

        # jockey/trainer 최근 30일 승률 계산
        if "jk_no" in df.columns:
            import pandas as _pd
            cutoff = (_pd.to_datetime(race_date) - _pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            hist_30d_res = client.table("race_entries")\
                .select("jk_no, trainer_id, final_rank")\
                .gte("race_date", cutoff)\
                .lt("race_date", race_date)\
                .not_.is_("final_rank", "null")\
                .execute()
            if hist_30d_res.data:
                hist_30d = _pd.DataFrame(hist_30d_res.data)
                hist_30d["_is_win"] = (hist_30d["final_rank"] == 1).astype(int)
                jk_rates = hist_30d.groupby("jk_no")["_is_win"].mean().to_dict()
                df["jockey_recent_win_rate"] = df["jk_no"].map(jk_rates).fillna(0.0)
                if "trainer_id" in df.columns:
                    tr_rates = hist_30d.groupby("trainer_id")["_is_win"].mean().to_dict()
                    df["trainer_recent_win_rate"] = df["trainer_id"].map(tr_rates).fillna(0.0)
                else:
                    df["trainer_recent_win_rate"] = 0.0
            else:
                df["jockey_recent_win_rate"] = 0.0
                df["trainer_recent_win_rate"] = 0.0

        # 조교현황 training_grade_enc: 양호→2, 기록없음→1, 불량계열→0
        if "horse_name" in df.columns:
            horse_names = df["horse_name"].dropna().unique().tolist()
            race_dt = pd.to_datetime(race_date)
            cutoff_tr = (race_dt - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            tr_res = client.table("horse_training_sessions")\
                .select("hrnm, remk_txt")\
                .in_("hrnm", horse_names)\
                .gte("trng_dt", cutoff_tr)\
                .lt("trng_dt", race_date)\
                .execute()
            if tr_res.data:
                grade_map: dict = {}
                for row in tr_res.data:
                    name = row["hrnm"]
                    grade = _training_grade(row.get("remk_txt", ""))
                    # 같은 말의 여러 기록 중 최악 등급 반영
                    grade_map[name] = min(grade_map.get(name, 2), grade)
                df["training_grade_enc"] = df["horse_name"].apply(
                    lambda n: float(grade_map.get(n, 1))
                )
            else:
                df["training_grade_enc"] = 1.0

        # horse_race_history 조인: 해당 날짜 이하의 가장 최신 통계
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if horse_ids:
            hist_res = client.table("horse_race_history").select(
                "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, prev_5_avg_rank, prev_total_avg_rank"
            ).in_("horse_id", horse_ids).lt("as_of_date", race_date).execute()

            if hist_res.data:
                hist_df = pd.DataFrame(hist_res.data)
                hist_latest = (
                    hist_df.sort_values("as_of_date", ascending=False)
                           .drop_duplicates(subset=["horse_id"], keep="first")
                           .drop(columns=["as_of_date"])
                )
                df = df.merge(hist_latest, on="horse_id", how="left")

        # days_since_last_race, horse_weight_diff
        if horse_ids:
            last_res = client.table("race_entries")\
                .select("horse_id, race_date, horse_weight")\
                .in_("horse_id", horse_ids)\
                .lt("race_date", race_date)\
                .gt("final_rank", 0)\
                .execute()
            if last_res.data:
                last_df = pd.DataFrame(last_res.data)
                last_df = (last_df.sort_values("race_date", ascending=False)
                                  .drop_duplicates(subset=["horse_id"], keep="first"))
                race_dt = pd.to_datetime(race_date)
                last_df["_days"] = (race_dt - pd.to_datetime(last_df["race_date"])).dt.days
                days_map = last_df.set_index("horse_id")["_days"].to_dict()
                prev_wt_map = last_df.set_index("horse_id")["horse_weight"].to_dict()
                df["days_since_last_race"] = df["horse_id"].map(days_map).fillna(60)
                df["_prev_weight"] = df["horse_id"].map(prev_wt_map).fillna(0)
                # horse_weight=0(미계체)인 경우 직전 경주 체중으로 대체
                zero_mask = df["horse_weight"].isna() | (df["horse_weight"] == 0)
                df.loc[zero_mask, "horse_weight"] = df.loc[zero_mask, "_prev_weight"]
                df["horse_weight_diff"] = df["horse_weight"].fillna(0) - df["_prev_weight"].fillna(0)
                df = df.drop(columns=["_prev_weight"])
            else:
                df["days_since_last_race"] = 60
                df["horse_weight_diff"] = 0.0

        # horse_win_rate_at_distance
        if horse_ids and "race_distance" in df.columns:
            dist_res = client.table("race_entries")\
                .select("horse_id, final_rank, races(race_distance)")\
                .in_("horse_id", horse_ids)\
                .lt("race_date", race_date)\
                .gt("final_rank", 0)\
                .execute()
            if dist_res.data:
                dist_rows = [
                    {
                        "horse_id": r["horse_id"],
                        "final_rank": r["final_rank"],
                        "race_distance": (r.get("races") or {}).get("race_distance"),
                    }
                    for r in dist_res.data
                ]
                dist_df = pd.DataFrame(dist_rows).dropna(subset=["race_distance"])
                dist_df["_is_win"] = (dist_df["final_rank"] == 1).astype(float)
                dist_stats = dist_df.groupby(["horse_id", "race_distance"]).agg(
                    _wins=("_is_win", "sum"),
                    _races=("_is_win", "count"),
                ).reset_index()
                dist_stats["horse_win_rate_at_distance"] = dist_stats["_wins"] / (dist_stats["_races"] + 1)
                df = df.merge(
                    dist_stats[["horse_id", "race_distance", "horse_win_rate_at_distance"]],
                    on=["horse_id", "race_distance"],
                    how="left",
                )
                df["horse_win_rate_at_distance"] = df["horse_win_rate_at_distance"].fillna(0.0)
            else:
                df["horse_win_rate_at_distance"] = 0.0

        # 신규 피처: horse_track_win_rate, jockey_horse_win_rate,
        #           jockey_track_win_rate, recent_rank_trend, weight_trend
        if horse_ids and "meet_code" in df.columns:
            combo_res = client.table("race_entries")\
                .select("horse_id, jk_no, final_rank, race_date, horse_weight, races(meet_code)")\
                .in_("horse_id", horse_ids)\
                .lt("race_date", race_date)\
                .gt("final_rank", 0)\
                .execute()
            if combo_res.data:
                hist = pd.DataFrame([{
                    "horse_id": r["horse_id"],
                    "jk_no": r.get("jk_no"),
                    "final_rank": r["final_rank"],
                    "race_date": r["race_date"],
                    "horse_weight": r.get("horse_weight") or 0,
                    "meet_code": (r.get("races") or {}).get("meet_code"),
                } for r in combo_res.data])
                hist["_is_win"] = (hist["final_rank"] == 1).astype(float)

                # horse_track_win_rate
                ht = hist.groupby(["horse_id", "meet_code"]).agg(
                    _w=("_is_win", "sum"), _r=("_is_win", "count")
                ).reset_index()
                ht["horse_track_win_rate"] = ht["_w"] / (ht["_r"] + 1)
                df = df.merge(
                    ht[["horse_id", "meet_code", "horse_track_win_rate"]],
                    on=["horse_id", "meet_code"], how="left"
                )
                df["horse_track_win_rate"] = df["horse_track_win_rate"].fillna(0.0)

                # jockey_horse_win_rate
                jh = hist.groupby(["horse_id", "jk_no"]).agg(
                    _w=("_is_win", "sum"), _r=("_is_win", "count")
                ).reset_index()
                jh["jockey_horse_win_rate"] = jh["_w"] / (jh["_r"] + 1)
                if "jk_no" in df.columns:
                    df = df.merge(
                        jh[["horse_id", "jk_no", "jockey_horse_win_rate"]],
                        on=["horse_id", "jk_no"], how="left"
                    )
                if "jockey_horse_win_rate" not in df.columns:
                    df["jockey_horse_win_rate"] = 0.0
                df["jockey_horse_win_rate"] = df["jockey_horse_win_rate"].fillna(0.0)

                # jockey_track_win_rate
                jt = hist.groupby(["jk_no", "meet_code"]).agg(
                    _w=("_is_win", "sum"), _r=("_is_win", "count")
                ).reset_index()
                jt["jockey_track_win_rate"] = jt["_w"] / (jt["_r"] + 1)
                if "jk_no" in df.columns:
                    df = df.merge(
                        jt[["jk_no", "meet_code", "jockey_track_win_rate"]],
                        on=["jk_no", "meet_code"], how="left"
                    )
                if "jockey_track_win_rate" not in df.columns:
                    df["jockey_track_win_rate"] = 0.0
                df["jockey_track_win_rate"] = df["jockey_track_win_rate"].fillna(0.0)

                # recent_rank_trend
                hist_s = hist.sort_values(["horse_id", "race_date"])
                trend_map = {}
                for hid, grp in hist_s.groupby("horse_id"):
                    last3 = grp["final_rank"].values[-3:]
                    if len(last3) >= 2:
                        slope = float(np.polyfit(np.arange(len(last3)), last3, 1)[0])
                    else:
                        slope = 0.0
                    trend_map[hid] = slope
                df["recent_rank_trend"] = df["horse_id"].map(trend_map).fillna(0.0)

                # weight_trend
                wt_map = {}
                for hid, grp in hist_s.groupby("horse_id"):
                    wts = grp["horse_weight"].values
                    diffs = np.diff(wts.astype(float))
                    last3d = diffs[-3:] if len(diffs) >= 1 else np.array([0.0])
                    wt_map[hid] = float(last3d.mean())
                df["weight_trend"] = df["horse_id"].map(wt_map).fillna(0.0)
            else:
                for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                            "jockey_track_win_rate", "recent_rank_trend", "weight_trend"]:
                    df[col] = 0.0
        else:
            for col in ["horse_track_win_rate", "jockey_horse_win_rate",
                        "jockey_track_win_rate", "recent_rank_trend", "weight_trend"]:
                df[col] = 0.0

        df = self.preprocessor.transform(df)
        return self.predict(df)


def predict_today(race_date: str) -> dict:
    """
    스케줄러 등 외부에서 재사용할 수 있는 예측 함수.
    반환: {
        "predictions": DataFrame,
        "race_info":   {race_id: {...}},
        "eligible_ids": set,
        "meet_name":   {meet_code: str},
    }
    데이터 없으면 None 반환.
    """
    from Utils.database import get_client
    from core.preprocessor import RacePreprocessor
    from core.trainer import RaceModelTrainer
    from core.model_manager import ModelManager
    from core.race_filter import RaceFilter

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음")
        return None

    predictor = RacePredictor(trainer, preprocessor)
    predictions = predictor.predict_from_db(race_date)
    if predictions.empty:
        return None

    client = get_client()
    races_res = client.table("races").select(
        "race_id, race_distance, race_grade, total_horses, race_no, meet_code"
    ).eq("race_date", race_date).execute()
    race_info = {r["race_id"]: r for r in (races_res.data or [])}

    race_filter = RaceFilter()
    eligible_ids = {rid for rid, info in race_info.items() if race_filter.is_eligible(info)}

    return {
        "predictions":  predictions,
        "race_info":    race_info,
        "eligible_ids": eligible_ids,
        "meet_name":    {1: "서울", 2: "제주", 3: "부산", 0: "미상"},
    }


def format_kakao_message(result: dict, race_date: str) -> str:
    """predict_today() 결과를 카카오톡 메시지 문자열로 변환"""
    predictions  = result["predictions"]
    race_info    = result["race_info"]
    eligible_ids = result["eligible_ids"]
    meet_name    = result["meet_name"]

    # 경마장별로 묶기
    groups: dict = {}
    for race_id, info in sorted(race_info.items(),
                                key=lambda x: (x[1].get("meet_code", 0),
                                               x[1].get("race_no", 0))):
        meet = info.get("meet_code", 0)
        groups.setdefault(meet, []).append((race_id, info))

    lines = [f"[경마 예측] {race_date}", ""]

    total_races = 0
    for meet_code, race_list in sorted(groups.items()):
        name = meet_name.get(meet_code, "미상")
        lines.append(f"■ {name} ({len(race_list)}경주)")

        for race_id, info in race_list:
            race_no = info.get("race_no", "?")
            dist    = info.get("race_distance", "?")
            flag    = "★" if race_id in eligible_ids else "  "

            # 해당 경주 1위 예측마
            race_preds = predictions[predictions["race_id"] == race_id]
            if race_preds.empty:
                continue
            top = race_preds.sort_values("win_probability", ascending=False).iloc[0]
            horse = top.get("horse_name") or top.get("horse_id", "?")
            prob  = top["win_probability"]

            lines.append(f" {flag}{race_no}경주 {dist}m | {horse} {prob:.1%}")
            total_races += 1

        lines.append("")

    lines.append(f"총 {total_races}경주 예측완료")
    return "\n".join(lines)
