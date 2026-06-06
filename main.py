import argparse
from datetime import datetime

from config import MODEL_DIR
from Utils.database import get_client, test_connection
from Utils.logger import get_logger
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager

logger = get_logger(__name__)


def cmd_train(args):
    logger.info("=== 모델 학습 시작 ===")
    import pandas as pd
    client = get_client()

    all_entries = []
    page_size = 1000
    offset = 0
    while True:
        result = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).gte("race_date", args.start_date).lte("race_date", args.end_date)\
            .gt("final_rank", 0)\
            .order("id")\
            .range(offset, offset + page_size - 1).execute()
        if not result.data:
            break
        all_entries.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_entries:
        logger.error("학습 데이터 없음")
        return
    logger.info(f"race_entries 수집: {len(all_entries)}건")

    rows = []
    for r in all_entries:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}

        h = r.get("horses") or {}
        flat["horse_name"] = h.get("name")
        flat["horse_age"] = h.get("age")
        flat["gender"] = h.get("gender")
        flat["horse_class"] = h.get("rank")

        rc = r.get("races") or {}
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

    # horse_race_history 일괄 조인 (경주일 이전 가장 최신 stats 사용, 당일 결과 누수 방지)
    all_hist = []
    page_size = 1000
    offset = 0
    while True:
        hist_res = client.table("horse_race_history").select(
            "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, prev_5_avg_rank, prev_total_avg_rank"
        ).lte("as_of_date", args.end_date)\
            .order("horse_id").order("as_of_date")\
            .range(offset, offset + page_size - 1).execute()
        if not hist_res.data:
            break
        all_hist.extend(hist_res.data)
        if len(hist_res.data) < page_size:
            break
        offset += page_size

    if all_hist:
        hist_df = pd.DataFrame(all_hist)
        hist_df["as_of_date"] = pd.to_datetime(hist_df["as_of_date"])
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        hist_sorted = hist_df.sort_values("as_of_date").reset_index(drop=True)
        df_sorted = df.sort_values("_race_dt").reset_index(drop=True)
        # merge_asof: 경주일보다 strictly 이전인 가장 최신 기록 사용
        merged = pd.merge_asof(
            df_sorted,
            hist_sorted.rename(columns={"as_of_date": "_race_dt"}),
            on="_race_dt",
            by="horse_id",
            direction="backward",
            allow_exact_matches=False,
        )
        df = merged.drop(columns=["_race_dt"])
        logger.info(f"horse_race_history 조인 완료: {len(all_hist)}개 행 (당일 이전 기록만 사용)")
    else:
        logger.warning("horse_race_history 비어있음 – Task 1 SQL을 먼저 실행하세요")

    # 조교현황 training_grade_enc 조인: 경주일 전 7일 이내 기록 (배치 쿼리)
    from core.feature_builder import _training_grade
    _horse_names = df["horse_name"].dropna().unique().tolist() if "horse_name" in df.columns else []
    if _horse_names:
        _batch_size = 100
        tr_all_data = []
        for i in range(0, len(_horse_names), _batch_size):
            _batch = _horse_names[i:i + _batch_size]
            _res = client.table("horse_training_sessions")\
                .select("hrnm, remk_txt, trng_dt")\
                .in_("hrnm", _batch)\
                .execute()
            if _res.data:
                tr_all_data.extend(_res.data)
        if tr_all_data:
            tr_df = pd.DataFrame(tr_all_data)
            tr_df["trng_dt"] = pd.to_datetime(tr_df["trng_dt"])
            tr_df["_grade"] = tr_df["remk_txt"].apply(_training_grade)
            df["_race_dt"] = pd.to_datetime(df["race_date"])
            grade_rows = []
            for _, row in df[["horse_name", "_race_dt"]].drop_duplicates().iterrows():
                name = row["horse_name"]
                race_dt = row["_race_dt"]
                cutoff = race_dt - pd.Timedelta(days=7)
                sub = tr_df[(tr_df["hrnm"] == name) &
                            (tr_df["trng_dt"] >= cutoff) &
                            (tr_df["trng_dt"] < race_dt)]
                grade = int(sub["_grade"].min()) if len(sub) else 1
                grade_rows.append({"horse_name": name, "_race_dt": race_dt, "training_grade_enc": float(grade)})
            grade_df = pd.DataFrame(grade_rows)
            df = df.merge(grade_df, on=["horse_name", "_race_dt"], how="left")
            df["training_grade_enc"] = df["training_grade_enc"].fillna(1.0)
            df = df.drop(columns=["_race_dt"])
            logger.info(f"조교 등급 조인 완료: {(df['training_grade_enc'] != 1).sum()}건 실제 기록 반영")
        else:
            df["training_grade_enc"] = 1.0
    else:
        df["training_grade_enc"] = 1.0

    # prev_win_odds: 해당 말의 직전 경주 배당 (당일 배당 누수 방지 - shift로 lag)
    if "win_odds" in df.columns and "horse_id" in df.columns:
        import pandas as _pd
        odds_df = df[["horse_id", "race_date", "win_odds"]].copy()
        odds_df["win_odds"] = pd.to_numeric(odds_df["win_odds"], errors="coerce")
        odds_df = odds_df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
        odds_df["prev_win_odds"] = odds_df.groupby("horse_id")["win_odds"].shift(1)
        df = df.merge(odds_df[["horse_id", "race_date", "prev_win_odds"]], on=["horse_id", "race_date"], how="left")
        median_odds = odds_df["win_odds"].median()
        df["prev_win_odds"] = df["prev_win_odds"].fillna(median_odds)
        logger.info(f"prev_win_odds 조인 완료 (중앙값 폴백={median_odds:.1f})")
    else:
        df["prev_win_odds"] = 15.0

    # prev_speed_figure, speed_figure_trend: 직전 경주 완주속도 (m/s), 누수 방지 - shift로 lag
    if "finish_time" in df.columns and "race_distance" in df.columns:
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
            on=["horse_id", "race_date"], how="left"
        )
        df["prev_speed_figure"] = df["prev_speed_figure"].fillna(0.0)
        df["speed_figure_trend"] = df["speed_figure_trend"].fillna(0.0)
        n_valid = (df["prev_speed_figure"] > 0).sum()
        logger.info(f"prev_speed_figure 계산 완료 (유효 {n_valid}건)")
    else:
        df["prev_speed_figure"] = 0.0
        df["speed_figure_trend"] = 0.0
        logger.warning("finish_time 없음 - prev_speed_figure=0으로 설정 (데이터 수집 후 재학습 필요)")

    before = len(df)
    df = df.dropna(subset=["horse_age", "race_distance"])
    logger.info(f"학습 데이터: {len(df)}건 (피처 없는 {before - len(df)}건 제외)")

    preprocessor = RacePreprocessor()
    df_processed = preprocessor.fit_transform(df)

    trainer = RaceModelTrainer()
    metrics = trainer.train(df_processed)

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics=metrics["metrics"])
    logger.info(f"모델 저장 완료: {version}")


def cmd_predict(args):
    logger.info(f"=== 예측 시작: {args.date} ===")

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()

    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음. 먼저 --mode train 실행")
        return

    from core.race_filter import RaceFilter
    from core.investor import Investor
    from Utils.database import get_client

    predictor = RacePredictor(trainer, preprocessor)
    predictions = predictor.predict_from_db(args.date, meet_code=args.meet)

    if predictions.empty:
        logger.warning(f"{args.date} 예측 결과 없음")
        return

    # 경주 필터 적용 (거리 1200/1400m, 일반 경주, 5두 이상)
    race_filter = RaceFilter()
    client = get_client()
    races_res = client.table("races")\
        .select("race_id, race_distance, race_grade, total_horses, race_no, meet_code")\
        .eq("race_date", args.date)\
        .execute()
    race_info = {r["race_id"]: r for r in (races_res.data or [])}
    eligible_ids = {rid for rid, info in race_info.items() if race_filter.is_eligible(info)}

    filtered = predictions[predictions["race_id"].isin(eligible_ids)].copy()
    total_races = predictions["race_id"].nunique()
    logger.info(f"경주 필터: {len(eligible_ids)}/{total_races}경주 통과")

    MEET_NAME = {1: "서울", 2: "제주", 3: "부산", 0: "미상"}

    def race_label(race_id):
        info = race_info.get(race_id, {})
        meet = MEET_NAME.get(info.get("meet_code"), "미상")
        race_no = info.get("race_no", "?")
        return f"[{meet} {race_no}경주]"

    print(f"\n=== 예측 결과 ({args.date}) ===")
    print(f"필터 통과: {len(eligible_ids)}/{total_races}경주 (1200/1400m·일반·5두 이상)\n")

    all_groups = predictions.groupby("race_id", sort=False)
    for race_id, group in sorted(all_groups, key=lambda x: (
        race_info.get(x[0], {}).get("meet_code", 0),
        race_info.get(x[0], {}).get("race_no", 0),
    )):
        top = group.sort_values("win_probability", ascending=False).iloc[0]
        label = race_label(race_id)
        name = top.get("horse_name") or top.get("horse_id", "?")
        prob = top["win_probability"]
        eligible_mark = "" if race_id in eligible_ids else " [필터 제외]"
        print(f"  {label} 1위 예측: {name} (확률: {prob:.3f}){eligible_mark}")

    # EV 기반 베팅 결정 (필터 통과 경주만)
    meet_code = args.meet if args.meet is not None else 1
    investor = Investor()
    decisions = investor.decide(filtered, args.date, meet_code)

    print()
    if decisions:
        print(f"베팅 추천: {len(decisions)}경주")
        for d in decisions:
            label = race_label(d["race_id"])
            print(
                f"  {label} → {d['horse_name']} | {d['bet_type']} {d['amount']:,}원 "
                f"(EV={d['ev']:+.2f}, 확률={d['model_prob']:.3f}, 배당={d['odds']:.1f}배, 출처={d['source']})"
            )
    else:
        print("베팅 추천: 없음 (EV 기준 미달)")


def cmd_status(args):
    print("=== 시스템 상태 ===")
    connected = test_connection()
    print(f"DB 연결: {'OK' if connected else 'FAIL'}")
    mm = ModelManager()
    models = list(MODEL_DIR.glob("model_*.pkl"))
    print(f"저장된 모델: {len(models)}개")
    if models:
        print(f"최신 모델: {sorted(models)[-1].name}")
    print(f"재학습 필요: {'YES' if mm.is_outdated() else 'NO'}")


def cmd_update_results(args):
    from library.scraper_score import update_all_meets
    from core.history_rebuilder import rebuild

    date = args.date
    logger.info(f"=== 경주 결과 업데이트 시작: {date} ===")

    counts = update_all_meets(date)
    total = sum(counts.values())
    print(f"\n[1/2] win_odds 업데이트: {total}경주")
    for name, cnt in counts.items():
        print(f"  {name}: {cnt}경주")

    if total == 0:
        print(f"  {date} 경주 데이터 없음 - 경마일이 아닐 수 있음")
        return

    print(f"\n[2/2] horse_race_history 재구성 ({date} ~ {date})")
    try:
        rebuild(start_date=date, end_date=date)
        print("  완료")
    except Exception as e:
        logger.error(f"  horse_race_history 재구성 실패: {e}")

    logger.info(f"=== 경주 결과 업데이트 완료: {date} ===")


def main():
    parser = argparse.ArgumentParser(description="경마 자동 투자 봇")
    parser.add_argument("--mode", choices=["train", "predict", "status", "update-results"],
                        default="status")
    parser.add_argument("--start-date", default="2024-01-01",
                        dest="start_date")
    parser.add_argument("--end-date",
                        default=datetime.now().strftime("%Y-%m-%d"),
                        dest="end_date")
    parser.add_argument("--date",
                        default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--meet", type=int, default=None,
                        help="1=서울, 2=부산, 3=제주")
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "predict":
        cmd_predict(args)
    elif args.mode == "status":
        cmd_status(args)
    elif args.mode == "update-results":
        cmd_update_results(args)


if __name__ == "__main__":
    main()
