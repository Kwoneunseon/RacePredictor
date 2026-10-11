import argparse
from datetime import datetime

from config import TOP3_PROB_THRESHOLD
from Utils.database import get_client
from Utils.logger import get_logger
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager
from core.ev_calculator import BET_TYPE_LABELS_KO

logger = get_logger(__name__)


def _resolve_training_range(args) -> tuple[str, str]:
    """train/tune용 --start-date/--end-date 미지정 시 기본값(2024-01-01 ~ 오늘) 적용."""
    start = args.start_date or "2024-01-01"
    end = args.end_date or datetime.now().strftime("%Y-%m-%d")
    return start, end


def _resolve_dates(args) -> list[str]:
    """update-results용. --start-date/--end-date가 지정되면 그 기간의 날짜 목록을,
    아니면 --date 단일 날짜를 반환."""
    if args.start_date or args.end_date:
        import pandas as pd
        start = args.start_date or args.date
        end = args.end_date or args.date
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]
    return [args.date]


def _fetch_race_entries(args) -> "pd.DataFrame":
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder

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
        return pd.DataFrame()

    logger.info(f"race_entries 수집: {len(all_entries)}건")
    fb = RaceFeatureBuilder()
    return fb.get_flatten_entries(all_entries)


def cmd_tune(args):
    logger.info("=== 하이퍼파라미터 튜닝 시작 (XGBoost·LightGBM) ===")
    from config import OPTUNA_N_TRIALS
    from core.feature_builder import RaceFeatureBuilder

    args.start_date, args.end_date = _resolve_training_range(args)
    df = _fetch_race_entries(args)
    if df.empty:
        logger.error("학습 데이터 없음")
        return

    fb = RaceFeatureBuilder()
    df = fb.build_for_training(df, args.end_date)
    df = df.dropna(subset=["horse_age", "race_distance"])

    preprocessor = RacePreprocessor()
    df_processed = preprocessor.fit_transform(df)

    trainer = RaceModelTrainer()
    trainer.tune(df_processed, n_trials=OPTUNA_N_TRIALS)
    logger.info("튜닝 완료. 이제 '모델 재학습'을 실행하세요 (저장된 파라미터 자동 적용).")


def cmd_train(args):
    logger.info("=== 모델 학습 시작 ===")
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder

    args.start_date, args.end_date = _resolve_training_range(args)
    df = _fetch_race_entries(args)
    if df.empty:
        logger.error("학습 데이터 없음")
        return

    fb = RaceFeatureBuilder()
    df = fb.build_for_training(df, args.end_date)

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

    if version:
        from core.feature_importance_report import generate_report, _load_meta
        report_path = generate_report(trainer, _load_meta(version))
        if report_path:
            logger.info(f"변수 중요도 리포트 갱신: {report_path}")


def _ensure_race_data(date: str) -> bool:
    """당일 race_entries가 없으면 자동 수집. 이미 있으면 스킵. True=수집완료/이미있음"""
    from Utils.database import get_client
    client = get_client()
    res = client.table("race_entries").select("id", count="exact")\
        .eq("race_date", date).limit(1).execute()
    count = res.count if res.count is not None else len(res.data or [])
    if count > 0:
        logger.info(f"{date} 경주 데이터 이미 존재 ({count}건), 수집 생략")
        return True

    logger.info(f"{date} 경주 데이터 없음 → 자동 수집 시작")
    try:
        from library.collector_races import collecting_race_results
        date_nodash = date.replace("-", "")
        collecting_race_results(start_date=date_nodash, end_date=date_nodash)
        logger.info(f"{date} 데이터 수집 완료")
        return True
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")
        return False


def _apply_user_criteria(
    win_id: str, t3_id: str,
    win_odds: float, t3_odds: float,
    is_eligible: bool,
):
    """사용자 베팅 기준 적용. (bet_horse_id, reason) 또는 (None, None) 반환"""
    same = (win_id == t3_id)
    if not is_eligible:
        if same and 1.0 <= win_odds < 2.0:
            return win_id, f"필터제외·동일·{win_odds:.2f}배"
        return None, None
    if same:
        if win_odds < 3.0:
            return win_id, f"동일·{win_odds:.2f}배"
        return None, None
    w1 = 1.0 <= win_odds < 2.0
    t1 = 1.0 <= t3_odds < 2.0
    if w1 and not t1:
        return win_id, f"단승1점대·{win_odds:.2f}배"
    if t1 and not w1:
        return t3_id, f"연승1점대·{t3_odds:.2f}배"
    if w1 and t1:
        if win_odds <= t3_odds:
            return win_id, f"둘다1점대→단승·{win_odds:.2f}배"
        return t3_id, f"둘다1점대→연승·{t3_odds:.2f}배"
    return None, None


def _format_actual_result(rows: list[dict]) -> str:
    """[{"entry_number":1,"final_rank":3}, ...] → 완주 순위 순 "3-1-2" 형태 문자열. 결과 없으면 빈 문자열."""
    ordered = sorted(
        (r for r in rows if r.get("entry_number") is not None),
        key=lambda r: r["final_rank"],
    )
    return "-".join(str(int(r["entry_number"])) for r in ordered)


def _fetch_actual_results(date: str, client) -> dict:
    """해당 날짜에 이미 완주 기록(final_rank)이 있는 경주만 race_id → "1-3-4-11" 형태로 매핑."""
    res = client.table("race_entries").select("race_id, entry_number, final_rank")\
        .eq("race_date", date).gt("final_rank", 0).execute()
    if not res.data:
        return {}
    by_race: dict = {}
    for row in res.data:
        by_race.setdefault(row["race_id"], []).append(row)
    return {race_id: _format_actual_result(rows) for race_id, rows in by_race.items()}


def cmd_predict(args):
    _ensure_race_data(args.date)
    logger.info(f"=== 예측 시작: {args.date} ===")

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()

    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음. 먼저 --mode train 실행")
        return

    from core.race_filter import RaceFilter
    from core.investor import Investor
    from core.select_odds import OddsCollector
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
    actual_results = _fetch_actual_results(args.date, client)

    total_races = predictions["race_id"].nunique()
    logger.info(f"경주 필터: {len(eligible_ids)}/{total_races}경주 통과")

    # 배당 수집 (경마장별 전체)
    oc = OddsCollector()
    all_odds: dict = {}
    meet_codes = {r.get("meet_code") for r in (races_res.data or []) if r.get("meet_code")}
    for mc in sorted(meet_codes):
        all_odds.update(oc.get_odds(args.date, mc))

    MEET_NAME = {1: "서울", 2: "제주", 3: "부산경남", 0: "미상"}

    def race_label(race_id):
        info = race_info.get(race_id, {})
        meet = MEET_NAME.get(info.get("meet_code"), "미상")
        race_no = info.get("race_no", "?")
        return f"[{meet} {race_no}경주]"

    def horse_no_str(no):
        try:
            return str(int(no))
        except (TypeError, ValueError):
            return "?"

    print(f"\n=== 예측 결과 ({args.date}) ===")
    print(f"필터 통과: {len(eligible_ids)}/{total_races}경주 (1200/1400m·일반·5두 이상)")
    print(f"연승 기준: top3_prob >= {TOP3_PROB_THRESHOLD:.2f} + 사용자 배당 조건 (배당<2.0)\n")

    top3_bet_races = []   # 실시간 배당 있음 + 조건 통과
    top3_candidates = []  # 실시간 배당 없음 → 사용자 직접 확인
    race_rows = []         # 전체 경주 예측 (리포트용)
    all_groups = predictions.groupby("race_id", sort=False)
    for race_id, group in sorted(all_groups, key=lambda x: (
        race_info.get(x[0], {}).get("meet_code", 0),
        race_info.get(x[0], {}).get("race_no", 0),
    )):
        top_win = group.sort_values("win_probability", ascending=False).iloc[0]
        top_t3  = group.sort_values("top3_probability", ascending=False).iloc[0]

        win_id   = top_win.get("horse_id")
        t3_id    = top_t3.get("horse_id")
        win_name = top_win.get("horse_name") or win_id or "?"
        t3_name  = top_t3.get("horse_name") or t3_id or "?"
        win_no   = horse_no_str(top_win.get("entry_number"))
        t3_no    = horse_no_str(top_t3.get("entry_number"))
        win_prob = float(top_win["win_probability"])
        t3_prob  = float(top_t3["top3_probability"])

        race_odds   = all_odds.get(race_id, {})
        odds_source = race_odds.get("_source", "fallback")
        win_odds_v  = float(race_odds.get(win_id, 0)) if win_id else 0
        t3_odds_v   = float(race_odds.get(t3_id, 0)) if t3_id else 0
        is_elig     = race_id in eligible_ids

        elig_mark = "" if is_elig else " [필터제외]"

        if t3_prob >= TOP3_PROB_THRESHOLD:
            if odds_source == "scraping" and win_odds_v > 0 and t3_odds_v > 0:
                bet_id, reason = _apply_user_criteria(
                    win_id, t3_id, win_odds_v, t3_odds_v, is_elig
                )
                if bet_id:
                    bet_name = win_name if bet_id == win_id else t3_name
                    bet_no   = win_no if bet_id == win_id else t3_no
                    bet_odds = win_odds_v if bet_id == win_id else t3_odds_v
                    top3_bet_races.append({
                        "race_id": race_id,
                        "label": race_label(race_id),
                        "horse_id": bet_id,
                        "horse_no": bet_no,
                        "horse": bet_name,
                        "top3_prob": t3_prob,
                        "odds": bet_odds,
                        "reason": reason,
                        "eligible": is_elig,
                    })
            else:
                same = (win_id == t3_id)
                top3_candidates.append({
                    "race_id":  race_id,
                    "label":    race_label(race_id),
                    "win_name": win_name,
                    "win_no":   win_no,
                    "t3_name":  t3_name,
                    "t3_no":    t3_no,
                    "t3_prob":  t3_prob,
                    "same":     same,
                    "eligible": is_elig,
                })

        race_rows.append({
            "label": race_label(race_id),
            "win_no": win_no, "win_name": win_name, "win_prob": win_prob,
            "t3_no": t3_no, "t3_name": t3_name, "t3_prob": t3_prob,
            "eligible": is_elig,
            "actual_result": actual_results.get(race_id, ""),
        })

        print(f"  {race_label(race_id)} "
              f"단승:{win_no}번 {win_name}({win_prob:.3f})  연승:{t3_no}번 {t3_name}({t3_prob:.3f})"
              f"{elig_mark}")

    print()
    if top3_bet_races:
        print(f"★ 연승 베팅 추천 ({len(top3_bet_races)}경주):")
        for r in top3_bet_races:
            elig_mark = "" if r["eligible"] else " [필터제외]"
            print(f"  {race_label(r['race_id'])} → {r['horse_no']}번 {r['horse']} "
                  f"(top3_prob={r['top3_prob']:.3f}, 배당={r['odds']:.2f}배) [{r['reason']}]{elig_mark}")
    else:
        print(f"연승 추천: 없음 (top3_prob >= {TOP3_PROB_THRESHOLD:.2f} + 배당 조건 미달)")

    if top3_candidates:
        print(f"\n★ 연승후보 — 배당 직접 확인 ({len(top3_candidates)}경주):")
        for c in top3_candidates:
            elig_mark = "" if c["eligible"] else " [필터제외]"
            if c["same"]:
                horse_info = f"{c['t3_no']}번 {c['t3_name']}"
            else:
                horse_info = f"단승:{c['win_no']}번 {c['win_name']} / 연승:{c['t3_no']}번 {c['t3_name']}"
            print(f"  {race_label(c['race_id'])} → {horse_info} (prob={c['t3_prob']:.3f}){elig_mark}")

    # EV 기반 베팅 결정 (필터 통과 경주만)
    filtered = predictions[predictions["race_id"].isin(eligible_ids)].copy()
    meet_code = args.meet if args.meet is not None else 1
    investor = Investor()
    decisions = investor.decide(filtered, args.date, meet_code)

    print()
    report_decisions = []
    if decisions:
        print(f"베팅 추천: {len(decisions)}경주")
        for d in decisions:
            label = race_label(d["race_id"])
            bet_type_label = BET_TYPE_LABELS_KO.get(d["bet_type"], d["bet_type"])
            print(
                f"  {label} → {horse_no_str(d.get('horse_no'))}번 {d['horse_name']} | {bet_type_label} {d['amount']:,}원 "
                f"(EV={d['ev']:+.2f}, 확률={d['model_prob']:.3f}, 배당={d['odds']:.1f}배, 출처={d['source']})"
            )
            report_decisions.append({
                "label": label, "horse_no": horse_no_str(d.get("horse_no")),
                "horse_name": d["horse_name"], "bet_type": d["bet_type"], "amount": d["amount"],
                "ev": d["ev"], "model_prob": d["model_prob"], "odds": d["odds"], "source": d["source"],
            })
    else:
        print("베팅 추천: 없음 (EV 기준 미달)")

    from core.prediction_report import save_report
    report_path = save_report({
        "date": args.date,
        "threshold": TOP3_PROB_THRESHOLD,
        "total_races": total_races,
        "eligible_races": len(eligible_ids),
        "race_rows": race_rows,
        "bet_races": top3_bet_races,
        "candidates": top3_candidates,
        "decisions": report_decisions,
    })
    print(f"\nHTML 리포트 저장: {report_path}")


def cmd_update_results(args):
    from library.scraper_score import update_all_meets, update_all_final_ranks
    from core.history_rebuilder import rebuild

    dates = _resolve_dates(args)
    logger.info(f"=== 경주 결과 업데이트 시작: {dates[0]} ~ {dates[-1]} ({len(dates)}일) ===")

    total_odds_all = 0
    total_rank_all = 0
    for date in dates:
        # 1. ScoreInfo 스크래핑으로 win_odds 업데이트 (서울·제주·부산 전체)
        counts = update_all_meets(date)
        total_odds = sum(counts.values())
        print(f"\n[{date}] win_odds 업데이트: {total_odds}경주")
        for name, cnt in counts.items():
            print(f"  {name}: {cnt}경주")

        # 2. 공공데이터 API로 final_rank 업데이트 (서울·제주·부산경남 전체)
        rank_counts = update_all_final_ranks(date)
        total_rank = sum(rank_counts.values())
        print(f"[{date}] final_rank 업데이트: {total_rank}건")
        for name, cnt in rank_counts.items():
            print(f"  {name}: {cnt}건")

        if total_odds == 0 and total_rank == 0:
            print(f"  {date} 경주 데이터 없음 - 경마일이 아닐 수 있음")

        total_odds_all += total_odds
        total_rank_all += total_rank

    # 3. horse_race_history 재구성 (전체 기간 한 번에)
    print(f"\nhorse_race_history 재구성 ({dates[0]} ~ {dates[-1]})")
    try:
        rebuild(start_date=dates[0], end_date=dates[-1])
        print("  완료")
    except Exception as e:
        logger.error(f"  horse_race_history 재구성 실패: {e}")

    logger.info(
        f"=== 경주 결과 업데이트 완료: {len(dates)}일 "
        f"(win_odds {total_odds_all}건, final_rank {total_rank_all}건) ==="
    )


def main():
    parser = argparse.ArgumentParser(description="경마 자동 투자 봇")
    parser.add_argument("--mode", choices=["train", "predict", "update-results", "tune"],
                        default="predict")
    parser.add_argument("--start-date", default=None,
                        dest="start_date",
                        help="train/tune: 학습 데이터 시작일(기본 2024-01-01). "
                             "update-results: 함께 지정 시 --date 대신 이 기간 전체를 처리")
    parser.add_argument("--end-date", default=None,
                        dest="end_date",
                        help="train/tune: 학습 데이터 종료일(기본 오늘). "
                             "update-results: 기간의 종료일")
    parser.add_argument("--date",
                        default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--meet", type=int, default=None,
                        help="1=서울, 2=제주, 3=부산경남")
    args = parser.parse_args()

    if args.mode == "tune":
        cmd_tune(args)
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "predict":
        cmd_predict(args)
    elif args.mode == "update-results":
        cmd_update_results(args)


if __name__ == "__main__":
    main()
