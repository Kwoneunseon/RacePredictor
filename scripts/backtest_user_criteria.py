"""
사용자 기준 백테스트
기간: 최근 경마일 데이터로 예측 vs 실제 결과 대조
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from Utils.database import get_client
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.model_manager import ModelManager
from core.predictor import RacePredictor
from core.race_filter import RaceFilter
from core.feature_builder import RaceFeatureBuilder
from Utils.logger import get_logger

logger = get_logger(__name__)

START_DATE = "2026-05-01"
END_DATE   = "2026-06-25"  # 완료된 경주만 (오늘 제외)

MEET_NAME = {1: "서울", 2: "제주", 3: "부산"}


def is_1x(odds: float) -> bool:
    """1점대 배당 (1.0 이상 2.0 미만)"""
    return 1.0 <= odds < 2.0


def is_2x_or_more(odds: float) -> bool:
    """2점대 이상"""
    return odds >= 2.0


def apply_user_criteria(win_horse, top3_horse, win_odds, top3_odds, is_eligible):
    """
    사용자 베팅 기준 적용.
    반환: (bet_horse, reason) or (None, reason)
    """
    same = (win_horse == top3_horse)

    if not is_eligible:
        # 필터 제외: 두 예측이 같고 배당이 1점대일 때만 베팅
        if same and is_1x(win_odds):
            return win_horse, f"필터제외·동일·1점대({win_odds:.2f})"
        return None, f"필터제외·{'동일' if same else '불일치'}·배당{win_odds:.2f}"
    else:
        if same:
            if win_odds < 3.0:
                return win_horse, f"동일·배당{win_odds:.2f}"
            else:
                return None, f"동일·배당{win_odds:.2f}≥3"
        else:
            # 단승≠연승
            w1, w3 = is_1x(win_odds), is_1x(top3_odds)
            if w1 and not w3:
                return win_horse, f"단승1점대({win_odds:.2f}),연승{top3_odds:.2f}"
            elif w3 and not w1:
                return top3_horse, f"연승1점대({top3_odds:.2f}),단승{win_odds:.2f}"
            elif w1 and w3:
                # 둘다 1점대 → 낮은 배당
                if win_odds <= top3_odds:
                    return win_horse, f"둘다1점대→단승낮음({win_odds:.2f})"
                else:
                    return top3_horse, f"둘다1점대→연승낮음({top3_odds:.2f})"
            else:
                return None, f"둘다2점대이상(단{win_odds:.2f}/연{top3_odds:.2f})"


def main():
    client = get_client()

    # 1. 완료된 경주 날짜 목록 조회
    races_res = client.table("races").select(
        "race_id, race_date, race_no, meet_code, race_distance, race_grade, total_horses"
    ).gte("race_date", START_DATE).lte("race_date", END_DATE).execute()

    if not races_res.data:
        print("데이터 없음")
        return

    race_df = pd.DataFrame(races_res.data)
    race_dates = sorted(race_df["race_date"].unique())
    print(f"대상 경마일: {len(race_dates)}일 ({race_dates[0]} ~ {race_dates[-1]})")

    # 2. 모델 로드
    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor):
        print("모델 없음")
        return
    predictor = RacePredictor(trainer, preprocessor)

    race_filter = RaceFilter()
    race_info = {r["race_id"]: r for r in races_res.data}
    eligible_ids = {rid for rid, info in race_info.items() if race_filter.is_eligible(info)}

    # 3. 날짜별 예측 + 실제 결과 대조
    results = []

    for date in race_dates:
        fb = RaceFeatureBuilder()
        df = fb.build_for_prediction(date)
        if df.empty:
            continue
        df = preprocessor.transform(df)
        preds = predictor.predict(df)
        if preds.empty:
            continue

        # 실제 결과 조회 (final_rank)
        entries_res = client.table("race_entries").select(
            "race_id, horse_id, final_rank, win_odds"
        ).eq("race_date", date).gt("final_rank", 0).execute()

        if not entries_res.data:
            continue

        result_map = {}
        for e in entries_res.data:
            result_map[(e["race_id"], e["horse_id"])] = e

        # 경주별 분석
        for race_id, group in preds.groupby("race_id"):
            info = race_info.get(race_id, {})
            is_eligible = race_id in eligible_ids

            # 단승 1위 예측
            top_win = group.sort_values("win_probability", ascending=False).iloc[0]
            # 연승 1위 예측
            top_top3 = group.sort_values("top3_probability", ascending=False).iloc[0]

            win_horse_id   = top_win.get("horse_id")
            top3_horse_id  = top_top3.get("horse_id")
            win_horse_name = top_win.get("horse_name") or win_horse_id
            top3_horse_name= top_top3.get("horse_name") or top3_horse_id

            # win_odds: DB에서 가져옴
            w_entry = result_map.get((race_id, win_horse_id), {})
            t_entry = result_map.get((race_id, top3_horse_id), {})
            win_odds_val  = float(w_entry.get("win_odds") or 0)
            top3_odds_val = float(t_entry.get("win_odds") or 0)

            if win_odds_val <= 0 or top3_odds_val <= 0:
                continue

            bet_horse_id, reason = apply_user_criteria(
                win_horse_id, top3_horse_id,
                win_odds_val, top3_odds_val,
                is_eligible
            )

            if bet_horse_id is None:
                continue

            # 실제 결과
            bet_entry = result_map.get((race_id, bet_horse_id), {})
            actual_rank = bet_entry.get("final_rank", 99)
            bet_horse_name = (
                (win_horse_name if bet_horse_id == win_horse_id else top3_horse_name)
            )
            bet_odds = (win_odds_val if bet_horse_id == win_horse_id else top3_odds_val)

            win1 = (actual_rank == 1)
            win3 = (actual_rank <= 3)

            meet = MEET_NAME.get(info.get("meet_code"), "?")
            results.append({
                "date": date,
                "race": f"{meet} {info.get('race_no', '?')}R",
                "eligible": is_eligible,
                "bet_horse": bet_horse_name,
                "bet_odds": bet_odds,
                "reason": reason,
                "actual_rank": actual_rank,
                "win1": win1,
                "win3": win3,
            })

    if not results:
        print("베팅 대상 없음")
        return

    df_res = pd.DataFrame(results)

    # 4. 출력
    print(f"\n{'='*70}")
    print(f"백테스트 결과: {START_DATE} ~ {END_DATE}")
    print(f"{'='*70}")
    print(f"총 베팅: {len(df_res)}건")
    print(f"단승 적중 (1등): {df_res['win1'].sum()}건 / {len(df_res)}건 "
          f"({df_res['win1'].mean()*100:.1f}%)")
    print(f"연승 적중 (3등 이내): {df_res['win3'].sum()}건 / {len(df_res)}건 "
          f"({df_res['win3'].mean()*100:.1f}%)")

    # 필터 통과/제외 분리
    elig = df_res[df_res["eligible"]]
    inelig = df_res[~df_res["eligible"]]
    if not elig.empty:
        print(f"\n[필터 통과] {len(elig)}건 | 단승 {elig['win1'].mean()*100:.1f}% | 연승 {elig['win3'].mean()*100:.1f}%")
    if not inelig.empty:
        print(f"[필터 제외] {len(inelig)}건 | 단승 {inelig['win1'].mean()*100:.1f}% | 연승 {inelig['win3'].mean()*100:.1f}%")

    print(f"\n{'─'*70}")
    print(f"{'날짜':<12} {'경주':<8} {'필터':<6} {'베팅말':<14} {'배당':>5} {'순위':>4} {'단승':>4} {'연승':>4}  사유")
    print(f"{'─'*70}")

    for _, row in df_res.iterrows():
        flag_e = "통과" if row["eligible"] else "제외"
        flag_w1 = "O" if row["win1"] else "X"
        flag_w3 = "O" if row["win3"] else "X"
        print(f"{row['date']:<12} {row['race']:<8} {flag_e:<6} {row['bet_horse']:<14} "
              f"{row['bet_odds']:>5.2f} {int(row['actual_rank']):>4}등 {flag_w1:>4} {flag_w3:>4}  {row['reason']}")

    # EV 간단 계산 (단승 기준, 회수율 75% 가정)
    hits = df_res[df_res["win1"]]
    if not hits.empty:
        avg_return = hits["bet_odds"].mean() * 0.75
        print(f"\n적중 시 평균 배당: {hits['bet_odds'].mean():.2f}배 → 환급 {avg_return:.2f}배")

    total_bets = len(df_res)
    total_wins = df_res["win1"].sum()
    avg_odds_all = df_res["bet_odds"].mean()
    expected = (total_wins * avg_odds_all * 0.75) / total_bets if total_bets > 0 else 0
    print(f"단순 기대수익률: {expected:.2f} (평균배당{avg_odds_all:.2f} × 적중률{total_wins/total_bets:.2%} × 환급75%)")


if __name__ == "__main__":
    main()
