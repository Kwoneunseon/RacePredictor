"""
연승 적중 실패 원인 심층 분석
- 배당 구간별 적중률
- 필터 통과/제외 구분
- 경마장별
- 출전 두수별
- 단승=연승 일치 여부
- 모델 top3_probability 분포
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from Utils.database import get_client
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.model_manager import ModelManager
from core.predictor import RacePredictor
from core.race_filter import RaceFilter
from core.feature_builder import RaceFeatureBuilder

START_DATE = "2026-05-01"
END_DATE   = "2026-06-25"
MEET_NAME  = {1: "서울", 2: "제주", 3: "부산"}


def apply_user_criteria(win_id, top3_id, win_odds, top3_odds, is_eligible):
    same = (win_id == top3_id)
    if not is_eligible:
        if same and 1.0 <= win_odds < 2.0:
            return win_id, win_odds, "필터제외·동일"
        return None, None, None
    else:
        if same:
            if win_odds < 3.0:
                return win_id, win_odds, "필터통과·동일"
            return None, None, None
        else:
            w1 = 1.0 <= win_odds < 2.0
            t1 = 1.0 <= top3_odds < 2.0
            if w1 and not t1:
                return win_id, win_odds, "필터통과·불일치·단승1점대"
            elif t1 and not w1:
                return top3_id, top3_odds, "필터통과·불일치·연승1점대"
            elif w1 and t1:
                if win_odds <= top3_odds:
                    return win_id, win_odds, "필터통과·불일치·둘다1점대"
                else:
                    return top3_id, top3_odds, "필터통과·불일치·둘다1점대"
            return None, None, None


def main():
    client = get_client()

    races_res = client.table("races").select(
        "race_id, race_date, race_no, meet_code, race_distance, race_grade, total_horses"
    ).gte("race_date", START_DATE).lte("race_date", END_DATE).execute()

    if not races_res.data:
        print("데이터 없음"); return

    race_df = pd.DataFrame(races_res.data)
    race_dates = sorted(race_df["race_date"].unique())

    preprocessor = RacePreprocessor()
    trainer      = RaceModelTrainer()
    mm           = ModelManager()
    if not mm.load(trainer, preprocessor):
        print("모델 없음"); return
    predictor = RacePredictor(trainer, preprocessor)

    race_filter = RaceFilter()
    race_info   = {r["race_id"]: r for r in races_res.data}
    eligible_ids = {rid for rid, info in race_info.items() if race_filter.is_eligible(info)}

    rows = []
    for date in race_dates:
        fb  = RaceFeatureBuilder()
        df  = fb.build_for_prediction(date)
        if df.empty: continue
        df  = preprocessor.transform(df)
        preds = predictor.predict(df)
        if preds.empty: continue

        entries_res = client.table("race_entries").select(
            "race_id, horse_id, final_rank, win_odds"
        ).eq("race_date", date).gt("final_rank", 0).execute()
        if not entries_res.data: continue

        result_map = {(e["race_id"], e["horse_id"]): e for e in entries_res.data}

        for race_id, group in preds.groupby("race_id"):
            info       = race_info.get(race_id, {})
            is_elig    = race_id in eligible_ids
            total_h    = info.get("total_horses") or len(group)
            meet       = MEET_NAME.get(info.get("meet_code"), "?")

            top_win  = group.sort_values("win_probability",  ascending=False).iloc[0]
            top_top3 = group.sort_values("top3_probability", ascending=False).iloc[0]

            win_id   = top_win.get("horse_id")
            top3_id  = top_top3.get("horse_id")
            win_prob  = float(top_win["win_probability"])
            top3_prob = float(top_top3["top3_probability"])

            w_e = result_map.get((race_id, win_id), {})
            t_e = result_map.get((race_id, top3_id), {})
            win_odds_v  = float(w_e.get("win_odds") or 0)
            top3_odds_v = float(t_e.get("win_odds") or 0)
            if win_odds_v <= 0 or top3_odds_v <= 0: continue

            bet_id, bet_odds, reason = apply_user_criteria(
                win_id, top3_id, win_odds_v, top3_odds_v, is_elig
            )
            if bet_id is None: continue

            bet_e       = result_map.get((race_id, bet_id), {})
            actual_rank = int(bet_e.get("final_rank", 99))
            top3_hit    = (actual_rank <= 3)

            # 배당 기준으로 확인하는 top3_prob는 bet_horse의 것
            bet_t3_prob = top3_prob if bet_id == top3_id else win_prob

            rows.append({
                "date":       date,
                "meet":       meet,
                "race_no":    info.get("race_no"),
                "eligible":   is_elig,
                "total_h":    total_h,
                "bet_odds":   bet_odds,
                "reason":     reason,
                "top3_prob":  bet_t3_prob,
                "actual_rank": actual_rank,
                "top3_hit":   top3_hit,
            })

    df_r = pd.DataFrame(rows)
    total = len(df_r)
    hits  = df_r["top3_hit"].sum()
    print(f"\n{'='*60}")
    print(f"연승 적중 심층 분석 ({START_DATE} ~ {END_DATE})")
    print(f"{'='*60}")
    print(f"전체: {hits}/{total}건 ({hits/total*100:.1f}%)\n")

    # 1. 배당 구간별
    print("[ 배당 구간별 연승 적중률 ]")
    bins   = [0, 1.5, 2.0, 2.5, 3.0, 99]
    labels = ["~1.5", "1.5~2.0", "2.0~2.5", "2.5~3.0", "3.0~"]
    df_r["odds_band"] = pd.cut(df_r["bet_odds"], bins=bins, labels=labels)
    for band, grp in df_r.groupby("odds_band", observed=True):
        n = len(grp); h = grp["top3_hit"].sum()
        print(f"  {band:>10}: {h:>3}/{n:>3}건 ({h/n*100:5.1f}%)")

    # 2. 단승=연승 일치 여부
    print("\n[ 단승=연승 일치 여부별 ]")
    df_r["same"] = df_r["reason"].str.contains("동일")
    for same, grp in df_r.groupby("same"):
        label = "단승=연승 동일" if same else "단승≠연승 불일치"
        n = len(grp); h = grp["top3_hit"].sum()
        print(f"  {label}: {h}/{n}건 ({h/n*100:.1f}%)")

    # 3. 필터 통과/제외별
    print("\n[ 필터 통과/제외별 ]")
    for elig, grp in df_r.groupby("eligible"):
        label = "필터통과" if elig else "필터제외"
        n = len(grp); h = grp["top3_hit"].sum()
        print(f"  {label}: {h}/{n}건 ({h/n*100:.1f}%)")

    # 4. 경마장별
    print("\n[ 경마장별 ]")
    for meet, grp in df_r.groupby("meet"):
        n = len(grp); h = grp["top3_hit"].sum()
        print(f"  {meet}: {h}/{n}건 ({h/n*100:.1f}%)")

    # 5. 출전 두수별
    print("\n[ 출전 두수별 ]")
    h_bins   = [0, 7, 9, 11, 99]
    h_labels = ["~7두", "8~9두", "10~11두", "12두+"]
    df_r["h_band"] = pd.cut(df_r["total_h"], bins=h_bins, labels=h_labels)
    for band, grp in df_r.groupby("h_band", observed=True):
        n = len(grp); h = grp["top3_hit"].sum()
        print(f"  {band}: {h:>3}/{n:>3}건 ({h/n*100:5.1f}%)")

    # 6. 모델 top3_probability 구간별
    print("\n[ 모델 top3_prob 구간별 ]")
    p_bins   = [0, 0.25, 0.30, 0.35, 0.40, 1.0]
    p_labels = ["~0.25", "0.25~0.30", "0.30~0.35", "0.35~0.40", "0.40+"]
    df_r["p_band"] = pd.cut(df_r["top3_prob"], bins=p_bins, labels=p_labels)
    for band, grp in df_r.groupby("p_band", observed=True):
        n = len(grp); h = grp["top3_hit"].sum()
        if n == 0: continue
        print(f"  {band}: {h:>3}/{n:>3}건 ({h/n*100:5.1f}%)")

    # 7. 조합 시뮬레이션: "배당 X 이하 + 동일 조건만" 적용 시 결과
    print("\n[ 기준 강화 시뮬레이션 ]")
    scenarios = [
        ("현행 기준",                    df_r),
        ("배당 2.0 미만",                df_r[df_r["bet_odds"] < 2.0]),
        ("배당 1.8 미만",                df_r[df_r["bet_odds"] < 1.8]),
        ("top3_prob 0.25+",              df_r[df_r["top3_prob"] >= 0.25]),
        ("top3_prob 0.27+",              df_r[df_r["top3_prob"] >= 0.27]),
        ("top3_prob 0.30+",              df_r[df_r["top3_prob"] >= 0.30]),
        ("배당<2.0 + prob≥0.23",         df_r[(df_r["bet_odds"] < 2.0) & (df_r["top3_prob"] >= 0.23)]),
        ("배당<2.0 + prob≥0.25",         df_r[(df_r["bet_odds"] < 2.0) & (df_r["top3_prob"] >= 0.25)]),
        ("배당<2.0 + prob≥0.27",         df_r[(df_r["bet_odds"] < 2.0) & (df_r["top3_prob"] >= 0.27)]),
        ("배당<2.0 + prob≥0.28",         df_r[(df_r["bet_odds"] < 2.0) & (df_r["top3_prob"] >= 0.28)]),
        ("배당<2.0 + prob≥0.30",         df_r[(df_r["bet_odds"] < 2.0) & (df_r["top3_prob"] >= 0.30)]),
        ("배당<1.8 + prob≥0.25",         df_r[(df_r["bet_odds"] < 1.8) & (df_r["top3_prob"] >= 0.25)]),
    ]
    for label, sub in scenarios:
        if sub.empty:
            print(f"  {label:<28}: 해당 없음")
            continue
        n = len(sub); h = sub["top3_hit"].sum()
        print(f"  {label:<28}: {h:>3}/{n:>3}건 ({h/n*100:5.1f}%)")

    # 8. 실패 케이스 - 배당 낮은데 크게 탈락 (upset)
    print("\n[ 이변 케이스 (배당<1.8인데 5등 이하) ]")
    upset = df_r[(df_r["bet_odds"] < 1.8) & (df_r["actual_rank"] >= 5)]
    for _, row in upset.iterrows():
        print(f"  {row['date']} {row['meet']} {row['race_no']}R | "
              f"배당{row['bet_odds']:.2f} | {int(row['actual_rank'])}등 | "
              f"두수{int(row['total_h'])} | prob{row['top3_prob']:.3f} | {row['reason']}")


if __name__ == "__main__":
    main()
