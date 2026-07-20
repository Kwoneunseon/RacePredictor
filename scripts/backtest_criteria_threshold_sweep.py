"""
사용자 배당 기준(_apply_user_criteria) 위에 top3_prob 임계값을 추가로 걸었을 때
정확도가 실제로 개선되는지 확인하는 스윕 백테스트.

odds-rule 단독 baseline(TOP3_PROB_THRESHOLD 없음) vs 여러 임계값 조합을 비교.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
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
END_DATE = "2026-06-25"

THRESHOLDS = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def is_1x(odds: float) -> bool:
    return 1.0 <= odds < 2.0


def apply_user_criteria(win_horse, top3_horse, win_odds, top3_odds, is_eligible):
    same = (win_horse == top3_horse)
    if not is_eligible:
        if same and is_1x(win_odds):
            return win_horse
        return None
    if same:
        if win_odds < 3.0:
            return win_horse
        return None
    w1, w3 = is_1x(win_odds), is_1x(top3_odds)
    if w1 and not w3:
        return win_horse
    if w3 and not w1:
        return top3_horse
    if w1 and w3:
        return win_horse if win_odds <= top3_odds else top3_horse
    return None


def main():
    client = get_client()
    races_res = client.table("races").select(
        "race_id, race_date, race_no, meet_code, race_distance, race_grade, total_horses"
    ).gte("race_date", START_DATE).lte("race_date", END_DATE).execute()
    if not races_res.data:
        print("데이터 없음")
        return

    race_df = pd.DataFrame(races_res.data)
    race_dates = sorted(race_df["race_date"].unique())
    print(f"대상 경마일: {len(race_dates)}일 ({race_dates[0]} ~ {race_dates[-1]})")

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

    rows = []  # 임계값 게이트 전, 배당 규칙만으로 뽑힌 모든 후보 저장 (t3_prob 포함)

    for date in race_dates:
        fb = RaceFeatureBuilder()
        df = fb.build_for_prediction(date)
        if df.empty:
            continue
        df = preprocessor.transform(df)
        preds = predictor.predict(df)
        if preds.empty:
            continue

        entries_res = client.table("race_entries").select(
            "race_id, horse_id, final_rank, win_odds"
        ).eq("race_date", date).gt("final_rank", 0).execute()
        if not entries_res.data:
            continue
        result_map = {(e["race_id"], e["horse_id"]): e for e in entries_res.data}

        for race_id, group in preds.groupby("race_id"):
            is_eligible = race_id in eligible_ids

            top_win = group.sort_values("win_probability", ascending=False).iloc[0]
            top_t3 = group.sort_values("top3_probability", ascending=False).iloc[0]

            win_horse_id = top_win.get("horse_id")
            top3_horse_id = top_t3.get("horse_id")
            t3_prob = float(top_t3["top3_probability"])

            w_entry = result_map.get((race_id, win_horse_id), {})
            t_entry = result_map.get((race_id, top3_horse_id), {})
            win_odds_val = float(w_entry.get("win_odds") or 0)
            top3_odds_val = float(t_entry.get("win_odds") or 0)
            if win_odds_val <= 0 or top3_odds_val <= 0:
                continue

            bet_horse_id = apply_user_criteria(
                win_horse_id, top3_horse_id, win_odds_val, top3_odds_val, is_eligible
            )
            if bet_horse_id is None:
                continue

            bet_entry = result_map.get((race_id, bet_horse_id), {})
            actual_rank = bet_entry.get("final_rank", 99)
            rows.append({
                "t3_prob": t3_prob,
                "win3": actual_rank <= 3,
            })

    if not rows:
        print("베팅 대상 없음")
        return

    df_res = pd.DataFrame(rows)
    total = len(df_res)
    print(f"\n{'='*60}")
    print(f"배당 규칙 + top3_prob 임계값 스윕: {START_DATE} ~ {END_DATE}")
    print(f"{'='*60}")
    print(f"{'임계값':>8}  {'베팅수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"{'-'*40}")
    for th in THRESHOLDS:
        sub = df_res[df_res["t3_prob"] >= th]
        if sub.empty:
            print(f"{th:>8.2f}  {0:>6}  {'N/A':>7}  {'0.0%':>8}")
            continue
        acc = sub["win3"].mean() * 100
        cov = len(sub) / total * 100
        print(f"{th:>8.2f}  {len(sub):>6}  {acc:>6.1f}%  {cov:>7.1f}%")


if __name__ == "__main__":
    main()
