"""
단승(1착) 백테스트

목표: 경주별로 모델이 고른 '1순위 말'이 실제로 1착인 비율을 계산하고,
      win_probability 임계값·두수 필터별 적중률을 확인한다.
      함께 "가장 인기있는(직전 배당 최저) 말을 그대로 찍었을 때"의 적중률과
      비교해 모델이 시장(배당) 대비 우위가 있는지 판단한다.

실행:
    python scripts/backtest_win.py
    python scripts/backtest_win.py --start 2025-01-01 --end 2026-06-07
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts._backtest_common import run_predictions


def run_backtest(start_date: str, end_date: str):
    result_df = run_predictions(start_date, end_date)
    if result_df is None:
        return

    # 경주별 모델 픽(win_prob 최고) + 인기마 픽(prev_win_odds 최저) 선정
    # 주의: prev_win_odds는 직전 배당 기록이 없는 말(데뷔마 등)의 경우
    # feature_builder에서 전체 중앙값으로 대체되므로, 그런 말이 여러 마리인
    # 경주에서는 인기마 픽이 동률 중 임의(행 순서)로 뽑힐 수 있다.
    def best_pick(grp):
        best = grp.loc[grp["win_prob"].idxmax()]
        favorite = grp.loc[grp["prev_win_odds"].idxmin()]
        return pd.Series({
            "race_date":       best["race_date"],
            "win_prob":        best["win_prob"],
            "is_win":          best["is_win_actual"],
            "n_horses":        best["n_horses"],
            "is_favorite_win": favorite["is_win_actual"],
        })

    race_results = result_df.groupby("race_id", sort=False).apply(best_pick).reset_index()

    total = len(race_results)
    overall_acc = race_results["is_win"].mean()
    favorite_acc = race_results["is_favorite_win"].mean()

    print(f"\n{'='*55}")
    print(f"  단승 백테스트 결과: {start_date} ~ {end_date}")
    print(f"{'='*55}")
    print(f"  총 경주 수:                {total}")
    print(f"  모델 Precision@1(필터 없음): {overall_acc:.1%}")
    print(f"  인기마(최저배당) 기준선:        {favorite_acc:.1%}")
    print()

    # ── win_probability 임계값별 적중률 ─────────────────────────────────────
    print("[ win_probability 임계값별 적중률 ]")
    print(f"  {'임계값':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for thresh in [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sub = race_results[race_results["win_prob"] >= thresh]
        if len(sub) == 0:
            break
        acc  = sub["is_win"].mean()
        ratio = len(sub) / total
        print(f"  {thresh:>7.2f}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

    # ── 두수 필터별 적중률 ───────────────────────────────────────────────────
    print()
    print("[ 두수 필터별 적중률 ]")
    print(f"  {'두수 ≤':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for max_h in [999, 12, 10, 9, 8, 7, 6]:
        sub = race_results[race_results["n_horses"] <= max_h]
        if len(sub) == 0:
            continue
        acc  = sub["is_win"].mean()
        ratio = len(sub) / total
        label = "전체" if max_h == 999 else str(max_h)
        print(f"  {label:>7}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단승 백테스트")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2026-06-07")
    args = parser.parse_args()
    run_backtest(args.start, args.end)
