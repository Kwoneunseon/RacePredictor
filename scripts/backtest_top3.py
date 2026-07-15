"""
연승(3등 이내) 백테스트

목표: 경주별로 모델이 고른 '1순위 말'이 실제로 3등 안에 드는 비율을 계산하고,
      두수 필터 × 확신도(top3_probability) 임계값 조합에서 최적 조건을 찾는다.

실행:
    python scripts/backtest_top3.py
    python scripts/backtest_top3.py --start 2025-01-01 --end 2026-06-07
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts._backtest_common import run_predictions


def run_backtest(start_date: str, end_date: str):
    result_df = run_predictions(start_date, end_date)
    if result_df is None:
        return

    # 경주별 top3_prob 최고 말 선택
    def best_pick(grp):
        best = grp.loc[grp["top3_prob"].idxmax()]
        return pd.Series({
            "race_date":  best["race_date"],
            "top3_prob":  best["top3_prob"],
            "win_prob":   best["win_prob"],
            "is_top3":    best["is_top3_actual"],
            "n_horses":   best["n_horses"],
        })

    race_results = result_df.groupby("race_id", sort=False).apply(best_pick).reset_index()

    total = len(race_results)
    overall_acc = race_results["is_top3"].mean()

    print(f"\n{'='*55}")
    print(f"  백테스트 결과: {start_date} ~ {end_date}")
    print(f"{'='*55}")
    print(f"  총 경주 수:          {total}")
    print(f"  전체 적중률(필터 없음): {overall_acc:.1%}")
    print()

    # ── 두수 필터만 ───────────────────────────────────────────────────────
    print("[ 두수 필터별 적중률 ]")
    print(f"  {'두수 ≤':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for max_h in [999, 12, 10, 9, 8, 7, 6]:
        sub = race_results[race_results["n_horses"] <= max_h]
        if len(sub) == 0:
            continue
        acc  = sub["is_top3"].mean()
        ratio = len(sub) / total
        label = "전체" if max_h == 999 else str(max_h)
        print(f"  {label:>7}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

    # ── 확신도 임계값만 ───────────────────────────────────────────────────
    print()
    print("[ top3_probability 임계값별 적중률 ]")
    print(f"  {'임계값':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for thresh in [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sub = race_results[race_results["top3_prob"] >= thresh]
        if len(sub) == 0:
            break
        acc  = sub["is_top3"].mean()
        ratio = len(sub) / total
        print(f"  {thresh:>7.2f}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

    # ── 두수 × 임계값 조합 (80% 라인 탐색) ────────────────────────────────
    print()
    print("[ 조합 분석 (적중률 ≥ 70% 조건만 표시) ]")
    print(f"  {'임계값':>7}  {'두수 ≤':>6}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*46}")

    rows = []
    for thresh in [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for max_h in [999, 12, 10, 9, 8, 7, 6]:
            mask = (race_results["top3_prob"] >= thresh) & (race_results["n_horses"] <= max_h)
            sub  = race_results[mask]
            if len(sub) < 10:
                continue
            acc  = sub["is_top3"].mean()
            ratio = len(sub) / total
            rows.append((thresh, max_h, len(sub), acc, ratio))

    # 적중률 내림차순 정렬
    rows.sort(key=lambda x: -x[3])
    for thresh, max_h, cnt, acc, ratio in rows:
        if acc < 0.70:
            continue
        label = "전체" if max_h == 999 else str(max_h)
        print(f"  {thresh:>7.2f}  {label:>6}  {cnt:>6}  {acc:>7.1%}  {ratio:>8.1%}")

    # ── 80% 달성을 위한 요약 권고 ────────────────────────────────────────
    print()
    print("[ 80% 달성 권고 조건 ]")
    found = False
    for thresh, max_h, cnt, acc, ratio in rows:
        if acc >= 0.80 and cnt >= 10:
            label = "전체" if max_h == 999 else f"≤{max_h}두"
            print(f"  top3_prob >= {thresh:.2f} + 두수 {label} → {cnt}경주, 적중률 {acc:.1%}")
            found = True
            break

    if not found:
        # 가장 근접한 조건 찾기
        best_row = max(rows, key=lambda x: (x[3], x[2])) if rows else None
        if best_row:
            thresh, max_h, cnt, acc, ratio = best_row
            label = "전체" if max_h == 999 else f"≤{max_h}두"
            print(f"  현재 최고: top3_prob >= {thresh:.2f} + 두수 {label} → {cnt}경주, 적중률 {acc:.1%}")
            gap = 0.80 - acc
            print(f"  80% 목표까지 {gap:.1%} 차이 → 모델 개선 또는 추가 필터 필요")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="연승 백테스트")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2026-06-07")
    args = parser.parse_args()
    run_backtest(args.start, args.end)
