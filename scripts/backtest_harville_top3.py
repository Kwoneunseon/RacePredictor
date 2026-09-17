"""
Harville 유도 연승확률 백테스트.

core/trainer.py의 top3 전용 classifier를 전혀 쓰지 않고, 이미 검증된
win_prob(캘리브레이션 적용)만으로 core/rank_distribution.py의 Harville
순차소거 공식을 적용해 연승확률을 유도한다
([[project_multi_bet_probability_model]] Phase 2, 2단계).

기존 raw classifier 벤치마크(TOP3_PROB_THRESHOLD=0.65 -> 약 80% 적중률/
61.6% 커버리지, 408경주, 2026-07-04 검증, [[project_next_task]])와 동일
기간·동일 리포트 구조(backtest_top3.py)로 비교한다.

캘리브레이션은 이 라운드에서 생략한다 - 임계값 스윕/픽 비교는 확률의
상대적 순서만 있으면 유효하고, EV 계산에 쓰이는 게 아니기 때문
(docs/superpowers/specs/2026-09-17-harville-top3-pair-comparison-design.md 참고).

실행:
    python scripts/backtest_harville_top3.py
    python scripts/backtest_harville_top3.py --start 2025-01-01 --end 2026-04-30
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts._backtest_common import run_predictions, capture_report, attach_harville_top3

# 기존 raw classifier 벤치마크(408경주, [[project_next_task]] 기록값) - 콘솔 인용용
EXISTING_CLASSIFIER_BENCHMARK = "0.65 임계값 -> 약 80% 적중률 / 61.6% 커버리지 (408경주, 2026-07-04 검증)"

# 기존 raw classifier의 0.65 임계값이 통과시켰던 경주 비율 - 동일 커버리지 지점 비교용
EXISTING_COVERAGE_RATIO = 0.616

THRESHOLDS = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def run_backtest(start_date: str, end_date: str):
    result_df = run_predictions(start_date, end_date)
    if result_df is None:
        return

    with capture_report("harville_top3") as rc:
        result_df = attach_harville_top3(result_df)
        result_df = result_df.dropna(subset=["harville_top3_prob"])
        if result_df.empty:
            print("[오류] Harville 연승확률 계산 가능한 경주 없음")
            return

        def best_pick(grp):
            best = grp.loc[grp["harville_top3_prob"].idxmax()]
            return pd.Series({
                "race_date":          best["race_date"],
                "harville_top3_prob": best["harville_top3_prob"],
                "win_prob":           best["win_prob"],
                "is_top3":            best["is_top3_actual"],
                "n_horses":           best["n_horses"],
            })

        race_results = result_df.groupby("race_id", sort=False).apply(best_pick).reset_index()
        total = len(race_results)
        overall_acc = race_results["is_top3"].mean()

        print(f"\n{'='*70}")
        print(f"  Harville 연승확률 백테스트: {start_date} ~ {end_date}")
        print(f"{'='*70}")
        print(f"  비교 대상(기존 raw classifier): {EXISTING_CLASSIFIER_BENCHMARK}")
        print(f"  총 경주 수:            {total}")
        print(f"  전체 적중률(필터 없음): {overall_acc:.1%}")

        print()
        print("[ harville_top3_prob 임계값별 적중률 ]")
        print(f"  {'임계값':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
        print(f"  {'-'*38}")
        for thresh in THRESHOLDS:
            sub = race_results[race_results["harville_top3_prob"] >= thresh]
            if len(sub) == 0:
                break
            acc = sub["is_top3"].mean()
            ratio = len(sub) / total
            print(f"  {thresh:>7.2f}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

        print()
        print("[ 기존 0.65 임계값과 동일 커버리지 지점 비교 ]")
        cutoff_n = max(int(total * EXISTING_COVERAGE_RATIO), 1)
        race_sorted = race_results.sort_values("harville_top3_prob", ascending=False)
        matched = race_sorted.iloc[:cutoff_n]
        print(f"  기존과 동일 커버리지({EXISTING_COVERAGE_RATIO:.1%}, 상위 {cutoff_n}경주)일 때 "
              f"Harville 적중률: {matched['is_top3'].mean():.1%}")
        print(f"  (기존 raw classifier는 동일 커버리지에서 약 80% 적중 - {EXISTING_CLASSIFIER_BENCHMARK})")

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harville 연승확률 백테스트")
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end",   default="2026-06-25")
    args = parser.parse_args()
    run_backtest(args.start, args.end)
