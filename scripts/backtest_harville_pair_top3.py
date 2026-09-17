"""
Harville pair(복연승) 확률 백테스트.

기존 임시 휴리스틱(scripts/backtest_quinella_place_heuristic.py - top3_prob
상위 2마리를 그냥 묶는 방식)과 무작위 베이스라인 대비, core/rank_distribution.py의
pair_top3_probabilities()로 "두 말이 동시에 top3에 들 결합확률"을 직접 최대화하는
페어를 픽했을 때 정확도가 더 나은지 비교한다
([[project_multi_bet_probability_model]] Phase 2, 3단계).

이 스크립트는 ROI/배당을 계산하지 않는다(복연승 조합 배당 데이터가 없음,
[[project_race_payouts_scraper]] 참고) - 적중률만 비교한다. 캘리브레이션도
이 라운드에서 생략한다(EV 아님, docs/superpowers/specs/
2026-09-17-harville-top3-pair-comparison-design.md 참고).

실행:
    python scripts/backtest_harville_pair_top3.py
    python scripts/backtest_harville_pair_top3.py --dev-start 2025-01-01 --dev-end 2026-04-30 \
        --test-start 2026-05-01 --test-end 2026-06-25
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts._backtest_common import run_predictions, capture_report, build_harville_pair_table
from scripts.backtest_quinella_place_heuristic import build_quinella_table

DEV_START_DEFAULT, DEV_END_DEFAULT = "2025-01-01", "2026-04-30"
TEST_START_DEFAULT, TEST_END_DEFAULT = "2026-05-01", "2026-06-25"

THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _threshold_sweep(df: pd.DataFrame, score_col: str, hit_col: str, label: str):
    total = len(df)
    print(f"\n  [{label}]")
    print(f"  {'임계값':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for thresh in THRESHOLDS:
        sub = df[df[score_col] >= thresh]
        if len(sub) == 0:
            break
        acc = sub[hit_col].mean()
        ratio = len(sub) / total
        print(f"  {thresh:>7.2f}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")


def report_window(harville_df: pd.DataFrame, heuristic_df: pd.DataFrame,
                   label: str, start_date: str, end_date: str):
    print(f"\n{'='*70}")
    print(f"  Harville 복연승 백테스트 - {label}: {start_date} ~ {end_date} ({len(harville_df)}경주)")
    print(f"{'='*70}")

    if harville_df.empty:
        print("  경주 없음 - 스킵")
        return

    total = len(harville_df)
    random_avg = harville_df["random_baseline"].mean()
    harville_acc = harville_df["harville_pair_hit"].mean()

    # 두 방식이 정확히 같은 경주 집합을 대상으로 하도록 race_id로 inner join
    merged = harville_df.merge(
        heuristic_df[["race_id", "quinella_hit_top3prob"]], on="race_id", how="inner"
    )
    heuristic_acc = merged["quinella_hit_top3prob"].mean()

    print(f"\n  [베이스라인] 완전 무작위 2마리 픽 기대 적중률(두수 가중평균): {random_avg:.1%}")
    print(f"  [기존 휴리스틱] top3_prob 상위 2마리 픽 적중률:           {heuristic_acc:.1%}  ({len(merged)}경주)")
    print(f"  [Harville] pair 확률 최댓값 페어 픽 적중률:               {harville_acc:.1%}  ({total}경주)")
    print(f"    무작위 대비 {harville_acc/random_avg:.2f}배, 기존 휴리스틱 대비 {harville_acc - heuristic_acc:+.1%}p")

    _threshold_sweep(harville_df, "harville_pair_prob", "harville_pair_hit",
                      "임계값 스윕 (픽된 페어의 harville_pair_prob >= 임계값)")


def run_backtest(dev_start: str, dev_end: str, test_start: str, test_end: str):
    result_df = run_predictions(dev_start, test_end)
    if result_df is None:
        return

    with capture_report("harville_pair_top3") as rc:
        harville_df = build_harville_pair_table(result_df)
        heuristic_df = build_quinella_table(result_df)
        if harville_df.empty:
            print("[오류] Harville 복연승 계산 가능한 경주 없음(전부 3두 미만이거나 강도합<=0)")
            return

        harville_df["race_date"] = harville_df["race_date"].astype(str)
        heuristic_df["race_date"] = heuristic_df["race_date"].astype(str)

        print(f"\n{'#'*70}")
        print(f"# Harville 복연승(pair_top3_probabilities) vs 기존 휴리스틱 vs 무작위")
        print(f"# dev : {dev_start} ~ {dev_end}")
        print(f"# test: {test_start} ~ {test_end}  (기존 408경주 검증 벤치마크 창)")
        print(f"{'#'*70}")

        dev_h  = harville_df[(harville_df["race_date"] >= dev_start) & (harville_df["race_date"] <= dev_end)]
        dev_q  = heuristic_df[(heuristic_df["race_date"] >= dev_start) & (heuristic_df["race_date"] <= dev_end)]
        test_h = harville_df[(harville_df["race_date"] >= test_start) & (harville_df["race_date"] <= test_end)]
        test_q = heuristic_df[(heuristic_df["race_date"] >= test_start) & (heuristic_df["race_date"] <= test_end)]

        if len(dev_h):
            report_window(dev_h, dev_q, "DEV", dev_start, dev_end)
        report_window(test_h, test_q, "TEST (408경주 벤치마크)", test_start, test_end)

        print(f"\n{'='*70}")
        print("  주의: 이 스크립트는 ROI/배당을 계산하지 않는다(복연승 조합 배당")
        print("  데이터 없음). 위 수치는 '적중률'만 의미한다.")
        print(f"{'='*70}")

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harville 복연승 백테스트")
    parser.add_argument("--dev-start", default=DEV_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    args = parser.parse_args()
    run_backtest(args.dev_start, args.dev_end, args.test_start, args.test_end)
