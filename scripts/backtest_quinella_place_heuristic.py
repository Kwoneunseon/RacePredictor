"""
복연승(2두 조합, 순서무관, 둘 다 1~3착) 휴리스틱 백테스트

가설: 정식 Harville 조합확률 모델([[project_multi_bet_probability_model]], 아직
미구현)이 없어도, 이미 검증된 단일 말 확률(top3_probability 또는
win_probability)로 경주별 상위 2마리를 뽑아 복연승 픽으로 삼으면 쓸 만한
적중률이 나오지 않을까?

이 스크립트는 core/predictor.py, core/trainer.py를 전혀 건드리지 않고,
scripts/_backtest_common.py의 run_predictions()가 반환하는 raw 확률
(top3_prob은 raw=True, [[project_top3_no_blend]] 원칙 준수)을 그대로 재사용해
"경주별 상위 2마리 선택"이라는 사후(post-hoc) 픽 로직만 새로 추가한다.

주의: 복연승 "배당" 데이터는 여전히 없다(조합 단위 배당이라 제주/부산경남
공공API의 plcOdds로도 못 얻음, [[project_race_payouts_scraper]] 참고). 이
스크립트는 ROI가 아니라 "적중률"만 검증한다 — 적중 여부는 배당과 무관하게
race_entries.final_rank(실제 결과)만으로 계산 가능하기 때문이다.

동률(tie) 처리: 경주 내 정렬 시 1차 키=대상 확률(top3_prob 또는 win_prob),
2차 키=보조 확률(win_prob 또는 top3_prob), 그래도 동률이면 pandas의 stable
sort(mergesort)가 원래 행 순서(DB 조회 순서)를 그대로 유지해 결정론적으로
같은 2마리가 뽑히도록 한다.

베이스라인:
  (a) 완전 무작위 2마리 픽의 기대 적중률 = C(3,2)/C(n,2) = 6/(n(n-1))
      (출전마 n두 중 정확히 3마리만 "1~3착"이므로, 비복원추출로 2마리를
      뽑았을 때 그 2마리가 모두 이 3마리 안에 들 확률의 closed-form 기댓값)
  (b) 기존 검증된 단일 top3 픽 적중률(TOP3_PROB_THRESHOLD=0.65 게이트,
      408경주 기준 약 70~71% 커버리지/80% 안팎 적중) — 같은 데이터셋에서
      동일하게 재계산해 나란히 비교한다.

실행:
    python scripts/backtest_quinella_place_heuristic.py
    python scripts/backtest_quinella_place_heuristic.py --dev-start 2025-01-01 --dev-end 2026-04-30 \
        --test-start 2026-05-01 --test-end 2026-06-25
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts._backtest_common import run_predictions, capture_report

# 기존에 검증된 연승(단일 픽) 임계값 - [[project_next_task]] 408경주 백테스트
TOP3_PROB_THRESHOLD = 0.65

# 기존 검증된 408경주 벤치마크 창(2026-07-04, 2026-08-02 재확인)
TEST_START_DEFAULT, TEST_END_DEFAULT = "2026-05-01", "2026-06-25"
# 더 넓은 dev 구간(선택) - CALIBRATION_EVAL_CUTOFF(2026-05-01)보다 이전만 사용해
# 캘리브레이션 학습 데이터와 겹치지 않게 한다.
DEV_START_DEFAULT, DEV_END_DEFAULT = "2025-01-01", "2026-04-30"

THRESHOLDS = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _random_baseline_prob(n: int) -> float:
    """출전마 n두 중 무작위로 2마리를 뽑았을 때 둘 다 1~3착일 기대 확률.
    = C(3,2)/C(n,2) = 6 / (n*(n-1)).  n<3이면 정의 불가(예외적으로 스킵)."""
    if n < 3:
        return float("nan")
    return 6.0 / (n * (n - 1))


def _pick_top2(grp: pd.DataFrame, score_col: str, tiebreak_col: str) -> pd.DataFrame:
    """경주(그룹) 내 score_col 내림차순 → 동률이면 tiebreak_col 내림차순 →
    그래도 동률이면 원래 행 순서(stable sort) 순으로 정렬해 상위 2행 반환."""
    ordered = grp.sort_values([score_col, tiebreak_col], ascending=[False, False], kind="mergesort")
    return ordered.iloc[:2]


def build_quinella_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """말 단위 result_df -> 경주 단위 복연승 휴리스틱 테이블.

    컬럼:
      n_horses, quinella_hit_top3prob/winprob(둘 다 top3인지),
      quinella_minprob_top3/winprob(선택된 2마리 중 더 약한 쪽의 확률 - 임계값
      스윕용), single_top1_hit_top3prob(비교용 기존 단일 픽 적중 여부),
      top3_prob_top1(단일 픽 확률, 0.65 게이트 비교용), random_baseline.
    """
    rows = []
    skipped_small = 0
    for race_id, grp in result_df.groupby("race_id", sort=False):
        n = len(grp)
        if n < 3:
            skipped_small += 1
            continue

        top2_t3 = _pick_top2(grp, "top3_prob", "win_prob")
        top2_wp = _pick_top2(grp, "win_prob", "top3_prob")

        hit_t3 = bool((top2_t3["final_rank"] <= 3).all())
        hit_wp = bool((top2_wp["final_rank"] <= 3).all())

        single_top1 = grp.sort_values(["top3_prob", "win_prob"], ascending=[False, False], kind="mergesort").iloc[0]

        rows.append({
            "race_id": race_id,
            "race_date": str(top2_t3["race_date"].iloc[0]),
            "n_horses": n,
            "quinella_hit_top3prob": hit_t3,
            "quinella_hit_winprob": hit_wp,
            "quinella_minprob_top3": float(top2_t3["top3_prob"].min()),
            "quinella_minprob_winprob": float(top2_wp["win_prob"].min()),
            "top3_prob_top1": float(single_top1["top3_prob"]),
            "single_top1_hit": bool(single_top1["final_rank"] <= 3),
            "random_baseline": _random_baseline_prob(n),
        })

    if skipped_small:
        print(f"  [주의] 출전마 3두 미만 경주 {skipped_small}건은 복연승 정의상 스킵")

    return pd.DataFrame(rows)


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


def report_window(df: pd.DataFrame, label: str, start_date: str, end_date: str):
    print(f"\n{'='*70}")
    print(f"  복연승 휴리스틱 백테스트 - {label}: {start_date} ~ {end_date} ({len(df)}경주)")
    print(f"{'='*70}")

    if df.empty:
        print("  경주 없음 - 스킵")
        return

    total = len(df)

    # n_horses 분포 (한국 경마는 특정 두수에 몰리는 경향이 있어 참고용으로 출력)
    dist = df["n_horses"].value_counts().sort_index()
    print("\n  [출전마 두수 분포]")
    for n_h, cnt in dist.items():
        print(f"    {n_h:>2}두: {cnt:>4}경주 ({cnt/total:.1%})")

    # ── 베이스라인: 완전 무작위 2마리 픽 기대 적중률 ─────────────────────
    random_avg = df["random_baseline"].mean()
    print(f"\n  [베이스라인] 완전 무작위 2마리 픽 기대 적중률(두수 가중평균): {random_avg:.1%}")

    # ── 기존 검증된 단일 top3 픽(비교 기준선) ────────────────────────────
    single_acc_all = df["single_top1_hit"].mean()
    gated = df[df["top3_prob_top1"] >= TOP3_PROB_THRESHOLD]
    single_acc_gated = gated["single_top1_hit"].mean() if len(gated) else float("nan")
    print(f"\n  [비교기준] 기존 단일 top3 픽(top3_prob 최고 1마리만) 적중률")
    print(f"    필터 없음:                     {single_acc_all:.1%}  ({total}경주)")
    print(f"    top3_prob >= {TOP3_PROB_THRESHOLD} 게이트(기존 검증값): "
          f"{single_acc_gated:.1%}  ({len(gated)}경주, 커버리지 {len(gated)/total:.1%})")

    # ── 복연승 휴리스틱: top3_prob 상위 2 vs win_prob 상위 2 ─────────────
    acc_t3 = df["quinella_hit_top3prob"].mean()
    acc_wp = df["quinella_hit_winprob"].mean()
    print(f"\n  [복연승 휴리스틱 - 필터 없음, 전체 {total}경주]")
    print(f"    top3_probability 상위 2마리 픽:  적중률 {acc_t3:.1%}  "
          f"(무작위 대비 {acc_t3/random_avg:.2f}배, 단일픽(무필터) 대비 {acc_t3-single_acc_all:+.1%}p)")
    print(f"    win_probability  상위 2마리 픽:  적중률 {acc_wp:.1%}  "
          f"(무작위 대비 {acc_wp/random_avg:.2f}배)")

    # ── 두 픽이 얼마나 겹치는지(같은 2마리를 뽑는지) 참고 정보 ────────────
    same_pick_rate = (df["quinella_hit_top3prob"] == df["quinella_hit_winprob"]).mean()
    print(f"    (참고) 두 방식의 적중 여부 일치율: {same_pick_rate:.1%}")

    # ── 임계값(선택된 2마리 중 더 약한 쪽의 확률) 스윕 ───────────────────
    print(f"\n  [임계값 스윕] 선택된 2마리 중 더 약한 쪽의 확률 >= 임계값")
    _threshold_sweep(df, "quinella_minprob_top3", "quinella_hit_top3prob",
                      "top3_probability 기준 (약한 쪽 top3_prob >= 임계값)")
    _threshold_sweep(df, "quinella_minprob_winprob", "quinella_hit_winprob",
                      "win_probability 기준 (약한 쪽 win_prob >= 임계값)")

    # ── 기존 0.65 단일픽 게이트를 통과한 경주 내에서 복연승 적중률 ───────
    print(f"\n  [기존 단일픽 0.65 게이트 통과 경주 내 복연승 적중률]")
    if len(gated):
        acc_t3_gated = gated["quinella_hit_top3prob"].mean()
        random_avg_gated = gated["random_baseline"].mean()
        print(f"    {len(gated)}경주, top3_probability 상위 2마리 픽 적중률: {acc_t3_gated:.1%} "
              f"(이 부분집합 무작위 기대치 {random_avg_gated:.1%})")
    else:
        print("    대상 없음")


def run_backtest(dev_start: str, dev_end: str, test_start: str, test_end: str):
    result_df = run_predictions(dev_start, test_end)
    if result_df is None:
        return

    with capture_report("quinella_place_heuristic") as rc:
        quinella_df = build_quinella_table(result_df)
        if quinella_df.empty:
            print("[오류] 복연승 계산 가능한 경주 없음(전부 3두 미만)")
            return
        quinella_df["race_date"] = quinella_df["race_date"].astype(str)

        dev_df = quinella_df[(quinella_df["race_date"] >= dev_start) & (quinella_df["race_date"] <= dev_end)].copy()
        test_df = quinella_df[(quinella_df["race_date"] >= test_start) & (quinella_df["race_date"] <= test_end)].copy()

        print(f"\n{'#'*70}")
        print(f"# 복연승(2두 조합, 순서무관) 휴리스틱 백테스트")
        print(f"# dev : {dev_start} ~ {dev_end}")
        print(f"# test: {test_start} ~ {test_end}  (기존 408경주 검증 벤치마크 창)")
        print(f"{'#'*70}")

        if len(dev_df):
            report_window(dev_df, "DEV", dev_start, dev_end)
        report_window(test_df, "TEST (408경주 벤치마크)", test_start, test_end)

        # ── 결론 요약 ────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("  결론 요약")
        print(f"{'='*70}")
        if len(test_df):
            acc_t3 = test_df["quinella_hit_top3prob"].mean()
            acc_wp = test_df["quinella_hit_winprob"].mean()
            random_avg = test_df["random_baseline"].mean()
            better = "top3_probability" if acc_t3 >= acc_wp else "win_probability"
            print(f"  TEST 구간 기준 top3_probability 상위 2마리 픽 적중률: {acc_t3:.1%}")
            print(f"  TEST 구간 기준 win_probability  상위 2마리 픽 적중률: {acc_wp:.1%}")
            print(f"  완전 무작위 픽 기대 적중률(동일 기간 두수 가중):      {random_avg:.1%}")
            print(f"  -> 복연승 픽으로는 [{better}] 기준이 더 낫다(적중률 관점).")
            print(f"  주의: 이 스크립트는 ROI/배당을 계산하지 않는다 (복연승 조합 배당")
            print(f"  데이터가 여전히 없음, project_race_payouts_scraper 참고). 위 수치는")
            print(f"  '적중률'만 의미하며, 이 픽으로 실제 베팅했을 때 수익이 나는지는")
            print(f"  배당 데이터 확보 후 별도 검증이 필요하다.")

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="복연승 휴리스틱(top3_prob/win_prob 상위 2마리 픽) 백테스트")
    parser.add_argument("--dev-start", default=DEV_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    args = parser.parse_args()
    run_backtest(args.dev_start, args.dev_end, args.test_start, args.test_end)
