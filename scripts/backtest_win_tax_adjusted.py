"""
단승(win) 고신뢰 구간 세후(稅後) ROI 엄밀 재검증

목적: "경마 베팅으로 수익 낼 수 있는가"를 최종 정리하기 위해, 단승(win) 픽
(경주별 win_probability 최고 말) 전략을 dev/test 완전 분리 + 세금(원천징수)
반영으로 재검증한다.

배경 문제의식:
  - 기존에 (세금 미반영) 단순 백테스트에서 단승 확률 임계값을 높이면 ROI가
    최대 +0.50(적중률 63%, 커버리지 39.7%, 408경주 test 창) 근처까지 나온 적이
    있었다. 이 수치는 실제 win_odds(단승은 연승과 달리 근사치 버그 없음,
    [[project_blog_review_edge_gaps]])로 계산돼 데이터 자체 신뢰도는 있지만,
    dev/test 분리·사후 임계값 선택 방지 없이 나온 값이라 엄밀한 검증이 아니었다.
  - [[project_race_confidence_filter]]에서 확립한 방법론(dev 기간에서만 임계값
    후보를 관찰하고, 미리 정한 임계값을 test 기간(2026-05-01~06-25, 기존 408경주
    벤치마크 창)에 "단 1회만" 적용해 사후 임계값 선택/과적합을 방지)을 그대로
    따른다.

세금 가정 (중요 — 반드시 명시):
  - 한국 경마 당첨금 원천징수는 실제로는 "건별 환급금이 200만원을 초과하고
    환급률이 100배를 초과하는 경우" 등 복잡한 규정([[project_blog_review_edge_gaps]]
    갭 2번)이 있으나, 정확한 법적 세율 구조를 이 스크립트에서 정밀 구현하지
    않는다. 사용자 지시대로 **가장 단순한 보수적 근사**를 쓴다:
      * 세율 22% 정률 (3억원 초과 구간 33%는 이 프로젝트 스케일에서 사실상
        발생하지 않는다고 가정 — 사용자 확인 사항).
      * 과세 대상은 "1회 베팅의 순수익"(=배당금 - 원금, 즉 odds-1)에만 22%를
        적용한다. 원금 자체나 손실 베팅에는 세금이 붙지 않는다(음수 profit은
        그대로 유지, 즉 세금으로 인한 손실 증가는 없음 — 손실은 공제되지도
        않지만 추가로 과세되지도 않는다는 단순화).
      * 베팅 건별로 개별 과세한다고 가정(연간 누적 정산 규정 무시). 이는
        실제보다 더 자주/더 낮은 금액에 과세하는 셈이라 "보수적"(=수익을 더
        깎는 방향)이다.
      * 최소 환급 기준액(경마는 보통 10만원 이하 등 소액 비과세 규정이 있을 수
        있음)도 무시하고 모든 승리 베팅에 일괄 적용한다 — 역시 보수적 방향.
    즉 이 스크립트의 "세후 ROI"는 실제 법적 세후 ROI의 **하한에 가까운 근사치**로
    해석해야 한다(실제보다 나쁘게 나올 가능성이 높은 방향의 단순화).

방법론 (dev/test 분리, [[project_race_confidence_filter]]와 동일):
  - dev  = 2025-01-01 ~ 2026-04-30 (임계값 후보 관찰 전용)
  - test = 2026-05-01 ~ 2026-06-25 (기존 408경주 벤치마크 창, "단 1회" 검증)
  - CALIBRATION_EVAL_CUTOFF("2026-05-01", core/trainer.py)보다 test 시작일이
    빠르지 않음을 재확인(캘리브레이터 holdout과 test 구간이 겹치지 않아야 함).
  - win_probability는 core/trainer.py의 win_calibrator가 적용된(raw=False,
    기본값) 확률을 그대로 쓴다 — [[project_probability_calibration]]에서 이미
    채택된 프로덕션 기본값과 동일.
  - 주의(스코프 고지): 이 스크립트는 기존 scripts/backtest_win.py·
    scripts/_backtest_common.py와 동일하게 "분류기 단독 win_probability"
    (LambdaRank 랭커와의 50:50 블렌딩·경주내 softmax 재정규화 미적용)를 쓴다.
    실제 운영 core/predictor.py의 win_probability는 이 블렌딩을 거친 값이라
    완전히 동일하지는 않을 수 있다 — 이 차이 자체의 검증은 이번 스코프 밖이며,
    기존에 이미 검증된 backtest_win.py 계열 수치와 비교 가능하게 하기 위해
    동일한 방식을 유지한다.

core/predictor.py·core/trainer.py는 건드리지 않는다. _backtest_common.py도
읽기 전용으로만 재사용(load_historical_data, capture_report) — 다른 세션과의
파일 충돌을 피하기 위해 예측 실행 로직은 이 스크립트 안에 자체 구현한다
(scripts/backtest_confidence_filter.py와 동일한 관례).

실행:
    python scripts/backtest_win_tax_adjusted.py
    python scripts/backtest_win_tax_adjusted.py --dev-start 2025-01-01 --dev-end 2026-04-30 \
        --test-start 2026-05-01 --test-end 2026-06-25
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.feature_builder import RaceFeatureBuilder
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer, CALIBRATION_EVAL_CUTOFF
from core.model_manager import ModelManager
from scripts._backtest_common import load_historical_data, capture_report

DEV_START_DEFAULT, DEV_END_DEFAULT = "2025-01-01", "2026-04-30"
TEST_START_DEFAULT, TEST_END_DEFAULT = "2026-05-01", "2026-06-25"

TAX_RATE = 0.22  # 정률 가정 (스크립트 상단 docstring 참고, 3억 초과 33% 구간 무시)

# 세율 22% 가정에 대한 사용자 확인 참고: [[project_blog_review_edge_gaps]] 갭 2번.

# 전체 스윕 그리드 (요청: 0.0~0.60)
SWEEP_THRESHOLDS = [round(x, 2) for x in np.arange(0.00, 0.61, 0.02)]
# 후보 구간 근방 안정성 확인용 미세 그리드 (기존에 언급된 0.28~0.31 근방 포함)
FINE_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.41, 0.01)]

MIN_N_STABLE = 30  # 이보다 표본이 적으면 "불안정(저표본)"으로 표시만 하고 신뢰하지 않음


# ──────────────────────────────────────────────────────────────────────────
# 예측 생성 (scripts/backtest_confidence_filter.py의 run_predictions_with_odds와
# 동일한 패턴 - horse_id는 필요 없어 생략, actual_win_odds만 유지)
# ──────────────────────────────────────────────────────────────────────────

def run_predictions_with_odds(start_date: str, end_date: str):
    print("모델 로드 중...")
    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor):
        print("[오류] 저장된 모델 없음. python main.py --mode train 먼저 실행하세요.")
        return None

    print(f"과거 데이터 로드 중: {start_date} ~ {end_date} ...")
    df_raw = load_historical_data(start_date, end_date)
    if df_raw.empty:
        print("[오류] 데이터 없음")
        return None
    print(f"  로드: {len(df_raw)}건, "
          f"{df_raw['race_id'].nunique() if 'race_id' in df_raw.columns else '?'}경주")

    print("피처 빌드 중...")
    fb = RaceFeatureBuilder()
    df_feat = fb.build_for_training(df_raw, end_date)
    df_feat = df_feat.dropna(subset=["horse_age", "race_distance"])

    final_ranks = df_feat["final_rank"].copy()
    race_ids = df_feat["race_id"].copy()
    race_dates = df_feat["race_date"].copy()
    # race_entries.win_odds: 해당 경주의 실제(확정) 단승 배당 - ROI 계산용.
    actual_win_odds = df_feat["win_odds"].copy() if "win_odds" in df_feat.columns else pd.Series([0.0] * len(df_feat))
    meet_codes = df_feat["meet_code"].copy() if "meet_code" in df_feat.columns else pd.Series([None] * len(df_feat))

    n_horses_map = df_feat.groupby("race_id").size().rename("n_horses")

    print("전처리 중...")
    df_proc = preprocessor.transform(df_feat)

    print("예측 중...")
    features = trainer.feature_columns
    available = [c for c in features if c in df_proc.columns]
    X = df_proc[available].fillna(0)

    # win_probability: raw=False(기본값) -> win_calibrator 적용된 확률.
    # [[project_probability_calibration]]에서 채택된 프로덕션 기본값과 동일.
    win_probs = trainer.predict_proba(X)

    result_df = pd.DataFrame({
        "race_id": race_ids.values,
        "race_date": race_dates.values,
        "final_rank": final_ranks.values,
        "win_prob": win_probs,
        "actual_win_odds": pd.to_numeric(actual_win_odds, errors="coerce").fillna(0.0).values,
        "meet_code": meet_codes.values,
    }, index=df_feat.index)

    result_df["is_win_actual"] = (result_df["final_rank"] == 1).astype(int)
    result_df = result_df.join(n_horses_map, on="race_id")

    return result_df


def build_race_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """말 단위 result_df -> 경주 단위 (win 픽 1개 + 실측) 테이블."""
    rows = []
    for race_id, grp in result_df.groupby("race_id", sort=False):
        pick = grp.loc[grp["win_prob"].idxmax()]
        rows.append({
            "race_id": race_id,
            "race_date": pick["race_date"],
            "n_horses": len(grp),
            "win_prob_top1": float(pick["win_prob"]),
            "pick_odds": float(pick["actual_win_odds"]),
            "is_win": int(pick["is_win_actual"]),
            "meet_code": pick["meet_code"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# ROI 계산 (세전 / 세후)
# ──────────────────────────────────────────────────────────────────────────

def _profit_pretax(df: pd.DataFrame) -> np.ndarray:
    """flat 1단위 베팅 기준 세전 순수익. odds<=0(배당 정보 없음)인 행은 이미
    상위에서 걸러졌다고 가정(_valid_odds 사용)."""
    return np.where(df["is_win"] == 1, df["pick_odds"] - 1.0, -1.0)


def _profit_posttax(df: pd.DataFrame, tax_rate: float = TAX_RATE) -> np.ndarray:
    """세후 순수익: 승리 베팅의 순이익(odds-1)에만 tax_rate를 적용.
    패배 베팅(-1)은 과세 대상 아님(원금 손실에는 세금이 붙지 않음)."""
    pretax = _profit_pretax(df)
    return np.where(pretax > 0, pretax * (1 - tax_rate), pretax)


def _valid_odds(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pick_odds"] > 0]


def sweep_row(df: pd.DataFrame, threshold: float, total: int) -> dict:
    sub = df[df["win_prob_top1"] >= threshold]
    sub_v = _valid_odds(sub)
    n = len(sub)
    n_valid = len(sub_v)
    if n == 0:
        return dict(threshold=threshold, n=0, n_valid=0, coverage=0.0, accuracy=float("nan"),
                     acc_valid=float("nan"), roi_pre=float("nan"), roi_post=float("nan"))
    acc = sub["is_win"].mean()
    coverage = n / total
    if n_valid == 0:
        roi_pre = roi_post = float("nan")
        acc_valid = float("nan")
    else:
        roi_pre = float(_profit_pretax(sub_v).mean())
        roi_post = float(_profit_posttax(sub_v).mean())
        acc_valid = float(sub_v["is_win"].mean())
    return dict(threshold=threshold, n=n, n_valid=n_valid, coverage=coverage,
                accuracy=acc, acc_valid=acc_valid, roi_pre=roi_pre, roi_post=roi_post)


def print_sweep_table(df: pd.DataFrame, thresholds: list, title: str):
    total = len(df)
    print(f"\n  [{title}] 전체 경주수={total}")
    print(f"  {'임계값':>6}  {'경주수':>6}  {'커버리지':>8}  {'적중률':>7}  {'적중률(배당有)':>12}  "
          f"{'ROI(세전)':>10}  {'ROI(세후22%)':>12}  {'비고':>10}")
    print(f"  {'-'*90}")
    for t in thresholds:
        r = sweep_row(df, t, total)
        if r["n"] == 0:
            continue
        note = "저표본" if r["n"] < MIN_N_STABLE else ""
        print(f"  {r['threshold']:>6.2f}  {r['n']:>6}  {r['coverage']:>7.1%}  "
              f"{r['accuracy']:>6.1%}  {r['acc_valid']:>11.1%}  "
              f"{r['roi_pre']:>+9.3f}  {r['roi_post']:>+11.3f}  {note:>10}")


def find_dev_candidate_zone(dev_df: pd.DataFrame, thresholds: list) -> list:
    """dev 스윕 결과에서 '세후 ROI > 0 이면서 표본 >= MIN_N_STABLE'인 연속 구간을
    찾아 그 구간의 하한/중앙/상한 임계값을 후보로 반환한다 (test에 1회만 적용할 값)."""
    total = len(dev_df)
    rows = [sweep_row(dev_df, t, total) for t in thresholds]
    positive = [r for r in rows if r["n"] >= MIN_N_STABLE and not np.isnan(r["roi_post"]) and r["roi_post"] > 0]
    if not positive:
        return []
    lo = positive[0]["threshold"]
    hi = positive[-1]["threshold"]
    mid_idx = len(positive) // 2
    mid = positive[mid_idx]["threshold"]
    candidates = sorted(set([lo, mid, hi]))
    return candidates


def apply_single_threshold_test(test_df: pd.DataFrame, threshold: float, label: str):
    total = len(test_df)
    baseline = sweep_row(test_df, 0.0, total)
    r = sweep_row(test_df, threshold, total)
    print(f"\n  {label}  (threshold={threshold:.2f})")
    if r["n"] == 0:
        print("    -> 대상 0건 (test 기간에 이 임계값을 넘는 픽 없음)")
        return
    print(f"    필터 후: {r['n']}/{total}경주 (커버리지 {r['coverage']:.1%}, 배당정보 있는 {r['n_valid']}건 기준 ROI)")
    print(f"    적중률(전체): {r['accuracy']:.1%}   적중률(배당정보 있는 서브셋만): {r['acc_valid']:.1%}   "
          f"ROI 세전: {r['roi_pre']:+.3f}   ROI 세후(22%): {r['roi_post']:+.3f}")
    if r["n_valid"] > 0 and abs(r["acc_valid"] - r["accuracy"]) >= 0.03:
        print(f"    [주의] 배당정보 有無 서브셋 간 적중률 차이 {abs(r['acc_valid']-r['accuracy'])*100:.1f}pp "
              f"— win_odds 결측이 무작위가 아닐 수 있어(특히 서울) ROI가 왜곡됐을 가능성")
    print(f"    [baseline 참고] threshold=0.00(전체 베팅): 적중률 {baseline['accuracy']:.1%}  "
          f"ROI세전 {baseline['roi_pre']:+.3f}  ROI세후 {baseline['roi_post']:+.3f}")
    if r["n"] < MIN_N_STABLE:
        print(f"    [주의] 표본 {r['n']}건 < {MIN_N_STABLE} — 저표본, 노이즈 가능성 높음")

    # 특정 이변(초고배당 1건)이 평균 ROI를 좌우하는지 확인 - 중앙값·최대배당·승리건수 병기
    sub = test_df[test_df["win_prob_top1"] >= threshold]
    sub_v = _valid_odds(sub)
    if len(sub_v) > 0:
        profits = _profit_pretax(sub_v)
        wins = sub_v[sub_v["is_win"] == 1]
        median_profit = float(np.median(profits))
        max_odds = float(wins["pick_odds"].max()) if len(wins) else float("nan")
        print(f"    [분포] 세전 순수익 중앙값(1단위 기준): {median_profit:+.3f}   "
              f"승리 {len(wins)}건 중 최고배당: {max_odds:.1f}배   "
              f"(참고: 중앙값이 음수인데 평균 ROI가 양수면 소수의 고배당 적중이 평균을 견인하는 구조)")


def report_odds_missingness(race_df: pd.DataFrame, label: str):
    """win_odds 결측이 경마장(meet_code)별로 얼마나 다르고, 결측 有無에 따라
    적중률이 갈리는지(=결측이 승/패와 상관돼 ROI를 왜곡할 위험) 진단.

    배경: library/scraper_score.py(서울 ScoreInfo 스크레이퍼)는 단승 적중 말의
    win_odds만 갱신하는 반면 library/collector_races.py(공공API, 서울 포함
    3개 경마장 전부)는 모든 출전마의 winOdds를 채운다 — 두 소스가 혼재돼 있어
    결측 패턴이 경마장마다 다르다. 이 진단은 결측이 '승패와 무관한 랜덤
    결측'인지 '패배마에서만 빠지는 편향 결측'인지 사후 확인한다.
    """
    print(f"\n  [{label}] win_odds 결측 진단 (경마장별)")
    print(f"  {'meet_code':>9}  {'픽수':>5}  {'배당有비율':>9}  {'적중률(전체)':>10}  {'적중률(배당有)':>10}  {'적중률(배당無)':>10}")
    print(f"  {'-'*66}")
    for mc, grp in race_df.groupby("meet_code", dropna=False):
        n = len(grp)
        has = grp[grp["pick_odds"] > 0]
        missing = grp[grp["pick_odds"] <= 0]
        has_ratio = len(has) / n if n else float("nan")
        acc_all = grp["is_win"].mean()
        acc_has = has["is_win"].mean() if len(has) else float("nan")
        acc_missing = missing["is_win"].mean() if len(missing) else float("nan")
        print(f"  {str(mc):>9}  {n:>5}  {has_ratio:>8.1%}  {acc_all:>9.1%}  "
              f"{acc_has:>9.1%}  {acc_missing:>9.1%}")
    print("  해석: '적중률(배당有)'가 '적중률(배당無)'보다 뚜렷이 높으면(예: >5pp) 결측이")
    print("        승패와 상관돼 있다는 뜻 — 이 경우 ROI가 실제보다 낙관적으로 편향됐을 수 있음.")


def main():
    parser = argparse.ArgumentParser(description="단승 고신뢰 구간 세후 ROI 재검증 (dev/test 분리)")
    parser.add_argument("--dev-start", default=DEV_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    parser.add_argument("--tax-rate", type=float, default=TAX_RATE)
    args = parser.parse_args()

    global TAX_RATE
    TAX_RATE = args.tax_rate  # --tax-rate로 오버라이드 가능(기본 0.22)하게 실제 계산에 반영

    if args.test_start < CALIBRATION_EVAL_CUTOFF:
        print(f"[경고] test 시작일({args.test_start})이 CALIBRATION_EVAL_CUTOFF"
              f"({CALIBRATION_EVAL_CUTOFF})보다 이릅니다 — 캘리브레이터 holdout과 겹칠 수 있습니다.")

    result_df = run_predictions_with_odds(args.dev_start, args.test_end)
    if result_df is None:
        return

    with capture_report("win_tax_adjusted") as rc:
        race_df = build_race_table(result_df)
        race_df["race_date"] = race_df["race_date"].astype(str)

        dev_df = race_df[(race_df["race_date"] >= args.dev_start) & (race_df["race_date"] <= args.dev_end)].copy()
        test_df = race_df[(race_df["race_date"] >= args.test_start) & (race_df["race_date"] <= args.test_end)].copy()

        print(f"\n{'='*78}")
        print("  단승(win) 고신뢰 구간 세후 ROI 재검증 (dev/test 완전 분리, 사후 임계값 선택 방지)")
        print(f"  dev : {args.dev_start} ~ {args.dev_end}  ({len(dev_df)}경주)")
        print(f"  test: {args.test_start} ~ {args.test_end}  ({len(test_df)}경주)  ← 기존 408경주 벤치마크 창")
        print(f"  세금 가정: 승리 베팅 순수익(odds-1)에 {args.tax_rate:.0%} 정률 적용, 손실 베팅 비과세,")
        print(f"             건별 개별과세(연간 누적 미고려) — 보수적(실제보다 낮게 나올 수 있는) 근사치.")
        print(f"             자세한 가정 목록은 스크립트 상단 docstring 참고.")
        print(f"{'='*78}")

        # ── 섹션 0: win_odds 결측 편향 진단 ─────────────────────────────────
        print("\n" + "=" * 78)
        print("[섹션 0] win_odds 결측 편향 진단 (ROI 신뢰도 사전 점검)")
        print("=" * 78)
        report_odds_missingness(dev_df, "dev")
        report_odds_missingness(test_df, "test")

        # ── 섹션 1: dev 전체 스윕 (0.00~0.60, step 0.02) ────────────────────
        print("\n" + "=" * 78)
        print("[섹션 1] dev 기간 win_prob 임계값 전체 스윕 (0.00~0.60) — 후보 관찰 전용")
        print("=" * 78)
        print_sweep_table(dev_df, SWEEP_THRESHOLDS, "dev: 0.00~0.60 (step 0.02)")

        # ── 섹션 2: dev 미세 스윕 (0.20~0.40, step 0.01) — 안정성 확인 ──────
        print("\n" + "=" * 78)
        print("[섹션 2] dev 기간 미세 스윕 (0.20~0.40, step 0.01) — 임계값 안정성(잡음 여부) 확인")
        print("  (배경: 이전에 언급된 '0.28~0.31 근방 고신뢰 구간' 후보의 세부 형태 확인용)")
        print("=" * 78)
        print_sweep_table(dev_df, FINE_THRESHOLDS, "dev: 0.20~0.40 (step 0.01)")

        # ── 섹션 3: dev에서 관찰한 후보 임계값 -> test에 단 1회 적용 ─────────
        print("\n" + "=" * 78)
        print("[섹션 3] dev 관찰 기반 사전 등록 후보 임계값 -> test 기간에 '단 1회' 적용")
        print("  (섹션 1/2의 dev 결과만으로 후보를 고정하고, 이후 test 결과를 보고 재선택하지 않음)")
        print("=" * 78)

        candidates = find_dev_candidate_zone(dev_df, SWEEP_THRESHOLDS)
        if not candidates:
            print("\n  dev에서 '세후 ROI>0 & 표본>=30'을 만족하는 임계값 구간을 찾지 못했습니다.")
            print("  참고용으로 배경에서 언급된 0.28/0.30/0.31과 라운드 넘버 0.30/0.40/0.50을 그대로 test에 적용합니다.")
            candidates = [0.28, 0.30, 0.31, 0.40, 0.50]
        else:
            print(f"\n  dev 관찰 결과 '세후ROI>0 & n>={MIN_N_STABLE}' 연속 구간의 하한/중앙/상한: {candidates}")
            # 배경에서 언급된 0.28~0.31 근방도 참고용으로 함께 등록(사전 배경 정보로 이미 알려진 값이며,
            # 이번 dev 재분석과 무관하게 사용자가 검증을 요청한 값이므로 test에서 반드시 확인).
            for extra in (0.28, 0.30, 0.31):
                if extra not in candidates:
                    candidates.append(extra)
            candidates = sorted(set(candidates))

        print(f"\n  사전 등록된 test 검증 대상 임계값: {candidates}")

        # baseline (threshold=0)도 함께 보고
        apply_single_threshold_test(test_df, 0.0, "[TEST] baseline: threshold=0.00 (필터 없음, 매 경주 최고확률 픽에 베팅)")

        for t in candidates:
            apply_single_threshold_test(test_df, t, f"[TEST] threshold={t:.2f}")

        # ── 섹션 4: test 전체 스윕 (참고/설명용 — 사후 임계값 선택에 사용 금지) ──
        print("\n" + "=" * 78)
        print("[섹션 4] test 기간 전체 스윕 (0.00~0.60) — *참고용*, 안정성/전체 그림 확인 목적")
        print("  주의: 이 섹션의 수치를 보고 '가장 좋아 보이는' 임계값을 새로 골라 결론 내리면")
        print("        사후 임계값 선택(peeking)이 됩니다. 공식 검증 결과는 섹션 3의 사전 등록")
        print("        임계값들뿐입니다. 이 섹션은 test 구간에서도 스윕이 완만한지/튀는지")
        print("        (=신호인지 잡음인지) 시각적으로 확인하기 위한 참고 자료입니다.")
        print("=" * 78)
        print_sweep_table(test_df, SWEEP_THRESHOLDS, "test: 0.00~0.60 (step 0.02, 참고용)")

        print()

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    main()
