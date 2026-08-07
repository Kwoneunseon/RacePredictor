"""
확률 캘리브레이션(isotonic) 진단 스크립트.

목표: core/trainer.py의 win_calibrator·top3_calibrator가 실제로 확률의 신뢰도를
      개선하는지(raw vs calibrated) reliability diagram·Brier score·ECE로 정량화하고,
      TOP3_PROB_THRESHOLD(0.65) 임계값 게이트에 미치는 영향을 함께 확인한다.

원칙:
- CALIBRATION_EVAL_CUTOFF(core/trainer.py, 기본 "2026-05-01") 이후 데이터만 평가에 쓴다
  (그 이전 데이터는 calibrator 학습(holdout)에 이미 쓰였을 수 있어 오염 위험).
- isotonic regression은 단조 함수이므로 경주 내 "어느 말이 1픽인가"는 raw든 calibrated든
  이론상 동일해야 한다 — 이 스크립트가 실제로 그런지도 검증한다(순위 보존 확인,
  project_top3_no_blend 원칙과 충돌하지 않는지 확인하는 목적).
- 이 스크립트는 core/trainer.py·core/predictor.py를 import만 하고 수정하지 않는다
  (병렬 세션과의 파일 충돌 방지).

실행:
    python scripts/diagnose_calibration.py
    python scripts/diagnose_calibration.py --start 2026-05-01 --end 2026-07-31
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer, CALIBRATION_EVAL_CUTOFF
from core.model_manager import ModelManager
from core.feature_builder import RaceFeatureBuilder
from scripts._backtest_common import load_historical_data, capture_report
from config import TOP3_PROB_THRESHOLD


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """등폭(equal-width) n_bins개 구간의 ECE와 구간별 상세 정보를 반환."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bin_edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}", "count": 0,
                         "mean_pred": None, "empirical": None, "gap": None})
            continue
        mean_pred = float(p[mask].mean())
        empirical = float(y[mask].mean())
        gap = abs(mean_pred - empirical)
        ece += (cnt / n) * gap
        rows.append({"bin": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}", "count": cnt,
                     "mean_pred": mean_pred, "empirical": empirical, "gap": gap})
    return ece, rows


def print_reliability_table(title: str, p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    ece, rows = expected_calibration_error(p, y, n_bins)
    brier = brier_score(p, y)
    print(f"\n[ {title} ]  Brier={brier:.4f}  ECE={ece:.4f}  n={len(p)}")
    print(f"  {'구간':>10}  {'표본수':>6}  {'평균예측':>8}  {'실제빈도':>8}  {'|gap|':>7}")
    print(f"  {'-'*48}")
    for r in rows:
        if r["count"] == 0:
            print(f"  {r['bin']:>10}  {0:>6}  {'-':>8}  {'-':>8}  {'-':>7}")
            continue
        print(f"  {r['bin']:>10}  {r['count']:>6}  {r['mean_pred']:>8.3f}  {r['empirical']:>8.3f}  {r['gap']:>7.3f}")
    return brier, ece


def threshold_gate_report(df: pd.DataFrame, prob_col: str, label: str, threshold: float = TOP3_PROB_THRESHOLD):
    """경주별 top3_prob 최고 말 기준, threshold 이상 통과 경주의 적중률/커버리지."""
    def best_pick(grp):
        best = grp.loc[grp[prob_col].idxmax()]
        return pd.Series({"prob": best[prob_col], "is_top3": best["is_top3_actual"]})

    race_results = df.groupby("race_id", sort=False)[list(df.columns)].apply(best_pick).reset_index()
    total = len(race_results)
    sub = race_results[race_results["prob"] >= threshold]
    acc = sub["is_top3"].mean() if len(sub) else float("nan")
    if len(sub):
        print(f"  [{label}] threshold>={threshold}: {len(sub)}/{total}경주 통과 "
              f"({len(sub)/total:.1%}), 적중률={acc:.1%}")
    else:
        print(f"  [{label}] threshold>={threshold}: 0/{total}경주 통과")
    return race_results


def threshold_sweep(df: pd.DataFrame, prob_col: str, label: str,
                     thresholds=(0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)):
    """raw/calibrated 각각의 임계값별 적중률·커버리지 표 (scripts/backtest_top3.py와 동일 방식)."""
    def best_pick(grp):
        best = grp.loc[grp[prob_col].idxmax()]
        return pd.Series({"prob": best[prob_col], "is_top3": best["is_top3_actual"]})

    race_results = df.groupby("race_id", sort=False)[list(df.columns)].apply(best_pick).reset_index()
    total = len(race_results)
    print(f"\n  [{label}] 임계값별 적중률/커버리지 (전체 {total}경주)")
    print(f"    {'임계값':>6}  {'경주수':>6}  {'적중률':>7}  {'커버리지':>8}")
    for t in thresholds:
        sub = race_results[race_results["prob"] >= t]
        if len(sub) == 0:
            print(f"    {t:>6.2f}  {0:>6}  {'-':>7}  {'0.0%':>8}")
            continue
        acc = sub["is_top3"].mean()
        print(f"    {t:>6.2f}  {len(sub):>6}  {acc:>7.1%}  {len(sub)/total:>8.1%}")


def rank_preservation_check(df: pd.DataFrame, raw_col: str, cal_col: str) -> tuple[int, int]:
    """경주별로 raw 기준 1픽과 calibrated 기준 1픽이 같은 말인지 확인."""
    same, total = 0, 0
    for _, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        raw_pick = grp.loc[grp[raw_col].idxmax(), "horse_id"] if "horse_id" in grp.columns else grp[raw_col].idxmax()
        cal_pick = grp.loc[grp[cal_col].idxmax(), "horse_id"] if "horse_id" in grp.columns else grp[cal_col].idxmax()
        total += 1
        if raw_pick == cal_pick:
            same += 1
    return same, total


def run_diagnosis(start_date: str, end_date: str):
    print("모델 로드 중...")
    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor):
        print("[오류] 저장된 모델 없음. python main.py --mode train 먼저 실행하세요.")
        return

    if trainer.win_calibrator is None and trainer.top3_calibrator is None:
        print("[경고] 로드된 모델에 win_calibrator/top3_calibrator가 없습니다 "
              "(구버전 모델 pkl). 재학습 후 다시 실행하세요.")
        return

    if start_date < CALIBRATION_EVAL_CUTOFF:
        print(f"[경고] --start({start_date})가 CALIBRATION_EVAL_CUTOFF({CALIBRATION_EVAL_CUTOFF})보다 "
              f"이릅니다 - calibrator 학습에 쓰인 데이터가 평가에 섞여 지표가 낙관적으로 오염될 수 있습니다.")

    print(f"과거 데이터 로드 중: {start_date} ~ {end_date} ...")
    df_raw = load_historical_data(start_date, end_date)
    if df_raw.empty:
        print("[오류] 데이터 없음")
        return
    print(f"  로드: {len(df_raw)}건, {df_raw['race_id'].nunique()}경주")

    print("피처 빌드 중...")
    fb = RaceFeatureBuilder()
    df_feat = fb.build_for_training(df_raw, end_date)
    df_feat = df_feat.dropna(subset=["horse_age", "race_distance"])

    print("전처리·예측 중...")
    df_proc = preprocessor.transform(df_feat)
    features = trainer.feature_columns
    available = [c for c in features if c in df_proc.columns]
    X = df_proc[available].fillna(0)

    raw_win = trainer.predict_proba(X, raw=True)
    cal_win = trainer.predict_proba(X, raw=False)
    raw_top3 = trainer.predict_proba_top3(X, raw=True)
    cal_top3 = trainer.predict_proba_top3(X, raw=False)

    y_win = (df_feat["final_rank"] == 1).astype(int).values
    y_top3 = (df_feat["final_rank"] <= 3).astype(int).values

    with capture_report("calib_diag") as rc:
        print(f"\n{'='*60}")
        print(f"  확률 캘리브레이션 진단: {start_date} ~ {end_date}")
        print(f"{'='*60}")

        print("\n" + "#"*60)
        print("# 단승(win) 확률")
        print("#"*60)
        print_reliability_table("win raw (uncalibrated)", raw_win, y_win)
        print_reliability_table("win calibrated (isotonic)", cal_win, y_win)

        if raw_top3 is not None and cal_top3 is not None:
            print("\n" + "#"*60)
            print("# 연승(top3) 확률")
            print("#"*60)
            print_reliability_table("top3 raw (uncalibrated)", raw_top3, y_top3)
            print_reliability_table("top3 calibrated (isotonic)", cal_top3, y_top3)

            eval_df = pd.DataFrame({
                "race_id": df_feat["race_id"].values,
                "horse_id": df_feat["horse_id"].values if "horse_id" in df_feat.columns else np.arange(len(df_feat)),
                "raw_top3": raw_top3,
                "cal_top3": cal_top3,
                "is_top3_actual": y_top3,
            })

            print("\n[ TOP3_PROB_THRESHOLD 게이트 영향 (raw vs calibrated) ]")
            threshold_gate_report(eval_df, "raw_top3", "raw")
            threshold_gate_report(eval_df, "cal_top3", "calibrated")
            threshold_sweep(eval_df, "raw_top3", "raw")
            threshold_sweep(eval_df, "cal_top3", "calibrated")

            same, total = rank_preservation_check(eval_df, "raw_top3", "cal_top3")
            if total:
                print(f"\n[ 순위 보존 확인 ] 경주별 1픽(raw) == 1픽(calibrated): {same}/{total} ({same/total:.1%})")
            else:
                print("\n[ 순위 보존 확인 ] 대상 경주 없음")
            if same != total:
                print("  [주의] isotonic 보정 후 동률(tie)이 늘어나 1픽이 달라진 경주가 있습니다 - 원인 확인 필요.")
        else:
            print("\n[경고] top3_models 없음 - top3 진단 생략")

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="확률 캘리브레이션(isotonic) 진단")
    parser.add_argument("--start", default=CALIBRATION_EVAL_CUTOFF)
    parser.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    run_diagnosis(args.start, args.end)
