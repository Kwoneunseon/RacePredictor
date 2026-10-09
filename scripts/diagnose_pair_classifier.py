"""
복연승(2마리 모두 top3) 페어 전용 이진분류기.

2026-10-09 페어 상호작용 스크리닝(scripts/diagnose_pair_interaction.py,
오프셋 로지스틱)에서 "두 말이 동시에 top3에 들 확률"이 독립가정
(top3_prob_i × top3_prob_j)만으로 설명되지 않고, 초반 페이스 상호작용으로
설명되는 잔차가 통계적으로 유의미하게 남는다는 결과가 나왔다(b1/b2/b3
전부 95% CI가 0 배제, test 최고 페어 픽 적중률 40.0%, 408경주).

이 스크립트는 그 결과를 더 유연한 모델(XGBoost)로 얼마나 더 뽑아낼 수
있는지 확인한다. 오프셋 로지스틱은 보정항을 "더하기만" 하는 선형
구조였지만, 이번엔 경주 내 모든 유효 말 쌍을 한 행으로 재구성해 "둘 다
top3"를 직접 라벨로 학습시킨다
(docs/superpowers/specs/2026-10-09-pair-classifier-design.md 참고).

튜닝(Optuna)·앙상블·프로덕션 연결은 이번 범위에 없다.

실행:
    python scripts/diagnose_pair_classifier.py
    python scripts/diagnose_pair_classifier.py --dev-end 2026-04-30
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss

from Utils.ranks import valid_rank_mask
from scripts._backtest_common import load_historical_data, capture_report
from scripts.diagnose_pair_interaction import fetch_avg_early_ord_asof

EARLY_LEADER_THRESHOLD = 3.0

LOAD_START_DEFAULT = "2025-01-01"
DEV_END_DEFAULT = "2026-04-30"
TEST_START_DEFAULT, TEST_END_DEFAULT = "2026-05-01", "2026-06-25"

FEATURE_COLUMNS = [
    "win_prob_i", "win_prob_j", "top3_prob_i", "top3_prob_j",
    "p_indep", "win_prob_diff", "win_prob_ratio",
    "both_early_leader", "entry_gap", "pace_pressure",
]


def build_classifier_pair_table(df: pd.DataFrame, avg_early_ord: pd.Series,
                                 early_leader_threshold: float = EARLY_LEADER_THRESHOLD) -> pd.DataFrame:
    """경주별로 유효 착순 말(valid_rank_mask)만 대상으로 모든 쌍(i,j)을 만들어
    분류기 학습용 피처와 라벨을 계산한다.

    df는 최소 race_id, race_date, entry_number, win_prob, top3_prob,
    final_rank 컬럼이 있어야 하고, avg_early_ord는 df와 같은 인덱스를 가진
    Series(과거 평균 초반통과순위, 과거 기록 없으면 NaN)여야 한다.

    반환 컬럼: race_id, race_date, entry_i, entry_j, win_prob_i, win_prob_j,
    top3_prob_i, top3_prob_j, p_indep, win_prob_diff, win_prob_ratio,
    both_early_leader, entry_gap, pace_pressure, label.

    출전마 3두 미만(유효 착순 기준)인 경주는 페어가 생성되지 않는다
    (자동 스킵, 예외 없음). win_prob_ratio는 i/j 순서에 무관하게 대칭이다
    (min/max 비율, 0으로 나누기는 분모를 1e-6으로 바닥 처리해 방지).
    NaN인 avg_early_ord는 "선행형 아님"으로 취급한다.
    """
    df = df.copy()
    df["_valid"] = valid_rank_mask(df["final_rank"])
    df["_early_leader"] = (avg_early_ord.reindex(df.index) <= early_leader_threshold).fillna(False)

    rows = []
    for race_id, grp in df[df["_valid"]].groupby("race_id", sort=False):
        n = len(grp)
        if n < 3:
            continue
        pace_pressure = int(grp["_early_leader"].sum())
        grp = grp.reset_index(drop=True)
        for a in range(n):
            for b in range(a + 1, n):
                ra, rb = grp.iloc[a], grp.iloc[b]
                wp_i, wp_j = float(ra["win_prob"]), float(rb["win_prob"])
                tp_i, tp_j = float(ra["top3_prob"]), float(rb["top3_prob"])
                lo, hi = (wp_i, wp_j) if wp_i <= wp_j else (wp_j, wp_i)
                rows.append({
                    "race_id": race_id,
                    "race_date": ra["race_date"],
                    "entry_i": ra["entry_number"],
                    "entry_j": rb["entry_number"],
                    "win_prob_i": wp_i,
                    "win_prob_j": wp_j,
                    "top3_prob_i": tp_i,
                    "top3_prob_j": tp_j,
                    "p_indep": tp_i * tp_j,
                    "win_prob_diff": abs(wp_i - wp_j),
                    "win_prob_ratio": lo / max(hi, 1e-6),
                    "both_early_leader": bool(ra["_early_leader"] and rb["_early_leader"]),
                    "entry_gap": abs(float(ra["entry_number"]) - float(rb["entry_number"])),
                    "pace_pressure": pace_pressure,
                    "label": int(ra["final_rank"] <= 3 and rb["final_rank"] <= 3),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pass
