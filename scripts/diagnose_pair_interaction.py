"""
복연승(2마리 모두 top3) 페어 상호작용 신호 스크리닝.

"두 말이 동시에 top3에 들 확률"이 독립가정(top3_prob_i × top3_prob_j)만으로
설명되는지, 초반 페이스 경쟁 관련 상호작용(같은 선행형 말들이 초반에 서로
견제하는 효과)으로 설명되는 잔차가 남는지 확인한다.

Harville 조합확률([[project_multi_bet_probability_model]])과 기존 임시
휴리스틱은 둘 다 이미 검증된 개별 말 확률(win_prob/top3_prob)을 재조합만
한 것이라 원재료 이상의 정보를 못 만든다. 이 스크립트는 대신 "두 말이
동시에 top3에 들었는가"를 직접 라벨로 삼아, 독립가정 위에 상호작용
보정항만 얹는 오프셋 로지스틱 회귀로 새 정보가 있는지 스크리닝한다
(docs/superpowers/specs/2026-09-27-pair-interaction-screening-design.md 참고).

새 의존성을 추가하지 않는다 - statsmodels 대신 scipy.optimize로 직접
적합하고, 이 프로젝트가 일관되게 써온 경주 단위 클러스터 부트스트랩으로
계수 유의성을 검정한다.

실행:
    python scripts/diagnose_pair_interaction.py
    python scripts/diagnose_pair_interaction.py --n-boot 500
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Utils.ranks import valid_rank_mask
from scripts._backtest_common import load_historical_data, capture_report

EARLY_LEADER_THRESHOLD = 3.0

LOAD_START_DEFAULT = "2025-01-01"
DEV_END_DEFAULT = "2026-04-30"
TEST_START_DEFAULT, TEST_END_DEFAULT = "2026-05-01", "2026-06-25"


def fetch_avg_early_ord_asof(df: pd.DataFrame) -> pd.Series:
    """말별로 해당 경주 날짜 이전 기록만 사용해 초반 통과순위(1코너, 없으면
    3코너로 대체) 원값의 평균을 leak-safe하게 계산해 df와 같은 인덱스의
    Series로 반환한다. df는 최소 horse_id, race_date 컬럼이 있어야 한다.
    과거 기록이 없는 말이나 데이터 자체가 없으면 NaN.

    core.feature_builder.RaceFeatureBuilder._fetch_running_style_for_training과
    동일한 as-of 방식(해당 경주 이전 기록만 사용)이지만, 델타
    (초반순위-최종순위)가 아니라 초반순위 원값의 평균을 낸다.
    core/feature_builder.py는 수정하지 않고 이 스크립트 안에 독립적으로
    재구현한다(공유 파일 미커밋 변경과의 충돌을 피하기 위함).
    """
    from Utils.database import get_client
    from core.feature_builder import _fetch_in_batches

    if "horse_id" not in df.columns or "race_date" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    client = get_client()
    horse_ids = df["horse_id"].dropna().unique().tolist()
    if not horse_ids:
        return pd.Series(np.nan, index=df.index)

    detail_rows = _fetch_in_batches(
        client, "race_entry_detail", "horse_id, race_id, sj_1c_ord, sj_3c_ord",
        "horse_id", horse_ids,
    )
    if not detail_rows:
        return pd.Series(np.nan, index=df.index)

    detail_df = pd.DataFrame(detail_rows)
    race_ids = detail_df["race_id"].unique().tolist()
    rank_rows = _fetch_in_batches(
        client, "race_entries", "race_id, horse_id, race_date, final_rank",
        "race_id", race_ids,
    )
    if not rank_rows:
        return pd.Series(np.nan, index=df.index)

    rank_df = pd.DataFrame(rank_rows)
    merged = detail_df.merge(rank_df, on=["race_id", "horse_id"], how="inner")
    merged = merged[valid_rank_mask(merged["final_rank"])].copy()
    merged["_early_ord"] = pd.to_numeric(
        merged["sj_1c_ord"].fillna(merged["sj_3c_ord"]), errors="coerce"
    )
    merged = merged.dropna(subset=["_early_ord"])
    merged["_race_dt"] = pd.to_datetime(merged["race_date"])

    df = df.copy()
    df["_race_dt"] = pd.to_datetime(df["race_date"])
    out = pd.Series(np.nan, index=df.index)
    for hid, grp in df.groupby("horse_id"):
        past = merged[merged["horse_id"] == hid]
        if past.empty:
            continue
        for idx, row in grp.iterrows():
            past_before = past[past["_race_dt"] < row["_race_dt"]]
            if past_before.empty:
                continue
            out.loc[idx] = past_before["_early_ord"].mean()
    return out


def build_pair_table(df: pd.DataFrame, avg_early_ord: pd.Series,
                      early_leader_threshold: float = EARLY_LEADER_THRESHOLD) -> pd.DataFrame:
    """경주별로 유효 착순 말(valid_rank_mask)만 대상으로 모든 쌍(i,j)을 만들어
    독립가정 오프셋(p_indep)과 상호작용 피처, 라벨을 계산한다.

    df는 최소 race_id, race_date, entry_number, top3_prob, final_rank 컬럼이
    있어야 하고, avg_early_ord는 df와 같은 인덱스를 가진 Series(과거 평균
    초반통과순위, 과거 기록 없으면 NaN)여야 한다.

    반환 컬럼: race_id, race_date, entry_i, entry_j, p_indep,
    both_early_leader, entry_gap, pace_pressure, label.

    출전마 3두 미만(유효 착순 기준)인 경주는 페어가 생성되지 않는다
    (자동 스킵, 예외 없음). NaN인 avg_early_ord는 "선행형 아님"으로 취급
    (both_early_leader 비교에서 NaN <= threshold는 False가 되므로 별도
    처리 없이 자연히 반영됨).
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
                rows.append({
                    "race_id": race_id,
                    "race_date": ra["race_date"],
                    "entry_i": ra["entry_number"],
                    "entry_j": rb["entry_number"],
                    "p_indep": float(ra["top3_prob"]) * float(rb["top3_prob"]),
                    "both_early_leader": bool(ra["_early_leader"] and rb["_early_leader"]),
                    "entry_gap": abs(float(ra["entry_number"]) - float(rb["entry_number"])),
                    "pace_pressure": pace_pressure,
                    "label": int(ra["final_rank"] <= 3 and rb["final_rank"] <= 3),
                })
    return pd.DataFrame(rows)


def fit_offset_logistic(pair_df: pd.DataFrame) -> np.ndarray:
    """오프셋(logit(p_indep)) 위에 상호작용 보정항 b0~b3을 음의 로그우도
    최소화로 적합한다. pair_df는 p_indep, both_early_leader, entry_gap,
    pace_pressure, label 컬럼이 있어야 한다.

    통계 라이브러리(statsmodels) 없이 scipy.optimize로 직접 적합한다
    (이 프로젝트에 statsmodels가 설치돼 있지 않고, 계수 유의성은 별도로
    bootstrap_coefficients의 경주 단위 클러스터 부트스트랩으로 검정하므로
    이 함수 자체는 점추정만 하면 된다).

    반환: [b0, b1, b2, b3] (절편, both_early_leader 계수, entry_gap 계수,
    pace_pressure 계수)
    """
    p_indep = pair_df["p_indep"].to_numpy().clip(1e-6, 1 - 1e-6)
    offset = np.log(p_indep / (1 - p_indep))
    X = np.column_stack([
        pair_df["both_early_leader"].astype(float).to_numpy(),
        pair_df["entry_gap"].to_numpy(),
        pair_df["pace_pressure"].to_numpy(),
    ])
    y = pair_df["label"].to_numpy().astype(float)

    def neg_log_lik(params: np.ndarray) -> float:
        b0 = params[0]
        b = params[1:]
        eta = np.clip(offset + b0 + X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        eps = 1e-9
        return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    result = minimize(neg_log_lik, x0=np.zeros(4), method="BFGS")
    return result.x


def bootstrap_coefficients(pair_df: pd.DataFrame, n_boot: int = 1000,
                            seed: int = 42, coef_clip: float = 20.0) -> dict:
    """경주 단위(race_id)로 재표본추출하는 부트스트랩으로 b0~b3의 95%
    신뢰구간을 낸다(이 프로젝트가 다른 백테스트들에서 일관되게 써온 경주
    단위 클러스터 부트스트랩 방식 - 한 경주 안의 여러 페어 행은 서로
    강하게 종속돼 있어 행 단위 부트스트랩은 안 된다).

    재표본 안에서 상호작용 피처(특히 both_early_leader)가 라벨을 거의
    완전분리하면 로지스틱 적합이 scipy상 "성공"으로 보고되면서도 계수가
    발산한다(코드 리뷰에서 실측: b1=41.3 같은 값이 나옴 - eta가 클리핑
    구간에 들어가면 그 지점의 기울기가 0이 되어 BFGS가 수렴으로 오판함,
    result.success 체크로도 못 잡음). 이를 방지하기 위해 각 재표본의
    적합 결과를 ±coef_clip으로 클리핑하고, 클리핑된 재표본 비율이 5%를
    넘으면 경고를 출력한다(신뢰구간이 발산값에 왜곡되지 않게 함 +
    투명성 확보).

    반환: {"b0": (low, high), "b1": (low, high), "b2": (low, high), "b3": (low, high)}
    """
    rng = np.random.default_rng(seed)
    groups = {rid: g for rid, g in pair_df.groupby("race_id", sort=False)}
    race_ids = np.array(list(groups.keys()))

    samples = []
    n_clipped = 0
    for _ in range(n_boot):
        chosen = rng.choice(race_ids, size=len(race_ids), replace=True)
        resampled = pd.concat([groups[rid] for rid in chosen], ignore_index=True)
        coefs = fit_offset_logistic(resampled)
        clipped = np.clip(coefs, -coef_clip, coef_clip)
        if not np.allclose(clipped, coefs):
            n_clipped += 1
        samples.append(clipped)
    arr = np.array(samples)

    if n_clipped / n_boot > 0.05:
        print(f"  [주의] 부트스트랩 재표본 중 {n_clipped}/{n_boot}건({n_clipped/n_boot:.1%})이 "
              f"발산(|계수|>{coef_clip})해서 클리핑됨 - 신뢰구간을 주의해서 해석할 것")

    names = ["b0", "b1", "b2", "b3"]
    return {
        name: (float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5)))
        for i, name in enumerate(names)
    }


if __name__ == "__main__":
    pass
