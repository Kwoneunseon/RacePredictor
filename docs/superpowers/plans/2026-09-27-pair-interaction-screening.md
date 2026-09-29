# 복연승 페어 상호작용 신호 스크리닝 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "두 말이 동시에 top3에 들 확률"이 독립가정(`top3_prob_i × top3_prob_j`)만으로 설명되는지, 초반 페이스 경쟁 상호작용으로 설명되는 잔차가 통계적으로 유의미하게 남는지 확인하는 진단 스크립트를 만들고 실제로 돌려서 결론을 낸다.

**Architecture:** 새 스크립트 `scripts/diagnose_pair_interaction.py` 하나에 (1) 말별 leak-safe 과거 평균 초반통과순위 계산, (2) 경주별 유효 착순 페어 테이블 생성(독립가정 오프셋 + 상호작용 피처 + 라벨), (3) 오프셋 로지스틱 회귀를 `scipy.optimize`로 직접 적합, (4) 경주 단위 클러스터 부트스트랩으로 계수 신뢰구간 산출을 전부 담는다. 순수 계산 함수(페어 테이블 생성, 적합, 부트스트랩)는 TDD로, DB/모델에 의존하는 함수는 기존 `scripts/backtest_*.py` 관례대로 스모크 테스트로 검증한다.

**Tech Stack:** Python 3.11, pandas, numpy, scipy(`scipy.optimize.minimize` — 새 의존성 아님, scikit-learn이 이미 의존하므로 설치돼 있음), pytest. `scripts/_backtest_common.py`의 `load_historical_data`/`capture_report`는 **읽기(import)만** 하고 수정하지 않는다. `core/feature_builder.py`도 수정하지 않는다.

**참고 문서:** `docs/superpowers/specs/2026-09-27-pair-interaction-screening-design.md`

---

### 사전 확인 사항 (구현자가 알아야 할 것)

- `core.rank_distribution`이나 `scripts/_backtest_common.py`의 `attach_harville_top3`/`build_harville_pair_table`을 재사용하지 않는다 — 이번 건 완전히 새로운 접근(오프셋 로지스틱)이다.
- `Utils/ranks.py::valid_rank_mask(series)`가 이미 있다(`1 <= final_rank < config.NON_FINISH_RANK_MIN` 판정, 커밋 5dca546). 그대로 import해서 써라.
- `core/feature_builder.py`의 module-level 함수 `_fetch_in_batches(client, table, select_cols, in_col, values, chunk_size=500, page_size=1000, filter_fn=None)`은 PostgREST의 1000행 응답 제한을 청크+페이지네이션으로 우회하는 필수 안전장치다(2026-07-20에 이 문제로 신규 피처 조인이 대부분 결측으로 빠진 사고가 있었음). **직접 `.in_()` 쿼리를 새로 짜지 말고 이 함수를 import해서 재사용해라**: `from core.feature_builder import _fetch_in_batches`.
- `race_entry_detail` 테이블 컬럼: `horse_id, race_id, sj_1c_ord, sj_3c_ord`(1코너/3코너 통과순위). `race_entries` 컬럼: `race_id, horse_id, race_date, final_rank, entry_number`.
- 이 프로젝트는 `scripts/*.py`(백테스트·진단 스크립트)는 단위테스트 대상이 아니고, DB/모델에 의존하는 함수는 대신 "실제 DB로 스모크 테스트"로 검증하는 관례를 따른다. 순수 계산 함수(입력 DataFrame/array → 출력, DB 호출 없음)는 TDD로 테스트한다.
- 저장소에는 이 계획과 무관한 사용자의 미커밋 변경 17개 + 미추적 3개가 있다(`core/trainer.py`, `library/scraper_score.py`, `scripts/_backtest_common.py` 등). **이번 계획에서 만드는 파일(`scripts/diagnose_pair_interaction.py`, `tests/test_diagnose_pair_interaction.py`)은 전부 새 파일이라 그 미커밋 변경과 충돌하지 않는다** — 그래도 커밋 전에는 항상 `git status`/`git diff --cached --name-only`로 의도한 새 파일만 스테이지됐는지 확인해라(다른 파일을 stage하지 마라).
- 테스트 실행: `python -m pytest tests/test_diagnose_pair_interaction.py -v` (리포지토리 루트에서). 콘솔 한글이 깨지면 `python -X utf8`.

---

### Task 1: 스크립트 골격 + `fetch_avg_early_ord_asof` (DB 의존, leak-safe)

**Files:**
- Create: `scripts/diagnose_pair_interaction.py`

이 함수는 DB에 의존해 단위테스트 대상이 아니다(기존 `core/feature_builder.py`의 `_fetch_running_style_for_training`과 동일한 관례). 문법/임포트만 검증한다.

- [ ] **Step 1: 파일 생성**

```python
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
    재구현한다(공유 파일 미커밧 변경과의 충돌을 피하기 위함).
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


if __name__ == "__main__":
    pass
```

- [ ] **Step 2: 문법 오류 없는지, DB 접근 없이 안전하게 import되는지 확인**

Run: `python -c "import ast; ast.parse(open('scripts/diagnose_pair_interaction.py', encoding='utf-8').read())"`
Expected: 아무 출력 없이 종료(exit 0)

Run: `python -c "import scripts.diagnose_pair_interaction"`
Expected: 아무 출력 없이 종료(exit 0) — `fetch_avg_early_ord_asof`는 함수 정의만이라 import 시점에 DB 접근 없음

- [ ] **Step 3: 커밋**

```bash
git add scripts/diagnose_pair_interaction.py
git commit -m "feat: 복연승 페어 상호작용 스크리닝 - 골격 + leak-safe 초반순위 집계

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 2: `build_pair_table` (페어 테이블 생성, TDD)

**Files:**
- Modify: `scripts/diagnose_pair_interaction.py`
- Test: `tests/test_diagnose_pair_interaction.py` (신규)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_diagnose_pair_interaction.py` 새로 생성:

```python
import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_pair_interaction import build_pair_table


def _make_df():
    """R1: 4두 전부 유효 착순(1,2,3,4위). R2: 2두인데 1두는 비착순 코드(99)라
    유효 말이 1두뿐 -> 페어 불가."""
    return pd.DataFrame({
        "race_id":      ["R1"] * 4 + ["R2"] * 2,
        "race_date":    ["2026-01-01"] * 4 + ["2026-01-02"] * 2,
        "entry_number": [1, 2, 3, 4, 1, 2],
        "top3_prob":    [0.7, 0.6, 0.3, 0.2, 0.5, 0.5],
        "final_rank":   [1, 2, 3, 4, 1, 99],
    })


def test_build_pair_table_pair_count_matches_combinations():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    assert len(pairs[pairs["race_id"] == "R1"]) == 6  # C(4,2)
    assert len(pairs[pairs["race_id"] == "R2"]) == 0  # 유효 말 1두뿐


def test_build_pair_table_label_is_both_top3():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["label"] == 1  # 둘 다 rank<=3
    row14 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row14["label"] == 0  # entry4는 rank=4


def test_build_pair_table_p_indep_is_product_of_top3_prob():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["p_indep"] == pytest.approx(0.7 * 0.6)


def test_build_pair_table_both_early_leader_flag():
    df = _make_df()
    # entry1,2 -> 초반순위 평균 1.0(<=3, 선행형), entry3,4 -> 5.0(선행형 아님)
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    both_leader = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert both_leader["both_early_leader"] == True
    mixed = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 3)].iloc[0]
    assert mixed["both_early_leader"] == False
    neither = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 3) & (pairs.entry_j == 4)].iloc[0]
    assert neither["both_early_leader"] == False


def test_build_pair_table_pace_pressure_counts_early_leaders_in_race():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    r1_pairs = pairs[pairs.race_id == "R1"]
    assert (r1_pairs["pace_pressure"] == 2).all()  # entry1,2 두 마리가 선행형


def test_build_pair_table_nan_avg_early_ord_treated_as_not_leader():
    df = pd.DataFrame({
        "race_id":      ["R3"] * 3,
        "race_date":    ["2026-01-03"] * 3,
        "entry_number": [1, 2, 3],
        "top3_prob":    [0.5, 0.4, 0.3],
        "final_rank":   [1, 2, 3],
    })
    avg_early_ord = pd.Series([1.0, np.nan, 1.0], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["both_early_leader"] == False  # entry2는 기록 없음(NaN) -> 선행형 아님 취급
    row13 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 3)].iloc[0]
    assert row13["both_early_leader"] == True


def test_build_pair_table_entry_gap_is_absolute_difference():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan], index=df.index)
    pairs = build_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row["entry_gap"] == pytest.approx(3.0)
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 7개 테스트 모두 `ImportError`(`build_pair_table` 미정의)로 FAIL

- [ ] **Step 3: `build_pair_table` 구현**

`scripts/diagnose_pair_interaction.py`의 `fetch_avg_early_ord_asof` 함수 뒤,
`if __name__ == "__main__":` 블록 앞에 추가:

```python
def build_pair_table(df: pd.DataFrame, avg_early_ord: pd.Series,
                      early_leader_threshold: float = EARLY_LEADER_THRESHOLD) -> pd.DataFrame:
    """경주별로 유효 착순 말(valid_rank_mask)만 대상으로 모든 쌍(i,j)을 만들어
    독립가정 오프셋(p_indep)과 상호작용 피처, 라벨을 계산한다.

    df는 최소 race_id, race_date, entry_number, top3_prob, final_rank 컬럼이
    있어야 하고, avg_early_ord는 df와 같은 인덱스를 가진 Series(과거 평균
    초반통과순위, 과거 기록 없으면 NaN)여야 한다.

    반환 컬럼: race_id, race_date, entry_i, entry_j, p_indep,
    both_early_leader, entry_gap, pace_pressure, label.

    출전마 2두 미만(유효 착순 기준)인 경주는 페어가 생성되지 않는다
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
        if n < 2:
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 7개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/diagnose_pair_interaction.py tests/test_diagnose_pair_interaction.py
git commit -m "feat: 복연승 페어 상호작용 스크리닝 - build_pair_table 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 3: `fit_offset_logistic` (오프셋 로지스틱 회귀 적합, TDD)

**Files:**
- Modify: `scripts/diagnose_pair_interaction.py`
- Test: `tests/test_diagnose_pair_interaction.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_diagnose_pair_interaction.py` 끝에 추가:

```python
from scripts.diagnose_pair_interaction import fit_offset_logistic


def test_fit_offset_logistic_recovers_strong_signal():
    rng = np.random.default_rng(0)
    n = 5000
    p_indep = rng.uniform(0.05, 0.5, n)
    both_early_leader = rng.integers(0, 2, n).astype(float)
    entry_gap = rng.uniform(0, 10, n)
    pace_pressure = rng.integers(0, 6, n).astype(float)

    true_b0, true_b1, true_b2, true_b3 = 0.0, 1.5, 0.0, 0.0
    offset = np.log(p_indep / (1 - p_indep))
    eta = offset + true_b0 + true_b1 * both_early_leader + true_b2 * entry_gap + true_b3 * pace_pressure
    p_true = 1 / (1 + np.exp(-eta))
    label = (rng.uniform(0, 1, n) < p_true).astype(int)

    pair_df = pd.DataFrame({
        "p_indep": p_indep, "both_early_leader": both_early_leader,
        "entry_gap": entry_gap, "pace_pressure": pace_pressure, "label": label,
    })
    b0, b1, b2, b3 = fit_offset_logistic(pair_df)
    assert b1 > 0.8  # 참값 1.5, 잡음 감안 여유 있는 하한
    assert abs(b2) < 0.3
    assert abs(b3) < 0.3


def test_fit_offset_logistic_null_signal_recovers_near_zero():
    rng = np.random.default_rng(1)
    n = 5000
    p_indep = rng.uniform(0.05, 0.5, n)
    both_early_leader = rng.integers(0, 2, n).astype(float)
    entry_gap = rng.uniform(0, 10, n)
    pace_pressure = rng.integers(0, 6, n).astype(float)
    label = (rng.uniform(0, 1, n) < p_indep).astype(int)  # 순수 독립가정으로 생성(상호작용 없음)

    pair_df = pd.DataFrame({
        "p_indep": p_indep, "both_early_leader": both_early_leader,
        "entry_gap": entry_gap, "pace_pressure": pace_pressure, "label": label,
    })
    b0, b1, b2, b3 = fit_offset_logistic(pair_df)
    assert abs(b0) < 0.3
    assert abs(b1) < 0.3
    assert abs(b2) < 0.1
    assert abs(b3) < 0.1
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 위 2개 테스트가 `ImportError`(`fit_offset_logistic` 미정의)로 FAIL, 기존 7개는 PASS

- [ ] **Step 3: `fit_offset_logistic` 구현**

`scripts/diagnose_pair_interaction.py`의 `build_pair_table` 함수 뒤에 추가:

```python
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 9개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/diagnose_pair_interaction.py tests/test_diagnose_pair_interaction.py
git commit -m "feat: 복연승 페어 상호작용 스크리닝 - fit_offset_logistic 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 4: `bootstrap_coefficients` (경주 단위 클러스터 부트스트랩, TDD)

**Files:**
- Modify: `scripts/diagnose_pair_interaction.py`
- Test: `tests/test_diagnose_pair_interaction.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_diagnose_pair_interaction.py` 끝에 추가:

```python
from scripts.diagnose_pair_interaction import bootstrap_coefficients


def test_bootstrap_coefficients_returns_expected_structure():
    rows = []
    for i in range(30):
        race_id = f"R{i}"
        for j in range(3):
            rows.append({
                "race_id": race_id,
                "p_indep": 0.2,
                "both_early_leader": float(j == 0),
                "entry_gap": 1.0,
                "pace_pressure": 2.0,
                "label": 1 if j == 0 else 0,
            })
    pair_df = pd.DataFrame(rows)
    ci = bootstrap_coefficients(pair_df, n_boot=20, seed=3)
    assert set(ci.keys()) == {"b0", "b1", "b2", "b3"}
    for lo, hi in ci.values():
        assert lo <= hi


def test_bootstrap_coefficients_detects_strong_signal_excludes_zero():
    rows = []
    for i in range(200):
        race_id = f"R{i}"
        rows.append({"race_id": race_id, "p_indep": 0.2, "both_early_leader": 1.0,
                      "entry_gap": 1.0, "pace_pressure": 2.0, "label": 1})
        rows.append({"race_id": race_id, "p_indep": 0.2, "both_early_leader": 0.0,
                      "entry_gap": 1.0, "pace_pressure": 2.0, "label": 0})
    pair_df = pd.DataFrame(rows)
    ci = bootstrap_coefficients(pair_df, n_boot=100, seed=5)
    lo, hi = ci["b1"]
    assert lo > 0  # both_early_leader=1이면 항상 적중, =0이면 항상 실패 -> 신호가 0을 배제해야 함
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 위 2개 테스트가 `ImportError`(`bootstrap_coefficients` 미정의)로 FAIL, 기존 9개는 PASS

- [ ] **Step 3: `bootstrap_coefficients` 구현**

`scripts/diagnose_pair_interaction.py`의 `fit_offset_logistic` 함수 뒤에 추가:

```python
def bootstrap_coefficients(pair_df: pd.DataFrame, n_boot: int = 1000,
                            seed: int = 42) -> dict:
    """경주 단위(race_id)로 재표본추출하는 부트스트랩으로 b0~b3의 95%
    신뢰구간을 낸다(이 프로젝트가 다른 백테스트들에서 일관되게 써온 경주
    단위 클러스터 부트스트랩 방식 - 한 경주 안의 여러 페어 행은 서로
    강하게 종속돼 있어 행 단위 부트스트랩은 안 된다).

    반환: {"b0": (low, high), "b1": (low, high), "b2": (low, high), "b3": (low, high)}
    """
    rng = np.random.default_rng(seed)
    groups = {rid: g for rid, g in pair_df.groupby("race_id", sort=False)}
    race_ids = np.array(list(groups.keys()))

    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(race_ids, size=len(race_ids), replace=True)
        resampled = pd.concat([groups[rid] for rid in chosen], ignore_index=True)
        samples.append(fit_offset_logistic(resampled))
    arr = np.array(samples)

    names = ["b0", "b1", "b2", "b3"]
    return {
        name: (float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5)))
        for i, name in enumerate(names)
    }
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 11개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/diagnose_pair_interaction.py tests/test_diagnose_pair_interaction.py
git commit -m "feat: 복연승 페어 상호작용 스크리닝 - bootstrap_coefficients 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 5: `_load_predictions_with_ids` + `run_diagnosis` 조립 + 실DB 스모크 테스트

**Files:**
- Modify: `scripts/diagnose_pair_interaction.py`

이 태스크는 실제 DB와 저장된 모델이 필요하다(단위테스트 대상 아님). 조립 후
작은 날짜 범위로 실제로 돌려서 컬럼 보존 가정(특히 `horse_id`/
`entry_number`가 `build_for_training` 이후에도 살아있는지)을 검증한다.

- [ ] **Step 1: `_load_predictions_with_ids`와 `run_diagnosis` 추가**

`scripts/diagnose_pair_interaction.py`의 `EARLY_LEADER_THRESHOLD`/날짜 상수
정의 바로 뒤(즉 `fetch_avg_early_ord_asof` 함수 앞)에 추가:

```python
def _load_predictions_with_ids(start_date: str, end_date: str):
    """scripts/_backtest_common.py의 run_predictions()와 동일한 순서
    (모델 로드 -> load_historical_data -> build_for_training ->
    preprocessor.transform -> predict_proba_top3)를 따르되, horse_id와
    entry_number를 결과에 유지한다(run_predictions()는 이 두 컬럼을
    버린다). scripts/_backtest_common.py는 수정하지 않고 이 스크립트
    안에 독립적으로 재구현한다(공유 파일 미커밋 변경과의 충돌을 피함).

    반환 컬럼: race_id, race_date, horse_id, entry_number, final_rank,
    top3_prob. 모델/데이터가 없으면 None.
    """
    from core.preprocessor import RacePreprocessor
    from core.trainer import RaceModelTrainer
    from core.model_manager import ModelManager
    from core.feature_builder import RaceFeatureBuilder

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
    print(f"  로드: {len(df_raw)}건")

    print("피처 빌드 중...")
    fb = RaceFeatureBuilder()
    df_feat = fb.build_for_training(df_raw, end_date)
    df_feat = df_feat.dropna(subset=["horse_age", "race_distance"])

    for col in ("horse_id", "entry_number"):
        if col not in df_feat.columns:
            print(f"[오류] build_for_training 이후 '{col}' 컬럼이 사라짐 - "
                  f"이 함수의 전제가 깨졌으니 core/feature_builder.py의 "
                  f"파이프라인을 확인해야 함")
            return None

    print("전처리 중...")
    df_proc = preprocessor.transform(df_feat)

    print("예측 중...")
    features = trainer.feature_columns
    available = [c for c in features if c in df_proc.columns]
    X = df_proc[available].fillna(0)

    top3_probs = trainer.predict_proba_top3(X, raw=True)
    if top3_probs is None:
        print("[오류] top3_models 없음")
        return None

    result_df = pd.DataFrame({
        "race_id":      df_feat["race_id"].values,
        "race_date":    df_feat["race_date"].values,
        "horse_id":     df_feat["horse_id"].values,
        "entry_number": df_feat["entry_number"].values,
        "final_rank":   df_feat["final_rank"].values,
        "top3_prob":    top3_probs,
    }, index=df_feat.index)

    return result_df


def run_diagnosis(load_start: str, dev_end: str, test_start: str, test_end: str, n_boot: int):
    df = _load_predictions_with_ids(load_start, test_end)
    if df is None:
        return

    print("과거 평균 초반통과순위(leak-safe) 계산 중...")
    avg_early_ord = fetch_avg_early_ord_asof(df)

    pair_df = build_pair_table(df, avg_early_ord)
    if pair_df.empty:
        print("[오류] 페어를 생성할 수 있는 경주가 없습니다")
        return
    pair_df["race_date"] = pair_df["race_date"].astype(str)

    with capture_report("pair_interaction") as rc:
        dev_df = pair_df[pair_df["race_date"] <= dev_end]
        test_df = pair_df[(pair_df["race_date"] >= test_start) & (pair_df["race_date"] <= test_end)]

        print(f"\n{'='*70}")
        print(f"  복연승 페어 상호작용 스크리닝: dev<= {dev_end}, test {test_start}~{test_end}")
        print(f"{'='*70}")
        print(f"dev 페어 수: {len(dev_df)} ({dev_df['race_id'].nunique()}경주)")
        print(f"test 페어 수: {len(test_df)} ({test_df['race_id'].nunique()}경주)")

        if dev_df.empty or test_df.empty:
            print("[오류] dev 또는 test 구간에 페어가 없습니다")
            return

        coefs = fit_offset_logistic(dev_df)
        b0, b1, b2, b3 = coefs
        print(f"\n[dev 적합 계수] b0(절편)={b0:.4f}  b1(both_early_leader)={b1:.4f}  "
              f"b2(entry_gap)={b2:.4f}  b3(pace_pressure)={b3:.4f}")

        print(f"\n부트스트랩({n_boot}회, 경주 단위 재표본) 95% 신뢰구간 계산 중...")
        ci = bootstrap_coefficients(dev_df, n_boot=n_boot)
        label_map = {"b0": "절편", "b1": "both_early_leader", "b2": "entry_gap", "b3": "pace_pressure"}
        print("\n[신뢰구간]")
        for key, label in label_map.items():
            lo, hi = ci[key]
            sig = "유의함(0 배제)" if (lo > 0 or hi < 0) else "유의하지 않음(0 포함)"
            print(f"  {label}: [{lo:.4f}, {hi:.4f}]  - {sig}")

        p_indep_test = test_df["p_indep"].to_numpy().clip(1e-6, 1 - 1e-6)
        offset_test = np.log(p_indep_test / (1 - p_indep_test))
        eta_test = offset_test + b0 \
            + b1 * test_df["both_early_leader"].astype(float).to_numpy() \
            + b2 * test_df["entry_gap"].to_numpy() \
            + b3 * test_df["pace_pressure"].to_numpy()
        test_df = test_df.copy()
        test_df["p_joint"] = 1.0 / (1.0 + np.exp(-np.clip(eta_test, -30, 30)))

        best_idx = test_df.groupby("race_id")["p_joint"].idxmax()
        best = test_df.loc[best_idx]
        acc = best["label"].mean()
        print(f"\n[test 최고 페어 픽 적중률] {acc:.1%}  ({len(best)}경주)")
        print("  비교 기준(같은 창은 아닐 수 있음): 무작위 7.2%, 기존 휴리스틱 38.5%, Harville 36.5%")

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="복연승 페어 상호작용 신호 스크리닝")
    parser.add_argument("--load-start", default=LOAD_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    run_diagnosis(args.load_start, args.dev_end, args.test_start, args.test_end, args.n_boot)
```

기존의 `if __name__ == "__main__": pass` 자리표시자는 이 새 `if __name__ ==
"__main__":` 블록으로 완전히 대체된다(둘이 같이 있으면 안 됨 - 기존 것을
지워라).

- [ ] **Step 2: 전체 테스트 스위트가 여전히 통과하는지 확인 (DB 불필요)**

Run: `python -m pytest tests/test_diagnose_pair_interaction.py -v`
Expected: 11개 테스트 모두 PASS (이번 스텝은 새 코드를 추가했을 뿐 기존
테스트 대상 함수는 안 바꿨으므로 그대로 통과해야 함)

- [ ] **Step 3: 실제 DB로 작은 범위 스모크 테스트 — 컬럼 보존 가정 검증**

Run: `python -X utf8 -c "
from scripts.diagnose_pair_interaction import _load_predictions_with_ids
df = _load_predictions_with_ids('2026-06-01', '2026-06-07')
print('컬럼:', list(df.columns) if df is not None else None)
print('행수:', len(df) if df is not None else 0)
if df is not None:
    print('horse_id 결측:', df[\"horse_id\"].isna().sum())
    print('entry_number 결측:', df[\"entry_number\"].isna().sum())
"`

Expected: `horse_id`/`entry_number` 컬럼이 존재하고 결측이 0(또는 매우
적음)이어야 한다. **만약 `[오류] build_for_training 이후 'horse_id' 컬럼이
사라짐` 메시지가 나오면**, `core/feature_builder.py`의
`build_for_training` 내부 어딘가에서 이 컬럼이 드롭되고 있다는 뜻이다 —
그 경우 `core/feature_builder.py`를 **수정하지 말고**(공유 파일), 대신
이 함수에서 `df_raw`(원본, `load_historical_data` 직후) 쪽의 `horse_id`/
`entry_number`를 별도로 보관해두고 `race_id`+`race_date`+원래 인덱스로
나중에 다시 merge하는 방식으로 바꿔라. 이 경우는 코드 변경이 필요하므로
계속 진행하기 전에 실제로 어떤지 확인하고 필요하면 고쳐라.

- [ ] **Step 4: 커밋**

```bash
git add scripts/diagnose_pair_interaction.py
git commit -m "feat: 복연승 페어 상호작용 스크리닝 - 전체 파이프라인 조립(run_diagnosis)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 6: 실제 실행 및 판정

**이 태스크는 DB 접근이 가능한 환경에서 실행한다** (이 세션은 지금 DB
접근이 가능함을 확인했다).

**Files:** 없음(실행 전용 태스크)

- [ ] **Step 1: 전체 진단 실행**

Run: `python -X utf8 scripts/diagnose_pair_interaction.py`

기본값은 `--load-start 2025-01-01 --dev-end 2026-04-30 --test-start
2026-05-01 --test-end 2026-06-25 --n-boot 1000`이다. 부트스트랩 1000회는
dev 경주 수에 따라 수 분 걸릴 수 있다 — 오래 걸리면 백그라운드로 돌리고
`BackTestReport/Result_pair_interaction_*.md`에 저장된 리포트를 나중에
확인해도 된다.

- [ ] **Step 2: 판정**

출력의 "[신뢰구간]" 섹션을 본다:

- `b1`(both_early_leader), `b2`(entry_gap), `b3`(pace_pressure) 중
  **하나 이상이 "유의함(0 배제)"**이면: 상호작용 신호가 있다는 근거
  확보. 후보 1(페어 전용 분류기, 이번 계획 범위 밖)에 착수할 근거가
  된다는 결론.
- **전부 "유의하지 않음"**이면: 이 방향에서 개선 없음. 이 프로젝트는
  "개선 없음"도 유효한 결과로 취급하고 기록해왔다
  ([[project_new_signal_features_rollout]] 등 참고).

"[test 최고 페어 픽 적중률]"도 함께 기록한다(참고용, 판정의 주 근거는
신뢰구간).

- [ ] **Step 3: 메모리 기록**

`C:\Users\PC\.claude\projects\F--Project-----------\memory\project_multi_bet_probability_model.md`에
새 섹션을 추가한다(기존 파일에 이어 붙임, 새 파일 만들지 않음):

```
## YYYY-MM-DD 복연승 페어 상호작용 신호 스크리닝 결과

오프셋 로지스틱 회귀(logit(p_joint) = logit(top3_prob_i*top3_prob_j) +
b0 + b1*both_early_leader + b2*entry_gap + b3*pace_pressure)로 상호작용
신호 존재 여부를 스크리닝. dev<=2026-04-30 적합, test 2026-05-01~06-25
평가, 경주 단위 클러스터 부트스트랩(n=1000) 95% CI.

결과: [b0/b1/b2/b3 신뢰구간 실제 수치를 여기 채워라]
판정: [신호 있음 -> 후보1 착수 근거 확보 / 개선 없음 -> 이 방향 종결]
test 최고 페어 픽 적중률: [수치]% ([N]경주, 참고용 - 무작위 7.2%/
휴리스틱 38.5%/Harville 36.5%와 표본이 다를 수 있음)

스크립트: scripts/diagnose_pair_interaction.py
```

`MEMORY.md`의 `project_multi_bet_probability_model.md` 인덱스 줄에도
이번 결과를 한 문장으로 덧붙인다.

- [ ] **Step 4: 사용자에게 결과 보고**

판정 결과, 신뢰구간 수치, test 적중률, 다음 단계 제안(신호 있으면
"후보 1 착수를 새 브레인스토밍/계획으로 진행할지" 확인, 없으면 "이
방향 종결하고 배당 데이터 축적/ROI 쪽으로 전환할지" 확인)을 사용자에게
전달한다.

---

## Self-Review 체크리스트

- **스펙 커버리지**: 설계 스펙의 데이터 파이프라인 3단계(예측값 로드,
  과거 평균 초반순위, 페어 테이블) → Task 1/2/5. 모델(오프셋 로지스틱)
  → Task 3. 검증(dev/test 분리, 부트스트랩, 정확도 비교) → Task 4/6.
  판정 기준 → Task 6 Step 2. 범위 외 항목(후보1/3, 프로덕션 연결,
  config.py/feature_builder.py 수정, ROI, 신규 의존성) 전부 이 계획에
  없음 — 스펙과 일치.
- **플레이스홀더 스캔**: Task 6의 메모리 기록 템플릿에 "[실제 수치를
  채워라]"가 있는데, 이건 실행 결과에 따라 달라지는 값이라 계획
  단계에서 채울 수 없는 게 맞다(플레이스홀더 금지 규칙은 "구현 방법을
  모호하게 남기는 것"에 적용되는 것이고, 실행 후에만 알 수 있는 실측값
  기입은 다른 범주). 나머지 태스크에는 TBD/모호한 지시 없음.
- **타입/이름 일관성**: `fetch_avg_early_ord_asof`, `build_pair_table`,
  `fit_offset_logistic`, `bootstrap_coefficients`, `_load_predictions_with_ids`,
  `run_diagnosis` 함수 시그니처와 컬럼명(`p_indep`, `both_early_leader`,
  `entry_gap`, `pace_pressure`, `label`, `entry_i`, `entry_j`)이 Task
  2~6 전체에서 동일하게 사용됨을 확인.
- **공유 파일 보호**: `scripts/_backtest_common.py`와
  `core/feature_builder.py`는 어느 태스크에서도 수정하지 않음(import만).
  Task 5 Step 3에 "만약 컬럼이 사라지면 공유 파일을 고치지 말고 이
  스크립트 안에서 해결하라"는 명시적 가드를 넣어둠.
