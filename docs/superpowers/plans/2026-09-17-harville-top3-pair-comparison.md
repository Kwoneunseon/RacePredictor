# Harville 연승/복연승 비교 백테스트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이미 구현된 Harville(Plackett-Luce) 순차소거 확률 계산기(`core/rank_distribution.py`)를 실제 과거 경주 데이터에 적용해, (1) 연승확률이 기존 raw classifier와 비교해 같거나 나은지, (2) 복연승 예측이 기존 임시 휴리스틱보다 나은지 정량적으로 확인한다.

**Architecture:** `scripts/_backtest_common.py`에 두 개의 순수 함수(`attach_harville_top3`, `build_harville_pair_table`)를 추가해 기존 `run_predictions()` 출력에 Harville 확률을 얹는다. 그 위에 기존 `backtest_top3.py`/`backtest_quinella_place_heuristic.py`와 동일한 리포트 골격을 가진 스크립트 2개(`backtest_harville_top3.py`, `backtest_harville_pair_top3.py`)를 신규 작성한다. 캘리브레이션·프로덕션 연결은 하지 않는다(설계 스펙 참고).

**Tech Stack:** Python 3.11, pandas, numpy, pytest. DB 접근은 `Utils.database.get_client()`(Supabase PostgREST) — 이 계획의 코드/테스트 작업(Task 1~4)은 DB 없이 완결되고, Task 5(실제 실행)만 DB 연결 가능한 환경이 필요하다.

**참고 문서:**
- 설계 스펙: `docs/superpowers/specs/2026-09-17-harville-top3-pair-comparison-design.md`
- 원 스펙(Phase 1 전체 맥락): `docs/superpowers/specs/2026-07-25-multi-bet-probability-model-design.md`

---

### 사전 확인 사항 (구현자가 알아야 할 것)

- `core/rank_distribution.py`는 이미 구현·검증 완료(수정 금지). 제공 함수:
  - `top3_marginal_probabilities(strengths) -> np.ndarray` — 말별 연승확률(합=3)
  - `pair_top3_probabilities(strengths) -> np.ndarray` — n×n 대칭 페어 확률 행렬(대각선 0, 상삼각합=3)
  - 3두 미만 또는 강도합 <= 0이면 `ValueError` 발생
- `scripts/_backtest_common.py`의 `run_predictions(start_date, end_date)`는 말 단위 `pd.DataFrame`을 반환하며 최소 다음 컬럼을 포함한다: `race_id`, `race_date`, `final_rank`, `top3_prob`(raw), `win_prob`(캘리브레이션됨, 경주 내 합=1 보장 안 됨 — Harville 계산 시 함수 내부에서 정규화됨), `n_horses`, `is_top3_actual`, `is_win_actual`. **이 함수는 수정하지 않는다.**
- `pytest.ini`에 `testpaths = tests` 설정, `tests/__init__.py` 존재로 repo root가 자동으로 import 경로에 잡힌다. `scripts/`는 `__init__.py`가 없는 네임스페이스 패키지지만 `from scripts._backtest_common import ...` 형태로 이미 다른 테스트/스크립트에서 문제없이 동작 확인됨.
- 테스트는 `python -m pytest tests/test_backtest_common.py -v` 로 실행(리포지토리 루트에서, 가상환경 `python` 사용). DB 접근 불필요.

---

### Task 1: `attach_harville_top3` 헬퍼 추가

**Files:**
- Modify: `scripts/_backtest_common.py`
- Test: `tests/test_backtest_common.py` (신규 파일)

- [ ] **Step 1: 테스트 파일 생성 및 실패하는 테스트 작성**

`tests/test_backtest_common.py` 새로 생성:

```python
import numpy as np
import pandas as pd
import pytest

from scripts._backtest_common import attach_harville_top3


def _make_result_df():
    """race_id별 그룹: R1(5두, 정상), R2(2두, 3두 미만이라 스킵 대상),
    R3(4두, 정상, final_rank는 순위 뒤섞임)."""
    return pd.DataFrame({
        "race_id":    ["R1"] * 5 + ["R2"] * 2 + ["R3"] * 4,
        "race_date":  ["2026-01-01"] * 5 + ["2026-01-02"] * 2 + ["2026-01-03"] * 4,
        "win_prob":   [0.4, 0.3, 0.15, 0.1, 0.05] + [0.6, 0.4] + [0.4, 0.3, 0.2, 0.1],
        "final_rank": [1, 2, 3, 4, 5] + [1, 2] + [2, 1, 4, 3],
    })


def test_attach_harville_top3_sums_to_three_per_race():
    df = _make_result_df()
    out = attach_harville_top3(df)

    r1 = out[out["race_id"] == "R1"]
    assert r1["harville_top3_prob"].sum() == pytest.approx(3.0, abs=1e-9)
    assert (r1["harville_top3_prob"] >= 0).all()


def test_attach_harville_top3_skips_races_with_fewer_than_three_horses():
    df = _make_result_df()
    out = attach_harville_top3(df)

    r2 = out[out["race_id"] == "R2"]
    assert r2["harville_top3_prob"].isna().all()


def test_attach_harville_top3_preserves_row_count():
    df = _make_result_df()
    out = attach_harville_top3(df)
    assert len(out) == len(df)


def test_attach_harville_top3_handles_zero_sum_strengths():
    df = pd.DataFrame({
        "race_id":    ["R4"] * 3,
        "race_date":  ["2026-01-04"] * 3,
        "win_prob":   [0.0, 0.0, 0.0],
        "final_rank": [1, 2, 3],
    })
    out = attach_harville_top3(df)
    assert out["harville_top3_prob"].isna().all()
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_backtest_common.py -v`
Expected: `ImportError: cannot import name 'attach_harville_top3' from 'scripts._backtest_common'` (또는 `ModuleNotFoundError` — 아직 함수 없음)

- [ ] **Step 3: `attach_harville_top3` 구현**

`scripts/_backtest_common.py` 상단 import에 추가:

```python
import numpy as np
```

(기존 `import pandas as pd` 바로 위 또는 아래, 다른 import 블록과 함께)

파일 끝(또는 `run_predictions` 함수 뒤)에 함수 추가:

```python
def attach_harville_top3(result_df: pd.DataFrame) -> pd.DataFrame:
    """race_id별로 win_prob을 강도점수 삼아 Harville 순차소거로 유도한
    연승확률(harville_top3_prob) 컬럼을 추가해 반환한다.

    core.rank_distribution.top3_marginal_probabilities()는 출전마 3두 미만이거나
    강도 합이 0 이하면 ValueError를 던지므로, 그런 경주는 harville_top3_prob을
    NaN으로 채우고 건너뛴다(전체 백테스트를 중단시키지 않기 위함,
    [[project_multi_bet_probability_model]] 참고).

    groupby(...)["win_prob"].transform(...)를 쓰는 이유: groupby.apply()에
    그룹핑 컬럼(race_id)이 포함된 프레임을 통째로 넘기면 pandas가
    DeprecationWarning을 낸다(향후 그룹핑 컬럼이 자동 제외될 예정) - transform은
    win_prob 컬럼 하나만 다루므로 이 문제가 없다.
    """
    def _harville_col(s: pd.Series) -> pd.Series:
        strengths = s.to_numpy()
        if len(strengths) < 3 or strengths.sum() <= 0:
            return pd.Series(np.nan, index=s.index)
        return pd.Series(top3_marginal_probabilities(strengths), index=s.index)

    out = result_df.copy()
    out["harville_top3_prob"] = out.groupby("race_id", sort=False)["win_prob"].transform(_harville_col)

    n_skipped_races = out.loc[out["harville_top3_prob"].isna(), "race_id"].nunique()
    if n_skipped_races:
        print(f"  [주의] Harville 연승확률 계산 불가 경주 {n_skipped_races}건 스킵 "
              f"(출전마 3두 미만 또는 win_prob 합<=0)")
    return out
```

`scripts/_backtest_common.py` 상단 import 블록에 `rank_distribution` 함수 import 추가:

```python
from core.rank_distribution import top3_marginal_probabilities, pair_top3_probabilities
```

(`from core.model_manager import ModelManager` 줄 아래에 추가 — Task 2에서 쓸
`pair_top3_probabilities`도 미리 한 번에 import)

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `python -m pytest tests/test_backtest_common.py -v`
Expected: 4개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/_backtest_common.py tests/test_backtest_common.py
git commit -m "feat: Harville 연승확률 계산 헬퍼(attach_harville_top3) 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 2: `build_harville_pair_table` 헬퍼 추가

**Files:**
- Modify: `scripts/_backtest_common.py`
- Test: `tests/test_backtest_common.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_backtest_common.py` 끝에 추가(같은 `_make_result_df` 재사용):

```python
from scripts._backtest_common import build_harville_pair_table
from core.rank_distribution import pair_top3_probabilities


def test_build_harville_pair_table_excludes_small_races():
    df = _make_result_df()
    table = build_harville_pair_table(df)
    assert "R2" not in set(table["race_id"])
    assert set(table["race_id"]) == {"R1", "R3"}


def test_build_harville_pair_table_picks_highest_probability_pair():
    df = _make_result_df()
    table = build_harville_pair_table(df)

    r1_row = table[table["race_id"] == "R1"].iloc[0]
    strengths = df[df["race_id"] == "R1"]["win_prob"].to_numpy()
    pair = pair_top3_probabilities(strengths)

    assert r1_row["harville_pair_prob"] == pytest.approx(pair.max())


def test_build_harville_pair_table_hit_reflects_final_rank():
    df = _make_result_df()
    table = build_harville_pair_table(df)

    strengths = df[df["race_id"] == "R3"]["win_prob"].to_numpy()
    final_ranks = df[df["race_id"] == "R3"]["final_rank"].to_numpy()
    pair = pair_top3_probabilities(strengths)
    n = len(strengths)
    iu = np.triu_indices(n, k=1)
    best_idx = np.argmax(pair[iu])
    a, b = iu[0][best_idx], iu[1][best_idx]
    expected_hit = bool(final_ranks[a] <= 3 and final_ranks[b] <= 3)

    r3_row = table[table["race_id"] == "R3"].iloc[0]
    assert bool(r3_row["harville_pair_hit"]) == expected_hit


def test_build_harville_pair_table_random_baseline_formula():
    df = _make_result_df()
    table = build_harville_pair_table(df)
    r1_row = table[table["race_id"] == "R1"].iloc[0]
    assert r1_row["random_baseline"] == pytest.approx(6.0 / (5 * 4))


def test_build_harville_pair_table_skips_zero_sum_strengths():
    df = pd.DataFrame({
        "race_id":    ["R5"] * 4,
        "race_date":  ["2026-01-05"] * 4,
        "win_prob":   [0.0, 0.0, 0.0, 0.0],
        "final_rank": [1, 2, 3, 4],
    })
    table = build_harville_pair_table(df)
    assert table.empty
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_backtest_common.py -v`
Expected: 새로 추가한 5개 테스트가 `ImportError` 또는 `AttributeError`로 FAIL (`build_harville_pair_table` 미정의), 기존 4개는 PASS

- [ ] **Step 3: `build_harville_pair_table` 구현**

`scripts/_backtest_common.py`의 `attach_harville_top3` 뒤에 추가:

```python
def build_harville_pair_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """race_id별로 win_prob을 강도점수 삼아 pair_top3_probabilities()를 계산하고,
    확률이 가장 높은 페어(복연승 픽)와 적중 여부를 담은 경주 단위 테이블을 반환한다.

    기존 임시 휴리스틱(scripts/backtest_quinella_place_heuristic.py의
    build_quinella_table — "확률 상위 2마리를 그냥 묶는" 방식)과 달리, 두 말이
    동시에 top3에 들 결합확률 자체를 최대화하는 페어를 고른다
    ([[project_multi_bet_probability_model]] Phase 2).

    반환 컬럼: race_id, race_date, n_horses, harville_pair_prob(픽된 페어의
    확률), harville_pair_hit(픽된 페어가 실제로 둘 다 1~3착인지),
    random_baseline(무작위 2마리 픽 기대 적중률 = 6/(n*(n-1))).

    출전마 3두 미만이거나 win_prob 합이 0 이하인 경주는 결과에서 제외한다.
    """
    rows = []
    skipped_small = 0
    skipped_degenerate = 0
    for race_id, grp in result_df.groupby("race_id", sort=False):
        n = len(grp)
        if n < 3:
            skipped_small += 1
            continue

        strengths = grp["win_prob"].to_numpy()
        if strengths.sum() <= 0:
            skipped_degenerate += 1
            continue

        pair = pair_top3_probabilities(strengths)
        iu = np.triu_indices(n, k=1)
        best_idx = np.argmax(pair[iu])
        a, b = iu[0][best_idx], iu[1][best_idx]

        final_ranks = grp["final_rank"].to_numpy()
        hit = bool(final_ranks[a] <= 3 and final_ranks[b] <= 3)

        rows.append({
            "race_id": race_id,
            "race_date": str(grp["race_date"].iloc[0]),
            "n_horses": n,
            "harville_pair_prob": float(pair[a, b]),
            "harville_pair_hit": hit,
            "random_baseline": 6.0 / (n * (n - 1)),
        })

    if skipped_small:
        print(f"  [주의] 출전마 3두 미만 경주 {skipped_small}건은 복연승 정의상 스킵")
    if skipped_degenerate:
        print(f"  [주의] win_prob 합<=0 경주 {skipped_degenerate}건 스킵")

    return pd.DataFrame(rows)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `python -m pytest tests/test_backtest_common.py -v`
Expected: 9개 테스트(Task 1의 4개 + Task 2의 5개) 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add scripts/_backtest_common.py tests/test_backtest_common.py
git commit -m "feat: Harville 복연승 페어 선정 헬퍼(build_harville_pair_table) 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 3: `scripts/backtest_harville_top3.py` 작성

**Files:**
- Create: `scripts/backtest_harville_top3.py`

이 스크립트는 리포트 출력 스크립트라 단위테스트 대상이 아니다(기존 `backtest_top3.py`와 동일 관례). 대신 문법/임포트 오류가 없는지 실행 가능 여부만 확인한다.

- [ ] **Step 1: 파일 작성**

```python
"""
Harville 유도 연승확률 백테스트.

core/trainer.py의 top3 전용 classifier를 전혀 쓰지 않고, 이미 검증된
win_prob(캘리브레이션 적용)만으로 core/rank_distribution.py의 Harville
순차소거 공식을 적용해 연승확률을 유도한다
([[project_multi_bet_probability_model]] Phase 2, 2단계).

기존 raw classifier 벤치마크(TOP3_PROB_THRESHOLD=0.65 -> 약 80% 적중률/
61.6% 커버리지, 408경주, 2026-07-04 검증, [[project_next_task]])와 동일
기간·동일 리포트 구조(backtest_top3.py)로 비교한다.

캘리브레이션은 이 라운드에서 생략한다 — 임계값 스윕/픽 비교는 확률의
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

    result_df = attach_harville_top3(result_df)
    result_df = result_df.dropna(subset=["harville_top3_prob"])
    if result_df.empty:
        print("[오류] Harville 연승확률 계산 가능한 경주 없음")
        return

    with capture_report("harville_top3") as rc:
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
```

- [ ] **Step 2: 문법 오류 없는지 확인 (DB 접근 없이)**

Run: `python -c "import ast; ast.parse(open('scripts/backtest_harville_top3.py', encoding='utf-8').read())"`
Expected: 아무 출력 없이 종료(문법 오류 없음)

Run: `python -c "import scripts.backtest_harville_top3"`
Expected: 아무 출력 없이 종료(import 시점에 DB 접근 없음 — `run_backtest()`는 `__main__` 블록에서만 호출되므로 import만으로는 DB에 접근하지 않는다)

- [ ] **Step 3: 커밋**

```bash
git add scripts/backtest_harville_top3.py
git commit -m "feat: Harville 연승확률 vs 기존 classifier 백테스트 스크립트 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 4: `scripts/backtest_harville_pair_top3.py` 작성

**Files:**
- Create: `scripts/backtest_harville_pair_top3.py`

- [ ] **Step 1: 파일 작성**

```python
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
```

- [ ] **Step 2: 문법/임포트 오류 없는지 확인 (DB 접근 없이)**

Run: `python -c "import scripts.backtest_harville_pair_top3"`
Expected: 아무 출력 없이 종료 (import만으로 DB 접근 없음, `scripts.backtest_quinella_place_heuristic`도 같은 이유로 안전하게 import됨을 확인하는 것도 겸함)

- [ ] **Step 3: 커밋**

```bash
git add scripts/backtest_harville_pair_top3.py
git commit -m "feat: Harville 복연승 vs 기존 휴리스틱 백테스트 스크립트 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 5: 실제 백테스트 실행 및 결론 기록

**이 태스크는 DB 접근이 가능한 환경(사용자 로컬 터미널 등)에서 실행해야 한다.**
이번 세션의 실행 환경은 Supabase 호스트로 네트워크 연결이 불가능함을
확인했다(`getaddrinfo failed`) — Task 1~4는 이 문제와 무관하게 완결되지만,
이 태스크는 별도 환경에서 사람 또는 DB 접근 가능한 에이전트가 이어서
실행해야 한다.

**Files:** 없음(실행 전용 태스크)

- [ ] **Step 1: 저장된 모델이 있는지 확인**

Run: `python -c "from core.model_manager import ModelManager; from core.trainer import RaceModelTrainer; from core.preprocessor import RacePreprocessor; mm=ModelManager(); t=RaceModelTrainer(); p=RacePreprocessor(); print('모델 로드 성공' if mm.load(t, p) else '모델 없음 - python main.py --mode train 먼저 실행 필요')"`

- [ ] **Step 2: 연승 비교 백테스트 실행**

Run: `python scripts/backtest_harville_top3.py`

출력에서 다음을 확인: 전체 적중률, 임계값별 적중률/커버리지 표, "기존 0.65
임계값과 동일 커버리지 지점 비교" 섹션의 Harville 적중률 수치.

- [ ] **Step 3: 복연승 백테스트 실행**

Run: `python scripts/backtest_harville_pair_top3.py`

출력에서 TEST 구간의 "[베이스라인]", "[기존 휴리스틱]", "[Harville]" 세 줄 비교
수치를 확인.

- [ ] **Step 4: 판정 및 기록**

설계 스펙의 판정 기준을 적용한다:

- **연승**: Task 2 결과에서 Harville이 동일 커버리지(61.6%)에서 기존 raw
  classifier(약 80% 적중)와 같거나 나은 적중률을 보이면 "교체 후보로 채택,
  다음 라운드에서 `core/predictor.py` 연결 제안"으로 결론. 열등하면 "기존
  classifier 유지"로 결론.
- **복연승**: Task 3 결과에서 Harville pair 픽이 기존 휴리스틱(38.5% 부근,
  408경주 기준 기록값)보다 나은 적중률을 보이면 "4단계(EV 연결) 채택 후보로
  기록". 열등하거나 비슷하면 "개선 없음"으로 기록(이 프로젝트는 개선 없음도
  유효한 결과로 취급해왔다, [[project_new_signal_features_rollout]]).

두 결론 모두 다음 경로에 project 메모리로 기록한다(quant-model-expert의
표준 절차):

`C:\Users\PC\.claude\projects\F--Project-----------\memory\project_multi_bet_probability_model.md`

- 기존 파일에 "2026-XX-XX 후속: Harville 연승/복연승 비교 백테스트 결과"
  섹션을 추가(새 파일 만들지 않음 — 이미 있는 주제).
- `memory/MEMORY.md`의 해당 인덱스 줄도 한 줄 요약을 덧붙여 갱신.

- [ ] **Step 5: (연승이 채택된 경우에만) 후속 작업 제안**

Harville이 채택되면, `core/predictor.py`의 top3_probability 산출 로직 교체는
**이 계획의 범위 밖**이다 — 별도 브레인스토밍/계획 세션으로 진행한다(설계
스펙의 "범위 외" 섹션 참고). 이 스텝에서는 사용자에게 다음 세션에서
진행할지 물어보는 것으로 마무리한다.

---

## Self-Review 체크리스트 (계획 작성자가 직접 확인함)

- **스펙 커버리지**: 설계 스펙의 "신규/수정 컴포넌트" 표 3항목(`_backtest_common.py` 헬퍼 2개, 백테스트 스크립트 2개) 전부 Task 1~4에 매핑됨. "판정 기준" 섹션은 Task 5에 매핑됨. "테스트 방침"(합성 데이터 단위테스트)은 Task 1~2의 테스트 코드로 충족. "에러 처리/엣지케이스"(3두 미만, 강도합<=0)는 Task 1~2 테스트 케이스로 명시적으로 커버됨.
- **플레이스홀더 스캔**: "TBD"/"나중에" 패턴 없음. 모든 코드 스텝에 완전한 코드 포함.
- **타입/이름 일관성**: `attach_harville_top3`, `build_harville_pair_table` 함수 시그니처가 Task 1~4 전체에서 동일하게 사용됨(`harville_top3_prob`, `harville_pair_prob`, `harville_pair_hit`, `random_baseline` 컬럼명 통일 확인).
- **DB 의존성 격리 확인**: Task 1~4는 DB 없이 완결(순수 함수 + 문법/임포트 체크). Task 5만 DB 필요 — 이 사실을 태스크 설명에 명시함.
