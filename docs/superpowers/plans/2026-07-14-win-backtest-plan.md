# 단승(win) 백테스트 스크립트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 단승(win) 예측 정확도를 측정할 백테스트 스크립트(`scripts/backtest_win.py`)를 추가하고, 기존 `scripts/backtest_top3.py`와 중복되는 로딩/예측 로직을 공유 모듈로 추출한다.

**Architecture:** 모델 로드 → 과거 `race_entries` 로드 → 피처 빌드 → 전처리 → 예측까지의 파이프라인을 `scripts/_backtest_common.py`의 `run_predictions()`로 추출한다. `backtest_top3.py`는 이 함수를 사용하도록 리팩터링하고(출력 로직은 그대로 유지), `backtest_win.py`는 같은 함수의 결과(`win_prob`, `prev_win_odds` 포함)로 단승 전용 리포트(Precision@1, 임계값 스윕, 두수 필터, 인기마 기준선 비교)를 출력한다.

**Tech Stack:** Python, pandas, Supabase(supabase-py client via `Utils.database.get_client`), 기존 `core/feature_builder.py` / `core/preprocessor.py` / `core/trainer.py` / `core/model_manager.py`

**Spec:** `docs/superpowers/specs/2026-07-14-win-backtest-design.md`

---

### Task 1: 공통 백테스트 로딩/예측 모듈 추출

**Files:**
- Create: `scripts/_backtest_common.py`
- Reference (읽기만, 아직 수정 안 함): `scripts/backtest_top3.py:1-124`

- [ ] **Step 1: `scripts/_backtest_common.py` 작성**

```python
"""backtest_top3.py / backtest_win.py 공통 로딩·예측 로직"""
import pandas as pd

from Utils.database import get_client
from core.feature_builder import RaceFeatureBuilder
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.model_manager import ModelManager


def load_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """final_rank 있는 과거 race_entries 로드"""
    client = get_client()
    all_entries = []
    page_size = 1000
    offset = 0
    while True:
        result = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).gte("race_date", start_date)\
         .lte("race_date", end_date)\
         .gt("final_rank", 0)\
         .order("id")\
         .range(offset, offset + page_size - 1).execute()
        if not result.data:
            break
        all_entries.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_entries:
        return pd.DataFrame()

    fb = RaceFeatureBuilder()
    return fb._flatten_entries(all_entries)


def run_predictions(start_date: str, end_date: str):
    """모델 로드 → 과거 데이터 로드 → 피처 빌드 → 전처리 → 예측까지 수행.

    반환 DataFrame 컬럼: race_id, race_date, final_rank, top3_prob, win_prob,
    prev_win_odds, n_horses, is_top3_actual, is_win_actual.
    모델이 없거나 데이터가 없으면 None (오류 메시지는 내부에서 출력).
    """
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
    print(f"  로드: {len(df_raw)}건, {df_raw['race_id'].nunique() if 'race_id' in df_raw.columns else '?'}경주")

    print("피처 빌드 중...")
    fb = RaceFeatureBuilder()
    df_feat = fb.build_for_training(df_raw, end_date)
    df_feat = df_feat.dropna(subset=["horse_age", "race_distance"])

    final_ranks   = df_feat["final_rank"].copy()
    race_ids      = df_feat["race_id"].copy() if "race_id" in df_feat.columns else pd.Series(range(len(df_feat)))
    race_dates    = df_feat["race_date"].copy() if "race_date" in df_feat.columns else pd.Series([""] * len(df_feat))
    prev_win_odds = df_feat["prev_win_odds"].copy() if "prev_win_odds" in df_feat.columns else pd.Series([0.0] * len(df_feat))

    n_horses_map = df_feat.groupby("race_id").size().rename("n_horses")

    print("전처리 중...")
    df_proc = preprocessor.transform(df_feat)

    print("예측 중...")
    features  = trainer.feature_columns
    available = [c for c in features if c in df_proc.columns]
    X = df_proc[available].fillna(0)

    top3_probs = trainer.predict_proba_top3(X)
    win_probs  = trainer.predict_proba(X)

    if top3_probs is None:
        print("[경고] top3_models 없음. win_probability로 대체합니다.")
        top3_probs = win_probs

    result_df = pd.DataFrame({
        "race_id":       race_ids.values,
        "race_date":     race_dates.values,
        "final_rank":    final_ranks.values,
        "top3_prob":     top3_probs,
        "win_prob":      win_probs,
        "prev_win_odds": prev_win_odds.values,
    }, index=df_feat.index)

    result_df["is_top3_actual"] = (result_df["final_rank"] <= 3).astype(int)
    result_df["is_win_actual"]  = (result_df["final_rank"] == 1).astype(int)
    result_df = result_df.join(n_horses_map, on="race_id")

    return result_df
```

- [ ] **Step 2: import 오류 없는지 확인**

Run: `python -c "from scripts._backtest_common import run_predictions, load_historical_data; print('ok')"`
Expected: `ok` 출력 (project root에서 실행)

- [ ] **Step 3: Commit**

```bash
git add scripts/_backtest_common.py
git commit -m "refactor: 백테스트 공통 로딩/예측 로직을 _backtest_common으로 추출"
```

---

### Task 2: `backtest_top3.py`가 공통 모듈을 쓰도록 리팩터링 (동작 불변 확인)

**Files:**
- Modify: `scripts/backtest_top3.py`

- [ ] **Step 1: 리팩터링 전 기준 출력 저장 (회귀 확인용)**

Run: `python scripts/backtest_top3.py --start 2026-06-01 --end 2026-06-25 > /tmp/backtest_top3_before.txt 2>&1` (Windows PowerShell이면 `python scripts/backtest_top3.py --start 2026-06-01 --end 2026-06-25 | Out-File -Encoding utf8 before.txt`)
Expected: 파일에 "백테스트 결과", "두수 필터별 적중률", "top3_probability 임계값별 적중률", "80% 달성 권고 조건" 섹션이 정상 출력됨 (오류 없음)

- [ ] **Step 2: `scripts/backtest_top3.py` 전체를 아래 내용으로 교체**

```python
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
```

- [ ] **Step 3: 리팩터링 후 동일한 범위로 재실행, Step 1 결과와 비교**

Run: `python scripts/backtest_top3.py --start 2026-06-01 --end 2026-06-25`
Expected: Step 1에서 저장한 출력과 **총 경주 수 / 전체 적중률 / 각 표의 수치가 동일**해야 함 (모델·데이터가 그대로이므로 리팩터링 전후 숫자가 바뀌면 버그).

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest_top3.py
git commit -m "refactor: backtest_top3.py가 _backtest_common.run_predictions 사용하도록 정리"
```

---

### Task 3: 단승 백테스트 스크립트 작성

**Files:**
- Create: `scripts/backtest_win.py`

- [ ] **Step 1: `scripts/backtest_win.py` 작성**

```python
"""
단승(1착) 백테스트

목표: 경주별로 모델이 고른 '1순위 말'이 실제로 1착인 비율을 계산하고,
      win_probability 임계값·두수 필터별 적중률을 확인한다.
      함께 "가장 인기있는(직전 배당 최저) 말을 그대로 찍었을 때"의 적중률과
      비교해 모델이 시장(배당) 대비 우위가 있는지 판단한다.

실행:
    python scripts/backtest_win.py
    python scripts/backtest_win.py --start 2025-01-01 --end 2026-06-07
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts._backtest_common import run_predictions


def run_backtest(start_date: str, end_date: str):
    result_df = run_predictions(start_date, end_date)
    if result_df is None:
        return

    # 경주별 모델 픽(win_prob 최고) + 인기마 픽(prev_win_odds 최저) 선정
    def best_pick(grp):
        best = grp.loc[grp["win_prob"].idxmax()]
        favorite = grp.loc[grp["prev_win_odds"].idxmin()]
        return pd.Series({
            "race_date":       best["race_date"],
            "win_prob":        best["win_prob"],
            "is_win":          best["is_win_actual"],
            "n_horses":        best["n_horses"],
            "is_favorite_win": favorite["is_win_actual"],
        })

    race_results = result_df.groupby("race_id", sort=False).apply(best_pick).reset_index()

    total = len(race_results)
    overall_acc = race_results["is_win"].mean()
    favorite_acc = race_results["is_favorite_win"].mean()

    print(f"\n{'='*55}")
    print(f"  단승 백테스트 결과: {start_date} ~ {end_date}")
    print(f"{'='*55}")
    print(f"  총 경주 수:                {total}")
    print(f"  모델 Precision@1(필터 없음): {overall_acc:.1%}")
    print(f"  인기마(최저배당) 기준선:        {favorite_acc:.1%}")
    print()

    # ── win_probability 임계값별 적중률 ─────────────────────────────────────
    print("[ win_probability 임계값별 적중률 ]")
    print(f"  {'임계값':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for thresh in [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sub = race_results[race_results["win_prob"] >= thresh]
        if len(sub) == 0:
            break
        acc  = sub["is_win"].mean()
        ratio = len(sub) / total
        print(f"  {thresh:>7.2f}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

    # ── 두수 필터별 적중률 ───────────────────────────────────────────────────
    print()
    print("[ 두수 필터별 적중률 ]")
    print(f"  {'두수 ≤':>7}  {'경주수':>6}  {'적중률':>7}  {'전체비율':>8}")
    print(f"  {'-'*38}")
    for max_h in [999, 12, 10, 9, 8, 7, 6]:
        sub = race_results[race_results["n_horses"] <= max_h]
        if len(sub) == 0:
            continue
        acc  = sub["is_win"].mean()
        ratio = len(sub) / total
        label = "전체" if max_h == 999 else str(max_h)
        print(f"  {label:>7}  {len(sub):>6}  {acc:>7.1%}  {ratio:>8.1%}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단승 백테스트")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2026-06-07")
    args = parser.parse_args()
    run_backtest(args.start, args.end)
```

- [ ] **Step 2: 실행해서 4개 섹션이 정상 출력되는지 확인**

Run: `python scripts/backtest_win.py --start 2026-06-01 --end 2026-06-25`
Expected: 오류 없이 다음이 모두 출력됨 —
  - "단승 백테스트 결과" 헤더 + 총 경주 수 / 모델 Precision@1 / 인기마 기준선
  - "[ win_probability 임계값별 적중률 ]" 표
  - "[ 두수 필터별 적중률 ]" 표

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_win.py
git commit -m "feat: 단승(win) 백테스트 스크립트 추가"
```

---

### Task 4: 전체 기간으로 실행해 기준선(baseline) 확보

**Files:** 없음 (실행만, 코드 변경 없음)

- [ ] **Step 1: 스펙에서 정한 기본 검증 기간(408경주 백테스트 창)으로 실행**

Run: `python scripts/backtest_win.py --start 2026-05-01 --end 2026-06-25`
Expected: 모델 Precision@1과 인기마 기준선 수치가 출력됨. 이 값이 이번 작업의 실제 산출물(측정된 단승 정확도 기준선)이다 — 정답은 정해져 있지 않으므로 스크립트가 에러 없이 완주하는지, 표가 비어있지 않은지만 확인한다.

- [ ] **Step 2: 결과 수치를 사용자에게 보고**

이 단계는 코드 변경이 아니라 보고다. 실행 결과(전체 Precision@1, 인기마 기준선, 임계값별 표)를 그대로 사용자에게 전달한다. 후속 조치(임계값 도입 여부, 다음 단계 설계)는 사용자 확인 후 별도로 진행한다.

---

## Self-Review Notes

- **Spec coverage:** `scripts/backtest_win.py`(신규) ✓, `scripts/_backtest_common.py` 추출 ✓, `backtest_top3.py` 동작 불변 확인 ✓, Precision@1 ✓, 임계값 스윕 ✓, 두수 필터 ✓, 인기마 기준선 비교 ✓, CLI 인자 ✓. 스펙의 "완료 기준" 두 항목(스크립트 정상 출력, top3 회귀 없음) 모두 Task 2/3에서 검증됨.
- **Placeholder scan:** 없음 — 모든 스텝에 실행 가능한 전체 코드/명령 포함.
- **Type/이름 일관성:** `run_predictions()`가 반환하는 컬럼명(`win_prob`, `top3_prob`, `prev_win_odds`, `n_horses`, `is_top3_actual`, `is_win_actual`)이 Task 2·3의 `best_pick()`에서 참조하는 이름과 동일함을 확인.
