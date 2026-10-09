# 복연승 페어 분류기(후보 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 경주 내 모든 유효 말 쌍을 한 행으로 재구성해 "둘 다 top3"를 직접 라벨로 학습하는 XGBoost 분류기를 만들어, 기존 오프셋 로지스틱 스크리닝 결과(test 최고 페어 픽 적중률 40.0%)를 넘어서는지 확인한다.

**Architecture:** 새 스크립트 `scripts/diagnose_pair_classifier.py` 하나에 (1) win_prob/top3_prob를 모두 보존하는 로더, (2) 페어 피처 테이블 생성(순수 함수, TDD), (3) `GroupKFold` 교차검증 + 최종 학습 + test 평가 + 피처 중요도를 전부 담는다. 기존 승인된 스크립트(`scripts/diagnose_pair_interaction.py`)의 leak-safe 헬퍼는 import해서 재사용하고 수정하지 않는다.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost(`XGBClassifier`, 기존 의존성), scikit-learn(`GroupKFold`, `accuracy_score`, `log_loss`, 기존 의존성), pytest.

**참고 문서:** `docs/superpowers/specs/2026-10-09-pair-classifier-design.md`

---

### 사전 확인 사항 (구현자가 알아야 할 것)

- `scripts/diagnose_pair_interaction.py`(5개 커밋 8cb2737~5bec97c, 전부 승인·실행 완료)에 `fetch_avg_early_ord_asof(df)`가 있다 — 말별 leak-safe 과거 평균 초반통과순위를 계산하는 함수. **이 파일은 수정하지 않고 import만 한다**: `from scripts.diagnose_pair_interaction import fetch_avg_early_ord_asof`.
- `Utils/ranks.py::valid_rank_mask(series)`로 유효 착순(`1 <= final_rank < config.NON_FINISH_RANK_MIN`)을 판정한다.
- `scripts/_backtest_common.py`의 `load_historical_data`, `capture_report`를 import해서 쓴다(수정 금지).
- 저장소에는 이 계획과 무관한 사용자의 미커밋 변경 17개 + 미추적 파일들이 있다. **이번 계획에서 만드는 파일(`scripts/diagnose_pair_classifier.py`, `tests/test_diagnose_pair_classifier.py`)은 전부 새 파일이라 충돌하지 않는다** — 그래도 커밋 전에 항상 `git status`/`git diff --cached --name-only`로 의도한 새 파일만 스테이지됐는지 확인해라.
- **실행 전 DB 연결을 가볍게 확인해라**: `python -c "from Utils.database import get_client; get_client().table('races').select('race_id', count='exact').limit(1).execute()"` — 이 Supabase 무료 프로젝트는 장기 미사용 시 자동 일시정지된 전례가 여러 번 있다. 실패하면 사용자에게 대시보드에서 Restore를 요청하고, 복구될 때까지 이후 태스크를 진행하지 마라.
- 테스트 실행: `python -m pytest tests/test_diagnose_pair_classifier.py -v` (리포지토리 루트에서). 콘솔 한글이 깨지면 `python -X utf8`.

---

### Task 1: 스크립트 골격 + `build_classifier_pair_table` (TDD, 순수 함수)

**Files:**
- Create: `scripts/diagnose_pair_classifier.py`
- Test: `tests/test_diagnose_pair_classifier.py` (신규)

- [ ] **Step 1: 실패하는 테스트와 함께 파일 생성**

`scripts/diagnose_pair_classifier.py` 생성:

```python
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
```

`tests/test_diagnose_pair_classifier.py` 생성:

```python
import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_pair_classifier import build_classifier_pair_table


def _make_df():
    """R1: 4두 전부 유효 착순(1,2,3,4위). R2: 3두 중 1두는 비착순 코드(99)라
    유효 말이 2두뿐 -> 3두 미만이라 페어 불가."""
    return pd.DataFrame({
        "race_id":      ["R1"] * 4 + ["R2"] * 3,
        "race_date":    ["2026-01-01"] * 4 + ["2026-01-02"] * 3,
        "entry_number": [1, 2, 3, 4, 1, 2, 3],
        "win_prob":     [0.5, 0.3, 0.15, 0.05, 0.4, 0.4, 0.2],
        "top3_prob":    [0.7, 0.6, 0.3, 0.2, 0.5, 0.5, 0.5],
        "final_rank":   [1, 2, 3, 4, 1, 2, 99],
    })


def test_build_classifier_pair_table_pair_count_matches_combinations():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    assert len(pairs[pairs["race_id"] == "R1"]) == 6  # C(4,2)
    assert len(pairs[pairs["race_id"] == "R2"]) == 0  # 유효 말 2두뿐(3두 미만)


def test_build_classifier_pair_table_label_is_both_top3():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["label"] == 1
    row14 = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row14["label"] == 0


def test_build_classifier_pair_table_win_prob_diff_is_absolute():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row["win_prob_diff"] == pytest.approx(abs(0.5 - 0.05))


def test_build_classifier_pair_table_win_prob_ratio_is_symmetric_and_bounded():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    expected = min(0.5, 0.05) / max(0.5, 0.05)
    assert row["win_prob_ratio"] == pytest.approx(expected)
    assert 0.0 <= row["win_prob_ratio"] <= 1.0


def test_build_classifier_pair_table_p_indep_is_product_of_top3_prob():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row["p_indep"] == pytest.approx(0.7 * 0.6)


def test_build_classifier_pair_table_both_early_leader_and_pace_pressure():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    r1 = pairs[pairs.race_id == "R1"]
    both_leader = r1[(r1.entry_i == 1) & (r1.entry_j == 2)].iloc[0]
    assert both_leader["both_early_leader"] == True
    assert (r1["pace_pressure"] == 2).all()


def test_build_classifier_pair_table_nan_avg_early_ord_treated_as_not_leader():
    df = pd.DataFrame({
        "race_id":      ["R3"] * 3,
        "race_date":    ["2026-01-03"] * 3,
        "entry_number": [1, 2, 3],
        "win_prob":     [0.5, 0.3, 0.2],
        "top3_prob":    [0.5, 0.4, 0.3],
        "final_rank":   [1, 2, 3],
    })
    avg_early_ord = pd.Series([1.0, np.nan, 1.0], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row12 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 2)].iloc[0]
    assert row12["both_early_leader"] == False
    row13 = pairs[(pairs.entry_i == 1) & (pairs.entry_j == 3)].iloc[0]
    assert row13["both_early_leader"] == True


def test_build_classifier_pair_table_entry_gap_is_absolute_difference():
    df = _make_df()
    avg_early_ord = pd.Series([1.0, 1.0, 5.0, 5.0, np.nan, np.nan, np.nan], index=df.index)
    pairs = build_classifier_pair_table(df, avg_early_ord)
    row = pairs[(pairs.race_id == "R1") & (pairs.entry_i == 1) & (pairs.entry_j == 4)].iloc[0]
    assert row["entry_gap"] == pytest.approx(3.0)
```

- [ ] **Step 2: 테스트 실행 → 통과 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v`
Expected: 8개 테스트 모두 PASS (이 함수는 Step 1에서 이미 구현까지 같이
작성했으므로, 바로 통과해야 한다 — 실패하면 Step 1의 코드를 다시 확인해라)

- [ ] **Step 3: 커밋**

```bash
git add scripts/diagnose_pair_classifier.py tests/test_diagnose_pair_classifier.py
git commit -m "feat: 복연승 페어 분류기 - 골격 + build_classifier_pair_table

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 2: `_load_predictions_with_probs` (DB/모델 의존, 실DB 스모크 테스트)

**Files:**
- Modify: `scripts/diagnose_pair_classifier.py`

이 함수는 DB/모델에 의존해 단위테스트 대상이 아니다(기존
`scripts/diagnose_pair_interaction.py::_load_predictions_with_ids`와 동일한
관례). 문법/임포트 검증 후 실DB로 직접 확인한다.

- [ ] **Step 1: 함수 추가**

`scripts/diagnose_pair_classifier.py`의 `FEATURE_COLUMNS` 정의 뒤,
`build_classifier_pair_table` 함수 앞에 추가:

```python
def _load_predictions_with_probs(start_date: str, end_date: str):
    """모델 로드 -> load_historical_data -> build_for_training ->
    preprocessor.transform까지 거쳐, 캘리브레이션된 win_prob와 raw
    top3_prob를 모두 계산해 horse_id/entry_number와 함께 유지한다.

    scripts/diagnose_pair_interaction.py::_load_predictions_with_ids와
    거의 같은 로직이지만 win_prob도 함께 계산한다는 점이 다르다.
    scripts/_backtest_common.py는 수정하지 않고 이 스크립트 안에
    독립적으로 재구현한다(공유 파일 미커밋 변경과의 충돌을 피함).

    반환 컬럼: race_id, race_date, horse_id, entry_number, final_rank,
    win_prob, top3_prob. 모델/데이터가 없으면 None.
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
            print(f"[오류] build_for_training 이후 '{col}' 컬럼이 사라짐")
            return None

    print("전처리 중...")
    df_proc = preprocessor.transform(df_feat)

    print("예측 중...")
    features = trainer.feature_columns
    available = [c for c in features if c in df_proc.columns]
    X = df_proc[available].fillna(0)

    win_probs = trainer.predict_proba(X)
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
        "win_prob":     win_probs,
        "top3_prob":    top3_probs,
    }, index=df_feat.index)

    return result_df
```

- [ ] **Step 2: 기존 테스트가 여전히 통과하는지 확인 (DB 불필요)**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v`
Expected: 8개 테스트 모두 PASS

- [ ] **Step 3: DB 연결 확인 후 실DB 스모크 테스트**

Run: `python -c "from Utils.database import get_client; r = get_client().table('races').select('race_id', count='exact').limit(1).execute(); print('OK', r.count)"`
Expected: `OK <건수>`. 실패하면(DNS/연결 오류) 사용자에게 Supabase 대시보드에서
프로젝트 Restore를 요청하고, 복구될 때까지 이후 단계를 진행하지 마라.

Run: `python -X utf8 -c "
from scripts.diagnose_pair_classifier import _load_predictions_with_probs
df = _load_predictions_with_probs('2026-06-01', '2026-06-07')
print('컬럼:', list(df.columns) if df is not None else None)
print('행수:', len(df) if df is not None else 0)
if df is not None:
    print('win_prob 결측:', df['win_prob'].isna().sum())
    print('top3_prob 결측:', df['top3_prob'].isna().sum())
    print('horse_id 결측:', df['horse_id'].isna().sum())
"`

Expected: 컬럼에 `win_prob`, `top3_prob`, `horse_id`, `entry_number`가
모두 있고 결측이 0이어야 한다(`horse_id`/`entry_number` 보존은 같은
`build_for_training` 파이프라인으로 이미 한 번 실측 확인됐다 —
2026-09-29 Task 5. `win_prob`는 이번에 처음 같은 파이프라인에 추가하는
것이므로 결측 여부를 직접 확인해라).

- [ ] **Step 4: 커밋**

```bash
git add scripts/diagnose_pair_classifier.py
git commit -m "feat: 복연승 페어 분류기 - _load_predictions_with_probs 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 3: 학습·평가 파이프라인 조립(`train_and_evaluate` + CLI)

**Files:**
- Modify: `scripts/diagnose_pair_classifier.py`

- [ ] **Step 1: 함수 추가 및 `__main__` 블록 교체**

`scripts/diagnose_pair_classifier.py`의 `build_classifier_pair_table` 함수
뒤, 기존 `if __name__ == "__main__": pass`를 **완전히 지우고** 다음으로
교체:

```python
def train_and_evaluate(load_start: str, dev_end: str, test_start: str, test_end: str):
    df = _load_predictions_with_probs(load_start, test_end)
    if df is None:
        return

    print("과거 평균 초반통과순위(leak-safe) 계산 중...")
    avg_early_ord = fetch_avg_early_ord_asof(df)

    pair_df = build_classifier_pair_table(df, avg_early_ord)
    if pair_df.empty:
        print("[오류] 페어를 생성할 수 있는 경주가 없습니다")
        return
    pair_df["race_date"] = pair_df["race_date"].astype(str)

    with capture_report("pair_classifier") as rc:
        dev_df = pair_df[pair_df["race_date"] <= dev_end].reset_index(drop=True)
        test_df = pair_df[(pair_df["race_date"] >= test_start) & (pair_df["race_date"] <= test_end)].reset_index(drop=True)

        print(f"\n{'='*70}")
        print(f"  복연승 페어 분류기: dev<= {dev_end}, test {test_start}~{test_end}")
        print(f"{'='*70}")
        print(f"dev 페어 수: {len(dev_df)} ({dev_df['race_id'].nunique()}경주)")
        print(f"test 페어 수: {len(test_df)} ({test_df['race_id'].nunique()}경주)")

        if dev_df.empty or test_df.empty:
            print("[오류] dev 또는 test 구간에 페어가 없습니다")
            return

        X_dev = dev_df[FEATURE_COLUMNS].astype(float)
        y_dev = dev_df["label"].astype(int)
        groups_dev = dev_df["race_id"]

        print("\nGroupKFold(5) 교차검증 중...")
        gkf = GroupKFold(n_splits=5)
        fold_accs = []
        fold_loglosses = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dev, y_dev, groups_dev), start=1):
            model = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
            )
            model.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
            val_probs = model.predict_proba(X_dev.iloc[val_idx])[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            acc = accuracy_score(y_dev.iloc[val_idx], val_preds)
            ll = log_loss(y_dev.iloc[val_idx], val_probs, labels=[0, 1])
            fold_accs.append(acc)
            fold_loglosses.append(ll)
            print(f"  fold {fold}: 정확도={acc:.3f}  로그손실={ll:.3f}")
        print(f"  평균: 정확도={np.mean(fold_accs):.3f}  로그손실={np.mean(fold_loglosses):.3f}")

        print("\ndev 전체로 최종 모델 학습 중...")
        final_model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
        final_model.fit(X_dev, y_dev)

        print("\n[피처 중요도]")
        importances = sorted(zip(FEATURE_COLUMNS, final_model.feature_importances_),
                              key=lambda x: -x[1])
        for name, imp in importances:
            print(f"  {name}: {imp:.4f}")

        X_test = test_df[FEATURE_COLUMNS].astype(float)
        test_df = test_df.copy()
        test_df["p_joint"] = final_model.predict_proba(X_test)[:, 1]

        best_idx = test_df.groupby("race_id")["p_joint"].idxmax()
        best = test_df.loc[best_idx]
        acc = best["label"].mean()
        print(f"\n[test 최고 페어 픽 적중률] {acc:.1%}  ({len(best)}경주)")
        print("  비교 기준(같은 창은 아닐 수 있음): 무작위 7.2%, 기존 휴리스틱 38.5%, "
              "Harville 36.5%, 오프셋 로지스틱(스크리닝) 40.0%")

    print(f"\n리포트 저장: {rc.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="복연승 페어 분류기(XGBoost) 학습·평가")
    parser.add_argument("--load-start", default=LOAD_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    args = parser.parse_args()
    train_and_evaluate(args.load_start, args.dev_end, args.test_start, args.test_end)
```

- [ ] **Step 2: 기존 테스트가 여전히 통과하는지 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v`
Expected: 8개 테스트 모두 PASS

- [ ] **Step 3: 커밋**

```bash
git add scripts/diagnose_pair_classifier.py
git commit -m "feat: 복연승 페어 분류기 - 학습·평가 파이프라인 조립(train_and_evaluate)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 4: 실제 실행 및 판정

**이 태스크는 DB 접근이 가능한 환경에서 실행한다.** 실행 직전에 Task 2
Step 3와 같은 방식으로 DB 연결을 다시 확인해라(시간이 지나 프로젝트가
다시 일시정지됐을 수 있다).

**Files:** 없음(실행 전용 태스크)

- [ ] **Step 1: 전체 파이프라인 실행**

Run: `python -X utf8 scripts/diagnose_pair_classifier.py`

기본값은 `--load-start 2025-01-01 --dev-end 2026-04-30 --test-start
2026-05-01 --test-end 2026-06-25`다. `BackTestReport/Result_pair_classifier_*.md`에
리포트가 저장된다.

- [ ] **Step 2: 판정**

"[test 최고 페어 픽 적중률]"을 본다:

- **40.0%(기존 오프셋 로지스틱 결과)를 넘으면** → 더 유연한 모델이
  추가 신호를 뽑아낸 것. 다음 단계(배당 데이터 확보 후 ROI 검증,
  캘리브레이션)를 검토할 근거 확보.
- **40.0% 근처거나 못 미치면** → 오프셋 로지스틱이 이미 신호를 거의
  다 뽑아낸 것. "개선 없음(오프셋 로지스틱 수준이 한계)"으로 기록.

"[피처 중요도]" 순위도 함께 기록한다 — 어떤 피처가 실제로 기여하는지는
결과와 무관하게 항상 유용한 정보다.

GroupKFold 5-fold 평균 정확도/로그손실도 참고용으로 기록한다(test
적중률과 fold 평균이 크게 다르면 과적합 신호일 수 있음을 염두에
두어라).

- [ ] **Step 3: 메모리 기록**

`C:\Users\PC\.claude\projects\F--Project-----------\memory\project_multi_bet_probability_model.md`에
새 섹션을 이어 붙인다(기존 파일에 추가, 새 파일 만들지 않음):

```
## YYYY-MM-DD 복연승 페어 분류기(XGBoost) 결과

경주 내 모든 유효 말 쌍을 라벨(둘 다 top3)로 직접 학습하는 XGBoost
단일 모델(튜닝 없음, GroupKFold(5) 교차검증). dev<=2026-04-30,
test 2026-05-01~06-25(408경주).

GroupKFold 평균: 정확도=[수치] 로그손실=[수치]
test 최고 페어 픽 적중률: [수치]% (비교: 무작위 7.2%, 휴리스틱 38.5%,
Harville 36.5%, 오프셋 로지스틱 40.0%)
피처 중요도 순위: [상위 몇 개]
판정: [40.0% 돌파 -> 추가 신호 확보 / 근처·미달 -> 오프셋 로지스틱이 한계]

스크립트: scripts/diagnose_pair_classifier.py
```

`MEMORY.md`의 `project_multi_bet_probability_model.md` 인덱스 줄에도
한 문장 덧붙인다.

- [ ] **Step 4: 사용자에게 결과 보고**

판정 결과, GroupKFold 수치, test 적중률, 피처 중요도, 다음 단계 제안을
짧게 전달한다.

---

## Self-Review 체크리스트

- **스펙 커버리지**: 설계 스펙의 데이터 파이프라인 3단계(확률 로드,
  leak-safe 초반순위, 페어 피처 테이블) → Task 1/2. 모델(XGBoost 고정
  하이퍼파라미터) → Task 3. 검증(GroupKFold + test 평가 + 피처
  중요도) → Task 3/4. 판정 기준 → Task 4 Step 2. 범위 외 항목(튜닝,
  앙상블, 프로덕션 연결, ROI, 공유 파일 수정) 전부 이 계획에 없음 —
  스펙과 일치.
- **플레이스홀더 스캔**: Task 4의 메모리 기록 템플릿에 "[실제 수치를
  채워라]"가 있는데, 실행 결과로만 알 수 있는 값이라 계획 단계에서
  채울 수 없는 게 맞다(다른 태스크에는 모호한 지시 없음).
- **타입/이름 일관성**: `FEATURE_COLUMNS`(Task 1에서 정의)가
  `build_classifier_pair_table`의 반환 컬럼명(Task 1), `train_and_evaluate`의
  `X_dev`/`X_test` 구성(Task 3)에서 전부 동일한 10개 이름을 씀을 확인.
  `_load_predictions_with_probs`(Task 2)가 반환하는 `win_prob`/`top3_prob`
  컬럼명이 Task 1의 `build_classifier_pair_table`이 기대하는 입력
  컬럼명과 일치함을 확인.
- **공유 파일 보호**: `scripts/diagnose_pair_interaction.py`,
  `scripts/_backtest_common.py`, `core/feature_builder.py` 어느
  태스크에서도 수정하지 않음(import만). Task 2 Step 3에 DB 연결 확인
  가드를 명시함.
