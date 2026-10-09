# 복연승 페어 랭커(3차, rank:pairwise) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scripts/diagnose_pair_classifier.py`에 `rank:pairwise` 목적함수를 쓰는 `XGBRanker` 경로(`train_and_evaluate_ranker`)를 추가해, 1차(38.5%)·2차(39.2%)를 넘고 오프셋 로지스틱(40.0%)에 도달하는지 확인한다.

**Architecture:** 기존 `train_and_evaluate`(분류기 버전)는 그대로 두고, `XGBRanker`를 쓰는 병렬 함수 `train_and_evaluate_ranker`를 추가한다. CLI에 `--model {classifier,ranker}` 플래그를 추가해 기본값은 기존 동작을 유지한다. 피처(13개)·하이퍼파라미터 규모·dev/test 창은 1~2차와 동일하게 유지해 "목적함수 전환의 효과"만 분리해서 본다.

**Tech Stack:** Python 3.11, xgboost 2.1.1(`XGBRanker`, 기존 의존성), scikit-learn(`GroupKFold`, 기존 의존성), pytest. 신규 의존성 없음.

**참고 문서:** `docs/superpowers/specs/2026-10-09-pair-ranker-design.md`

---

### 사전 확인 사항 (구현자가 알아야 할 것)

- 1~2차(커밋 3b5556f/94274ac/4566840/5172a7c/0934411/dfc5167)가 이미 완료·리뷰 승인되어 `scripts/diagnose_pair_classifier.py`에 `build_classifier_pair_table`, `add_race_relative_features`, `_load_predictions_with_probs`, `FEATURE_COLUMNS`(13개), `train_and_evaluate`, CLI가 전부 들어있고 `tests/test_diagnose_pair_classifier.py`에 15개 테스트가 통과 상태입니다. 2차 실행 결과(39.2%, `BackTestReport/Result_pair_classifier_20261009_180300.md`)는 오프셋 로지스틱(40.0%)에 아직 못 미쳤습니다.
- **`train_and_evaluate`/`build_classifier_pair_table`/`add_race_relative_features`는 이번 계획에서 수정하지 않습니다.** 새 함수만 추가합니다.
- `XGBRanker.fit(X, y, qid=...)`는 **같은 qid(경주)에 속한 행이 데이터 안에서 서로 붙어있어야** 합니다(xgboost 2.1.1 sklearn API 요구사항). `predict()`는 이 제약이 없습니다. 자세한 내용은 스펙 문서 "기술적 제약" 절 참고.
- 저장소에는 이 계획과 무관한 사용자의 미커밋 변경 17개 + 미추적 파일 3개가 있습니다. **절대 건드리지 마세요.** `git add`는 항상 이번에 수정하는 파일만 명시적으로 지정하세요.
- **실행 전 DB 연결을 가볍게 확인해라**: `python -c "from Utils.database import get_client; get_client().table('races').select('race_id', count='exact').limit(1).execute()"` — Supabase 무료 프로젝트 자동 일시정지 전례 있음. 실패하면 사용자에게 Restore를 요청하고 복구될 때까지 진행하지 마라. (Task 1은 DB가 필요 없으니 Task 2 직전에만 확인하면 된다.)
- 테스트 실행: `python -m pytest tests/test_diagnose_pair_classifier.py -v`

---

### Task 1: `_pick_accuracy`(TDD) + `train_and_evaluate_ranker` + CLI `--model` 플래그

**Files:**
- Modify: `scripts/diagnose_pair_classifier.py`
- Modify: `tests/test_diagnose_pair_classifier.py`

- [ ] **Step 1: `_pick_accuracy` 실패하는 테스트 추가**

`tests/test_diagnose_pair_classifier.py` 파일 끝에 추가(기존 15개 테스트
아래, 새 import 1줄 포함):

```python
from scripts.diagnose_pair_classifier import _pick_accuracy


def test_pick_accuracy_picks_highest_score_per_race():
    df = pd.DataFrame({
        "race_id": ["R1", "R1", "R1", "R2", "R2"],
        "p_joint": [0.9, 0.3, 0.1, 0.2, 0.8],
        "label":   [1,   0,   0,   0,   1],
    })
    # R1: 최고점(0.9)이 label=1, R2: 최고점(0.8)이 label=1 -> 2/2 = 100%
    assert _pick_accuracy(df) == pytest.approx(1.0)


def test_pick_accuracy_wrong_pick_lowers_accuracy():
    df = pd.DataFrame({
        "race_id": ["R1", "R1", "R2", "R2"],
        "p_joint": [0.9, 0.3, 0.8, 0.2],
        "label":   [0,   1,   0,   1],
    })
    # 두 경주 모두 최고점이 label=0 -> 0/2 = 0%
    assert _pick_accuracy(df) == pytest.approx(0.0)


def test_pick_accuracy_custom_score_column():
    df = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "my_score": [0.4, 0.6],
        "label":    [1,   0],
    })
    assert _pick_accuracy(df, score_col="my_score") == pytest.approx(0.0)
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v -k pick_accuracy`
Expected: FAIL with `ImportError: cannot import name '_pick_accuracy'`

- [ ] **Step 3: `_pick_accuracy` 구현 + xgboost import 수정**

`scripts/diagnose_pair_classifier.py`에서 다음 줄을 찾아:

```python
from xgboost import XGBClassifier
```

다음으로 교체:

```python
from xgboost import XGBClassifier, XGBRanker
```

그 다음, `train_and_evaluate` 함수 전체가 끝나는 지점(`print(f"\n리포트
저장: {rc.path}")` 다음 줄) 바로 뒤, `if __name__ == "__main__":` 블록
앞에 추가:

```python
def _pick_accuracy(df: pd.DataFrame, score_col: str = "p_joint") -> float:
    """경주별로 score_col이 가장 높은 행을 골라 label==1 비율을 낸다."""
    best_idx = df.groupby("race_id")[score_col].idxmax()
    return df.loc[best_idx]["label"].mean()
```

- [ ] **Step 4: 테스트 실행 → `_pick_accuracy` 3개 통과 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v -k pick_accuracy`
Expected: 3개 전부 PASS

- [ ] **Step 5: `train_and_evaluate_ranker` 추가**

같은 파일에서, 방금 추가한 `_pick_accuracy` 함수 바로 뒤, `if __name__
== "__main__":` 블록 앞에 추가:

```python
def train_and_evaluate_ranker(load_start: str, dev_end: str, test_start: str, test_end: str):
    df = _load_predictions_with_probs(load_start, test_end)
    if df is None:
        return

    print("과거 평균 초반통과순위(leak-safe) 계산 중...")
    avg_early_ord = fetch_avg_early_ord_asof(df)

    pair_df = build_classifier_pair_table(df, avg_early_ord)
    if pair_df.empty:
        print("[오류] 페어를 생성할 수 있는 경주가 없습니다")
        return
    pair_df = add_race_relative_features(pair_df)
    pair_df["race_date"] = pair_df["race_date"].astype(str)

    with capture_report("pair_ranker") as rc:
        dev_df = pair_df[pair_df["race_date"] <= dev_end].reset_index(drop=True)
        test_df = pair_df[(pair_df["race_date"] >= test_start) & (pair_df["race_date"] <= test_end)].reset_index(drop=True)

        print(f"\n{'='*70}")
        print(f"  복연승 페어 랭커(rank:pairwise): dev<= {dev_end}, test {test_start}~{test_end}")
        print(f"{'='*70}")
        print(f"dev 페어 수: {len(dev_df)} ({dev_df['race_id'].nunique()}경주)")
        print(f"test 페어 수: {len(test_df)} ({test_df['race_id'].nunique()}경주)")

        if dev_df.empty or test_df.empty:
            print("[오류] dev 또는 test 구간에 페어가 없습니다")
            return

        X_dev = dev_df[FEATURE_COLUMNS].astype(float)
        y_dev = dev_df["label"].astype(int)
        groups_dev = dev_df["race_id"]

        ranker_params = dict(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="rank:pairwise", random_state=42,
        )

        print("\nGroupKFold(5) 교차검증 중...")
        gkf = GroupKFold(n_splits=5)
        fold_pick_accs = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dev, y_dev, groups_dev), start=1):
            train_fold = dev_df.iloc[train_idx].sort_values("race_id").reset_index(drop=True)
            val_fold = dev_df.iloc[val_idx].sort_values("race_id").reset_index(drop=True)

            model = XGBRanker(**ranker_params)
            model.fit(
                train_fold[FEATURE_COLUMNS].astype(float),
                train_fold["label"].astype(int),
                qid=train_fold["race_id"],
            )
            val_fold = val_fold.copy()
            val_fold["p_joint"] = model.predict(val_fold[FEATURE_COLUMNS].astype(float))
            acc = _pick_accuracy(val_fold)
            fold_pick_accs.append(acc)
            print(f"  fold {fold}: 최고픽 적중률={acc:.3f}")
        print(f"  평균: 최고픽 적중률={np.mean(fold_pick_accs):.3f}")

        print("\ndev 전체로 최종 모델 학습 중...")
        dev_sorted = dev_df.sort_values("race_id").reset_index(drop=True)
        final_model = XGBRanker(**ranker_params)
        final_model.fit(
            dev_sorted[FEATURE_COLUMNS].astype(float),
            dev_sorted["label"].astype(int),
            qid=dev_sorted["race_id"],
        )

        print("\n[피처 중요도]")
        importances = sorted(zip(FEATURE_COLUMNS, final_model.feature_importances_),
                              key=lambda x: -x[1])
        for name, imp in importances:
            print(f"  {name}: {imp:.4f}")

        test_df = test_df.copy()
        X_test = test_df[FEATURE_COLUMNS].astype(float)
        test_df["p_joint"] = final_model.predict(X_test)

        acc = _pick_accuracy(test_df)
        n_races = test_df["race_id"].nunique()
        print(f"\n[test 최고 페어 픽 적중률] {acc:.1%}  ({n_races}경주)")
        print("  비교 기준: 1차(분류기) 38.5%, 2차(경주-상대 피처) 39.2%, "
              "오프셋 로지스틱(스크리닝) 40.0%, 무작위 7.2%, Harville 36.5%")

    print(f"\n리포트 저장: {rc.path}")
```

- [ ] **Step 6: CLI에 `--model` 플래그 추가**

같은 파일의 기존 `__main__` 블록을 찾아:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="복연승 페어 분류기(XGBoost) 학습·평가")
    parser.add_argument("--load-start", default=LOAD_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    args = parser.parse_args()
    train_and_evaluate(args.load_start, args.dev_end, args.test_start, args.test_end)
```

다음으로 교체:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="복연승 페어 분류기(XGBoost) 학습·평가")
    parser.add_argument("--load-start", default=LOAD_START_DEFAULT)
    parser.add_argument("--dev-end", default=DEV_END_DEFAULT)
    parser.add_argument("--test-start", default=TEST_START_DEFAULT)
    parser.add_argument("--test-end", default=TEST_END_DEFAULT)
    parser.add_argument("--model", choices=["classifier", "ranker"], default="classifier")
    args = parser.parse_args()
    if args.model == "ranker":
        train_and_evaluate_ranker(args.load_start, args.dev_end, args.test_start, args.test_end)
    else:
        train_and_evaluate(args.load_start, args.dev_end, args.test_start, args.test_end)
```

- [ ] **Step 7: 전체 테스트 재확인 + import 검증**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v`
Expected: 15(기존) + 3(신규) = **18개 전부 PASS**

Run: `python -c "import scripts.diagnose_pair_classifier"`
Expected: 에러 없음(import만으로 `XGBRanker` 참조 등 문법 오류가 없는지
확인)

- [ ] **Step 8: 커밋**

```bash
git add scripts/diagnose_pair_classifier.py tests/test_diagnose_pair_classifier.py
git commit -m "feat: 복연승 페어 랭커(3차) - train_and_evaluate_ranker + --model 플래그 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 2: 실제 실행 + 1차/2차/오프셋 로지스틱 대비 판정 + 메모리 기록

**Files:** 없음(실행+기록 전용)

- [ ] **Step 1: DB 연결 확인**

Run: `python -c "from Utils.database import get_client; r = get_client().table('races').select('race_id', count='exact').limit(1).execute(); print('OK', r.count)"`
Expected: `OK <건수>`. 실패하면 사용자에게 Supabase 대시보드에서 Restore를
요청하고, 복구될 때까지 이후 단계를 진행하지 마라.

- [ ] **Step 2: 랭커 파이프라인 실행**

Run: `python -X utf8 scripts/diagnose_pair_classifier.py --model ranker`

기본값(`--load-start 2025-01-01 --dev-end 2026-04-30 --test-start
2026-05-01 --test-end 2026-06-25`)과 함께 사용된다. `BackTestReport/
Result_pair_ranker_*.md`에 새 리포트가 저장된다(1~2차 리포트
`Result_pair_classifier_*.md`는 그대로 남아있다 — 삭제하지 않는다).

- [ ] **Step 3: 판정**

콘솔 출력(또는 리포트 파일)에서 `[test 최고 페어 픽 적중률]`과
`[피처 중요도]`, GroupKFold 평균 최고픽 적중률을 확인한다.

- **39.2%(2차)보다 오르고 40.0%(오프셋 로지스틱)에 도달하거나 넘으면**
  → 목적함수 전환이 남은 간극을 메운 것. 이 경로가 오프셋 로지스틱과
  동등하거나 그 이상이라는 근거 확보.
- **39.2%와 40.0% 사이면** → 또 부분 개선. 모델링 축(피처 추가·목적함수
  전환)으로 3연속 "부분 개선"만 나온 것이므로, 오프셋 로지스틱(40.0%)을
  실무 상한으로 받아들이고 ROI/배당 데이터 쪽으로 넘어가는 것을
  권고안으로 보고한다.
- **39.2% 이하면** → 목적함수 전환이 도움이 안 된 것. 모델링 축을
  종료하고 오프셋 로지스틱(40.0%)을 최종 상한으로 기록, ROI/배당 데이터
  쪽으로 넘어가는 것을 권고안으로 보고한다.

세 경우 모두 1~2차 피처 중요도(2차: `p_indep_rank_in_race` 0.4311,
`p_indep` 0.2940, `p_indep_ratio_to_max` 0.1116, `win_prob_i` 0.0217,
`entry_gap` 0.0204, `win_prob_diff` 0.0201, `n_pairs_in_race` 0.0191,
`win_prob_ratio` 0.0191, `top3_prob_i` 0.0185, `top3_prob_j` 0.0183,
`win_prob_j` 0.0150, `pace_pressure` 0.0111, `both_early_leader` 0.0000)
와 이번 결과를 나란히 비교해 보고한다.

- [ ] **Step 4: 메모리 기록**

`C:\Users\PC\.claude\projects\F--Project-----------\memory\
project_multi_bet_probability_model.md`를 **Read 도구로 먼저 읽고**,
파일 끝에 새 섹션을 이어 붙인다(기존 내용 변경 없이 추가만):

```
## 2026-10-09 복연승 페어 랭커 3차(rank:pairwise) 결과

1차(38.5%, 분류기)/2차(39.2%, 경주-상대 피처) 둘 다 오프셋 로지스틱
(40.0%) 미달. 같은 피처 13개, 같은 하이퍼파라미터 규모로 목적함수만
XGBClassifier(이진분류) -> XGBRanker(rank:pairwise)로 교체해 재검증.
GroupKFold 5-fold는 fold별 "경주 내 최고픽 적중률"로 평가(분류 지표
대신, 평가 목표와 일치시킴).

GroupKFold 평균 최고픽 적중률: [실제값]
test 최고 페어 픽 적중률: [실제값]% (1차 38.5%, 2차 39.2%, 오프셋
로지스틱 40.0%, 휴리스틱 38.5%, Harville 36.5%, 무작위 7.2%)
피처 중요도: [실제값]
판정: [40.0% 도달/부분개선/효과없음] - [결과 해석 한 줄]
결론: [모델링 축 지속 여부 및 다음 권고]

스크립트: scripts/diagnose_pair_classifier.py --model ranker
```

(각괄호 안은 실제 실행 결과로 채운다 — 템플릿 그대로 두지 않는다)

그 다음 `C:\Users\PC\.claude\projects\F--Project-----------\memory\
MEMORY.md`를 Read로 읽고, `project_multi_bet_probability_model.md` 항목
줄 끝에 한 문장만 덧붙인다:

```
**2026-10-09(3차): rank:pairwise 랭커 재검증 — test 적중률 [실제값]%
(2차 39.2%/오프셋 로지스틱 40.0% 대비 [도달/부분개선/효과없음]) →
[판정 한 줄, 모델링 축 지속 여부 포함]**
```

MEMORY.md는 그 한 줄만 덧붙이고 다른 내용은 건드리지 않는다.

- [ ] **Step 5: 보고**

실제 콘솔 출력(GroupKFold 최고픽 적중률, 피처 중요도 전체, test
적중률), 1~3차 대비 판정, 생성된 리포트 파일 경로, 메모리 파일 2개에
추가한 내용을 보고한다. 이 Task는 git 커밋이 없다(실행+기록 전용,
리포트 파일은 `.gitignore` 대상).

---

## Self-Review 체크리스트

- **스펙 커버리지**: `_pick_accuracy` 정의 → Task 1 Step 1~4.
  `train_and_evaluate_ranker`(qid 정렬, ranker 하이퍼파라미터, fold별
  최고픽 적중률 평가) → Task 1 Step 5. CLI `--model` 분기 → Task 1
  Step 6. 판정 기준 3가지 → Task 2 Step 3. 메모리 기록 형식 → Task 2
  Step 4. 범위 외 항목(튜닝/다른 랭킹 목적함수 비교/앙상블/기존 함수
  수정) 전부 이 계획에 없음 — 스펙과 일치.
- **플레이스홀더 스캔**: Task 2 Step 4의 메모리 템플릿 "[실제값]"은
  실행 결과로만 알 수 있는 값이라 계획 단계에서 채울 수 없는 게 맞다.
  코드 스텝은 전부 완전한 코드로 제공됨.
- **타입/이름 일관성**: `_pick_accuracy(df, score_col="p_joint")`가
  Task 1의 테스트 3개와 `train_and_evaluate_ranker` 안의 두 호출(`val_fold`,
  `test_df`에 대해 둘 다 컬럼명 `"p_joint"`를 씀) 모두에서 동일한
  시그니처로 쓰임을 확인. `FEATURE_COLUMNS`는 기존 13개를 그대로
  참조(재정의 없음). `ranker_params` 딕셔너리가 fold 루프와 최종 학습
  양쪽에서 동일하게 재사용됨(복붙 오류 방지 — 1~2차는 두 곳에 각각
  하이퍼파라미터를 복붙했지만 이번엔 변수로 한 번만 정의).
- **qid 정렬 제약 반영**: fold별 `train_fold`/최종 `dev_sorted`는 전부
  `.sort_values("race_id")` 후 `.fit(..., qid=...)`에 전달됨.
  `val_fold`/`test_df`의 `predict()` 호출에는 정렬이 없음(스펙에 명시된
  대로 `predict()`는 정렬 불필요).
