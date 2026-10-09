# 복연승 페어 분류기 2차(경주-상대 피처) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scripts/diagnose_pair_classifier.py`에 경주-상대 피처 3개(`p_indep_rank_in_race`, `p_indep_ratio_to_max`, `n_pairs_in_race`)를 추가해, 1차 결과(test 적중률 38.5%, 오프셋 로지스틱 40.0% 미달)를 넘어서는지 재검증한다.

**Architecture:** 기존 `build_classifier_pair_table`은 그대로 두고, 그 출력을 입력으로 받아 컬럼 3개를 덧붙이는 새 순수 함수 `add_race_relative_features`를 추가한다. `FEATURE_COLUMNS`를 10개→13개로 확장하고, `train_and_evaluate` 안에서 `build_classifier_pair_table` 호출 직후 이 함수를 끼워 넣는다. 모델 하이퍼파라미터·목적함수·dev/test 창은 1차와 완전히 동일하게 유지해 "피처 추가 자체의 효과"만 분리해서 본다.

**Tech Stack:** Python 3.11, pandas(groupby/rank/transform), pytest. 신규 의존성 없음.

**참고 문서:** `docs/superpowers/specs/2026-10-09-pair-classifier-round2-race-relative-features-design.md`

---

### 사전 확인 사항 (구현자가 알아야 할 것)

- 1차(Task 1~3, 커밋 3b5556f/94274ac/4566840/5172a7c)가 이미 완료·리뷰 승인되어 `scripts/diagnose_pair_classifier.py`에 `build_classifier_pair_table`, `_load_predictions_with_probs`, `train_and_evaluate`, CLI가 전부 들어있고 `tests/test_diagnose_pair_classifier.py`에 9개 테스트가 있습니다. 1차 실행 결과(38.5%, `BackTestReport/Result_pair_classifier_20261009_160627.md`)는 기존 오프셋 로지스틱 결과(40.0%)에 못 미쳤습니다.
- **`build_classifier_pair_table`은 이번 계획에서 수정하지 않습니다.** 새 함수만 추가합니다.
- 저장소에는 이 계획과 무관한 사용자의 미커밋 변경 17개 + 미추적 파일 3개가 있습니다. **절대 건드리지 마세요.** `git add`는 항상 이번에 만들거나 수정하는 파일만 명시적으로 지정하세요(`git add -A`/`git add .` 금지).
- **실행 전 DB 연결을 가볍게 확인해라**: `python -c "from Utils.database import get_client; get_client().table('races').select('race_id', count='exact').limit(1).execute()"` — 이 Supabase 무료 프로젝트는 장기 미사용 시 자동 일시정지된 전례가 여러 번 있다. 실패하면 사용자에게 대시보드에서 Restore를 요청하고, 복구될 때까지 이후 태스크를 진행하지 마라. (Task 1은 DB가 필요 없으니 Task 2 직전에만 확인하면 된다.)
- 테스트 실행: `python -m pytest tests/test_diagnose_pair_classifier.py -v` (리포지토리 루트에서).

---

### Task 1: `add_race_relative_features` + `FEATURE_COLUMNS` 확장 (TDD, 순수 함수)

**Files:**
- Modify: `scripts/diagnose_pair_classifier.py`
- Modify: `tests/test_diagnose_pair_classifier.py`

- [ ] **Step 1: `FEATURE_COLUMNS` 확장**

`scripts/diagnose_pair_classifier.py`에서 다음 기존 코드를 찾아:

```python
FEATURE_COLUMNS = [
    "win_prob_i", "win_prob_j", "top3_prob_i", "top3_prob_j",
    "p_indep", "win_prob_diff", "win_prob_ratio",
    "both_early_leader", "entry_gap", "pace_pressure",
]
```

다음으로 교체:

```python
FEATURE_COLUMNS = [
    "win_prob_i", "win_prob_j", "top3_prob_i", "top3_prob_j",
    "p_indep", "win_prob_diff", "win_prob_ratio",
    "both_early_leader", "entry_gap", "pace_pressure",
    "p_indep_rank_in_race", "p_indep_ratio_to_max", "n_pairs_in_race",
]
```

- [ ] **Step 2: 실패하는 테스트 추가**

`tests/test_diagnose_pair_classifier.py` 파일 끝에 추가(기존 9개 테스트
아래, 새 import 1줄 포함):

```python
from scripts.diagnose_pair_classifier import add_race_relative_features


def _make_race_relative_df():
    """R1: 3개 페어, p_indep이 전부 다름(0.5, 0.5, 0.2 중 둘이 동일 -> 동률
    케이스 포함). R2: 2개 페어, p_indep 전부 다름."""
    return pd.DataFrame({
        "race_id":  ["R1", "R1", "R1", "R2", "R2"],
        "entry_i":  [1, 1, 2, 1, 1],
        "entry_j":  [2, 3, 3, 2, 3],
        "p_indep":  [0.5, 0.5, 0.2, 0.6, 0.3],
    })


def test_add_race_relative_features_rank_highest_p_indep_is_1():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    row = result[(result.race_id == "R2") & (result.entry_j == 2)].iloc[0]
    assert row["p_indep_rank_in_race"] == pytest.approx(1.0)


def test_add_race_relative_features_tied_p_indep_gets_average_rank():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    r1 = result[result.race_id == "R1"]
    tied_rows = r1[r1["p_indep"] == 0.5]
    assert len(tied_rows) == 2
    assert (tied_rows["p_indep_rank_in_race"] == pytest.approx(1.5)).all()
    lowest = r1[r1["p_indep"] == 0.2].iloc[0]
    assert lowest["p_indep_rank_in_race"] == pytest.approx(3.0)


def test_add_race_relative_features_ratio_to_max_is_one_for_top_pair():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    r2 = result[result.race_id == "R2"]
    top = r2[r2["p_indep"] == 0.6].iloc[0]
    assert top["p_indep_ratio_to_max"] == pytest.approx(1.0)
    other = r2[r2["p_indep"] == 0.3].iloc[0]
    assert other["p_indep_ratio_to_max"] == pytest.approx(0.5)
    assert 0.0 <= other["p_indep_ratio_to_max"] <= 1.0


def test_add_race_relative_features_n_pairs_in_race_matches_group_size():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    assert (result[result.race_id == "R1"]["n_pairs_in_race"] == 3).all()
    assert (result[result.race_id == "R2"]["n_pairs_in_race"] == 2).all()


def test_add_race_relative_features_races_do_not_mix():
    df = _make_race_relative_df()
    result = add_race_relative_features(df)
    # R2는 페어가 2개뿐이지만 R1(3개)의 n_pairs_in_race에 영향을 주지 않는다
    assert (result[result.race_id == "R1"]["n_pairs_in_race"] != 2).all()
    # R1 순위가 R2 p_indep 값(0.6, 0.3)에 의해 바뀌지 않는다
    r1_ranks = sorted(result[result.race_id == "R1"]["p_indep_rank_in_race"].tolist())
    assert r1_ranks == [1.5, 1.5, 3.0]


def test_add_race_relative_features_order_independent():
    df = _make_race_relative_df()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    result = add_race_relative_features(shuffled)
    r2 = result[result.race_id == "R2"]
    top = r2[r2["p_indep"] == 0.6].iloc[0]
    assert top["p_indep_rank_in_race"] == pytest.approx(1.0)
    assert top["p_indep_ratio_to_max"] == pytest.approx(1.0)
```

- [ ] **Step 3: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v -k race_relative`
Expected: FAIL with `ImportError: cannot import name 'add_race_relative_features'`
(함수가 아직 없으므로)

- [ ] **Step 4: 함수 구현**

`scripts/diagnose_pair_classifier.py`에서 `build_classifier_pair_table`
함수 전체가 끝나는 지점(해당 함수의 `return pd.DataFrame(rows)` 다음 줄)
바로 뒤, `_load_predictions_with_probs` 함수 앞에 추가:

```python
def add_race_relative_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """build_classifier_pair_table의 출력에 경주-상대 피처 3개를 덧붙인다.

    p_indep_rank_in_race: 그 경주에서 이 페어의 p_indep 순위(1=최댓값).
    동률은 평균 순위(method="average")로 처리한다.
    p_indep_ratio_to_max: 이 페어의 p_indep ÷ 그 경주 p_indep 최댓값
    (0 나누기는 분모를 1e-6으로 바닥 처리해 방지, 0~1 범위, 그 경주의
    1등 페어는 정확히 1.0).
    n_pairs_in_race: 그 경주에 속한 페어의 총 개수.

    pair_df는 build_classifier_pair_table의 반환 형식(race_id, p_indep
    컬럼 포함)이어야 한다. 원본을 변경하지 않고 새 DataFrame을 반환한다.
    """
    pair_df = pair_df.copy()
    g = pair_df.groupby("race_id")["p_indep"]
    pair_df["p_indep_rank_in_race"] = g.rank(ascending=False, method="average")
    race_max = g.transform("max")
    pair_df["p_indep_ratio_to_max"] = pair_df["p_indep"] / race_max.clip(lower=1e-6)
    pair_df["n_pairs_in_race"] = pair_df.groupby("race_id")["race_id"].transform("size")
    return pair_df
```

- [ ] **Step 5: 테스트 실행 → 전체 통과 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v`
Expected: 9(기존) + 6(신규) = **15개 전부 PASS**

- [ ] **Step 6: 커밋**

```bash
git add scripts/diagnose_pair_classifier.py tests/test_diagnose_pair_classifier.py
git commit -m "feat: 복연승 페어 분류기 2차 - add_race_relative_features 추가

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 2: `train_and_evaluate`에 연결 + 실제 실행 + 1차 대비 판정 + 메모리 기록

**Files:**
- Modify: `scripts/diagnose_pair_classifier.py`

- [ ] **Step 1: DB 연결 확인**

Run: `python -c "from Utils.database import get_client; r = get_client().table('races').select('race_id', count='exact').limit(1).execute(); print('OK', r.count)"`
Expected: `OK <건수>`. 실패하면 사용자에게 Supabase 대시보드에서 Restore를
요청하고, 복구될 때까지 이후 단계를 진행하지 마라.

- [ ] **Step 2: `train_and_evaluate`에 연결**

`scripts/diagnose_pair_classifier.py`에서 `train_and_evaluate` 함수 안의
다음 코드를 찾아:

```python
    pair_df = build_classifier_pair_table(df, avg_early_ord)
    if pair_df.empty:
        print("[오류] 페어를 생성할 수 있는 경주가 없습니다")
        return
    pair_df["race_date"] = pair_df["race_date"].astype(str)
```

다음으로 교체(한 줄 추가):

```python
    pair_df = build_classifier_pair_table(df, avg_early_ord)
    if pair_df.empty:
        print("[오류] 페어를 생성할 수 있는 경주가 없습니다")
        return
    pair_df = add_race_relative_features(pair_df)
    pair_df["race_date"] = pair_df["race_date"].astype(str)
```

그 외 `train_and_evaluate`의 나머지 코드(GroupKFold, 최종 학습, 피처
중요도, test 평가)는 수정하지 않는다 — `FEATURE_COLUMNS`가 Task 1에서
이미 13개로 확장됐으므로 `X_dev`/`X_test` 구성 코드가 자동으로 새 피처
3개를 포함한다.

- [ ] **Step 3: 기존 테스트가 여전히 통과하는지 확인**

Run: `python -m pytest tests/test_diagnose_pair_classifier.py -v`
Expected: 15개 전부 PASS

- [ ] **Step 4: 커밋**

```bash
git add scripts/diagnose_pair_classifier.py
git commit -m "feat: 복연승 페어 분류기 2차 - train_and_evaluate에 경주-상대 피처 연결

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

- [ ] **Step 5: 전체 파이프라인 실행**

Run: `python -X utf8 scripts/diagnose_pair_classifier.py`

기본값(`--load-start 2025-01-01 --dev-end 2026-04-30 --test-start
2026-05-01 --test-end 2026-06-25`)으로 실행한다. `BackTestReport/
Result_pair_classifier_*.md`에 새 리포트가 저장된다(1차 리포트
`Result_pair_classifier_20261009_160627.md`는 그대로 남아있다 — 삭제하지
않는다, 비교용으로 보존).

- [ ] **Step 6: 판정**

콘솔 출력(또는 새 리포트 파일)에서 `[test 최고 페어 픽 적중률]`과
`[피처 중요도]`를 확인한다.

- **38.5%(1차)보다 오르고 40.0%(오프셋 로지스틱)를 넘으면** → 경주-상대
  피처가 학습-평가 간극을 메운 것. 이 방향(피처 엔지니어링)을 더 밀고
  나갈 근거 확보.
- **38.5%보다는 올랐지만 40.0% 미달이면** → 방향은 맞지만 아직 부족.
  "부분 개선, 추가 투자 필요"로 기록.
- **38.5% 근처거나 그대로면** → 이 진단(학습-평가 목표 불일치)이 주된
  원인이 아니었다는 뜻. 피처 중요도에서 새 3개 피처(`p_indep_rank_in_race`,
  `p_indep_ratio_to_max`, `n_pairs_in_race`)가 실제로 얼마나 쓰였는지
  (0에 가까우면 가설 자체가 기각) 반드시 같이 확인해 기록한다.

세 경우 모두 1차 피처 중요도(`p_indep` 0.3091, `win_prob_j` 0.1748,
`top3_prob_j` 0.1456, `top3_prob_i` 0.1453, `win_prob_i` 0.1103,
`win_prob_diff` 0.0439, `win_prob_ratio` 0.0288, `entry_gap` 0.0215,
`pace_pressure` 0.0123, `both_early_leader` 0.0085)와 이번 결과를 나란히
비교해 보고한다.

- [ ] **Step 7: 메모리 기록**

`C:\Users\PC\.claude\projects\F--Project-----------\memory\
project_multi_bet_probability_model.md`를 **Read 도구로 먼저 읽고**,
파일 끝에 새 섹션을 이어 붙인다(기존 내용 변경 없이 추가만):

```
## 2026-10-09 복연승 페어 분류기 2차(경주-상대 피처) 결과

1차(38.5%, p_indep 피처중요도 0.31 압도) 분석 결과 학습(독립 이진분류)과
평가(경주 내 argmax) 목표 불일치로 진단, 경주-상대 피처 3개
(p_indep_rank_in_race, p_indep_ratio_to_max, n_pairs_in_race) 추가해
재검증. 모델/하이퍼파라미터/dev·test창은 1차와 동일 유지.

test 최고 페어 픽 적중률: [실제값]% (1차 38.5%, 오프셋 로지스틱 40.0%,
휴리스틱 38.5%, Harville 36.5%, 무작위 7.2%)
신규 피처 중요도: [실제값]
판정: [40.0% 돌파/부분개선/효과없음] - [가설 검증 결과 한 줄]
```

(각괄호 안은 실제 실행 결과로 채운다 — 템플릿 그대로 두지 않는다)

그 다음 `C:\Users\PC\.claude\projects\F--Project-----------\memory\
MEMORY.md`를 Read로 읽고, `project_multi_bet_probability_model.md` 항목
줄 끝에 한 문장만 덧붙인다:

```
**2026-10-09(2차): 경주-상대 피처 추가 재검증 — test 적중률 [실제값]%
(1차 38.5%/오프셋 로지스틱 40.0% 대비 [상회/부분개선/변화없음]) →
[판정 한 줄]**
```

MEMORY.md는 그 한 줄만 덧붙이고 다른 내용은 건드리지 않는다.

- [ ] **Step 8: 보고**

실제 콘솔 출력(피처 중요도 전체, test 적중률), 1차 대비 판정, 생성된
리포트 파일 경로, 메모리 파일 2개에 추가한 내용을 보고한다. 이 Task는
Step 2/4 외 추가 git 커밋이 없다(Step 5~7은 실행+기록 전용, 리포트
파일은 `.gitignore` 대상이라 커밋하지 않는다).

---

## Self-Review 체크리스트

- **스펙 커버리지**: 설계 스펙의 `add_race_relative_features` 정의(순위/
  비율/개수, 동률 처리, 0 나누기 방지) → Task 1. `FEATURE_COLUMNS` 13개
  확장 → Task 1 Step 1. `train_and_evaluate` 연결 지점 → Task 2 Step 2.
  판정 기준 3가지 경우 → Task 2 Step 6. 메모리 기록 형식 → Task 2 Step 7.
  범위 외 항목(튜닝/랭킹 목적함수/앙상블/build_classifier_pair_table
  수정) 전부 이 계획에 없음 — 스펙과 일치.
- **플레이스홀더 스캔**: Task 2 Step 7의 메모리 템플릿 "[실제값]"은
  실행 결과로만 알 수 있는 값이라 계획 단계에서 채울 수 없는 게 맞다.
  그 외 코드 스텝은 전부 완전한 코드로 제공됨.
- **타입/이름 일관성**: `add_race_relative_features`가 반환하는 3개
  컬럼명(`p_indep_rank_in_race`, `p_indep_ratio_to_max`,
  `n_pairs_in_race`)이 Task 1의 `FEATURE_COLUMNS` 확장과 Task 1의
  테스트 6개에서 전부 동일하게 쓰임을 확인. `g.transform("max")`가
  반환하는 `race_max`와 `p_indep_ratio_to_max` 계산이 일관됨을 확인.
- **1차 함수 보호**: `build_classifier_pair_table`은 어느 Task에서도
  수정하지 않음(Task 2 Step 2는 그 호출 "직후"에 새 줄을 끼워 넣을
  뿐). Task 2 Step 1에 DB 연결 확인 가드를 명시함.
