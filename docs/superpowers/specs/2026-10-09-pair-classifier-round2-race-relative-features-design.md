# 복연승 페어 분류기 2차: 경주-상대 피처 추가 설계

## 배경

1차 시도(`scripts/diagnose_pair_classifier.py`, 튜닝 없는 XGBoost 단일 모델,
피처 10개)의 test 최고 페어 픽 적중률은 38.5%(408경주)로, 기존 오프셋
로지스틱 스크리닝 결과 40.0%에 못 미쳤다(실행 리포트
`BackTestReport/Result_pair_classifier_20261009_160627.md`, 메모리
[[project_multi_bet_probability_model]] "2026-10-09 복연승 페어
분류기(XGBoost) 결과" 절 참고).

피처 중요도를 보면 원인이 드러난다: 1위가 `p_indep`(독립가정, 0.3091)이고,
오프셋 로지스틱에서 유의했던 상호작용 피처 3종(`both_early_leader`
0.0085, `entry_gap` 0.0215, `pace_pressure` 0.0123, 합 4.2%)은 거의
쓰이지 않았다.

이 결과를 구조적으로 재해석하면: 평가 방식(`[test 최고 페어 픽 적중률]`)은
**경주 내 상대 비교**(그 경주에서 가장 높은 `p_joint`를 가진 페어를
고름)인데, 학습 방식은 페어 하나하나를 **독립적인 이진분류**로 다룬다.
`p_indep`은 절댓값이 크고 분산도 커서 이진분류 손실(로그손실)을 줄이는
데는 유리하지만, 상호작용 보정항은 절댓값이 작아서(오프셋 로지스틱
계수가 -0.02~0.27 수준) "같은 경주 안에서 1위를 뒤집을 만큼"의 영향력을
트리 분할 과정에서 거의 얻지 못한다. 즉 모델이 상호작용을 무시하는 게
아니라, 학습 목표(이진분류 손실)와 평가 목표(경주 내 argmax)가 어긋나
있어서 상호작용이 평가에 반영될 기회 자체가 적다.

## 목표

모델에 "이 페어가 같은 경주의 다른 페어들과 비교해 몇 등인가"를 직접
알려주는 피처를 추가해, 학습 신호와 평가 목표의 간극을 줄인다. 이번
라운드는 **피처 추가라는 한 가지 변화만** 주고(하이퍼파라미터·목적함수는
1차와 동일하게 유지), 그 효과만 분리해서 확인한다.

## 결정된 사항

- **기존 `build_classifier_pair_table`(scripts/diagnose_pair_classifier.py,
  1차 Task 1에서 구현·승인됨)은 수정하지 않는다.** 이미 테스트·리뷰를
  통과한 함수의 계약(반환 컬럼, 동작)을 그대로 유지한다. 새 함수
  `add_race_relative_features(pair_df)`를 추가해, 그 출력을 입력으로 받아
  컬럼 3개를 덧붙인 DataFrame을 반환한다.
- **모델 하이퍼파라미터는 1차와 완전히 동일하게 유지한다**
  (`XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
  random_state=42)`). 튜닝은 이번 범위에 없다 — 피처 추가 자체의 효과를
  튜닝 효과와 섞지 않기 위해서다.
- **목적함수/모델 구조도 그대로(이진분류, GroupKFold(5))**. 랭킹 목적함수
  전환(브레인스토밍에서 제시한 "방향 2")은 이번 라운드에 포함하지 않는다.
- **dev/test 창도 1차와 동일하게 유지**: dev≤2026-04-30, test
  2026-05-01~06-25(408경주). 1차 결과(38.5%)와 직접 비교 가능해야 한다.
- 앙상블·프로덕션 연결·ROI 계산 없음(1차와 동일한 범위 외 항목).

## 신규 함수

`scripts/diagnose_pair_classifier.py`에 추가(`build_classifier_pair_table`
함수 뒤, `_load_predictions_with_probs` 함수 뒤 — 두 함수 모두보다 아래,
`train_and_evaluate` 함수보다 위):

```python
def add_race_relative_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """build_classifier_pair_table의 출력에 경주-상대 피처 3개를 덧붙인다.

    p_indep_rank_in_race: 그 경주에서 이 페어의 p_indep 순위(1=최댓값).
    동률은 평균 순위(method="average")로 처리한다.
    p_indep_ratio_to_max: 이 페어의 p_indep ÷ 그 경주 p_indep 최댓값
    (0 나누기는 분모를 1e-6으로 바닥 처리해 방지, 0~1 범위, 그 경주의
    1등 페어는 정확히 1.0).
    n_pairs_in_race: 그 경주에 속한 페어의 총 개수.

    pair_df는 build_classifier_pair_table의 반환 형식(race_id 컬럼 포함)
    이어야 한다. 원본을 변경하지 않고 새 DataFrame을 반환한다.
    """
    pair_df = pair_df.copy()
    g = pair_df.groupby("race_id")["p_indep"]
    pair_df["p_indep_rank_in_race"] = g.rank(ascending=False, method="average")
    race_max = g.transform("max")
    pair_df["p_indep_ratio_to_max"] = pair_df["p_indep"] / race_max.clip(lower=1e-6)
    pair_df["n_pairs_in_race"] = pair_df.groupby("race_id")["race_id"].transform("size")
    return pair_df
```

`FEATURE_COLUMNS`(기존 10개)에 3개를 추가해 13개로 확장:

```python
FEATURE_COLUMNS = [
    "win_prob_i", "win_prob_j", "top3_prob_i", "top3_prob_j",
    "p_indep", "win_prob_diff", "win_prob_ratio",
    "both_early_leader", "entry_gap", "pace_pressure",
    "p_indep_rank_in_race", "p_indep_ratio_to_max", "n_pairs_in_race",
]
```

## 파이프라인 변경

`train_and_evaluate` 안에서 `build_classifier_pair_table(df, avg_early_ord)`
호출 직후, `pair_df.empty` 체크보다 먼저(빈 DataFrame에
`add_race_relative_features`를 호출해도 안전하지만, 순서상 자연스러운
위치는 pair_df가 만들어진 직후):

```python
pair_df = build_classifier_pair_table(df, avg_early_ord)
if pair_df.empty:
    print("[오류] 페어를 생성할 수 있는 경주가 없습니다")
    return
pair_df = add_race_relative_features(pair_df)
```

그 외 `train_and_evaluate`의 로직(dev/test 분리, GroupKFold, 최종 학습,
피처 중요도, test 평가)은 1차와 동일하되, `X_dev`/`X_test`가
`FEATURE_COLUMNS`(13개)를 참조하므로 자동으로 새 피처 3개를 포함한다.

## 에러 처리/엣지케이스

- `race_max.clip(lower=1e-6)`로 0 나누기 방지(기존 `win_prob_ratio`와 같은
  관례).
- 동률 처리: `rank(method="average")`를 명시적으로 써서, p_indep이 완전히
  같은 페어가 여러 개 있어도 순위가 결정적으로 계산된다(행 순서에 의존
  하지 않음) — 이번 설계가 해결하려는 "경주 내 상대 순위" 개념 자체이므로,
  순위 계산에서 또 다른 순서 의존 버그를 만들지 않도록 명시한다.
- `pair_df`가 비어있으면(이론상 `build_classifier_pair_table`이 빈
  DataFrame을 반환할 수 있음, 1차 코드 품질 리뷰에서 이미 확인된 사실)
  `train_and_evaluate`가 이미 그 전에 조기 반환하므로
  `add_race_relative_features`가 빈 입력을 받는 경우는 실제로 발생하지
  않는다. 그래도 함수 자체는 빈 입력에 대해 예외 없이 빈 출력을
  반환해야 한다(groupby/rank/transform이 빈 DataFrame에서도 안전하게
  동작하는 pandas 표준 동작에 의존).

## 테스트 방침

`add_race_relative_features`는 순수 함수이므로
`tests/test_diagnose_pair_classifier.py`에 합성 데이터로 단위테스트를
추가한다:

- `p_indep_rank_in_race`가 경주 내에서 올바르게 계산되는지(가장 큰
  p_indep을 가진 페어가 순위 1)
- `p_indep_ratio_to_max`가 그 경주 1등 페어에서 정확히 1.0인지, 나머지는
  0~1 범위인지
- `n_pairs_in_race`가 그 경주의 실제 페어 개수와 일치하는지
- 동률 케이스(두 페어의 p_indep이 완전히 같을 때) 순위가 평균값으로
  나오고 순서에 의존하지 않는지(입력 행 순서를 바꿔도 같은 결과)
- 서로 다른 두 경주의 값이 섞이지 않는지(경주 A의 순위/비율이 경주 B
  페어 수에 영향받지 않는지)

`train_and_evaluate`는 기존 관례대로 단위테스트 대상이 아니다(DB/모델
의존, 실행+결과 해석으로 검증).

## 판정 기준

- **test 최고 페어 픽 적중률이 1차(38.5%)보다 오르고 40.0%(오프셋
  로지스틱)를 넘으면** → 경주-상대 피처가 학습-평가 간극을 메운 것.
  이 방향(피처 엔지니어링)을 더 밀고 나갈 근거 확보.
- **38.5%보다는 올랐지만 40.0% 미달이면** → 방향은 맞지만 아직 부족.
  피처를 더 추가하거나(예: win_prob 기반 경주-상대 피처), 랭킹
  목적함수 전환(브레인스토밍 "방향 2")을 다음 후보로 고려.
- **38.5% 근처거나 그대로면** → 이 진단(학습-평가 목표 불일치)이 주된
  원인이 아니었다는 뜻. 피처 중요도에서 새 3개 피처가 실제로 얼마나
  쓰였는지(0에 가까우면 가설 자체가 기각) 같이 확인해 기록한다.
- 세 경우 모두 피처 중요도 전체 순위를 1차와 나란히 비교해 보고한다.

## 범위 외 (이번에 하지 않음)

- 하이퍼파라미터 튜닝(Optuna)
- 랭킹 목적함수(`rank:pairwise` 등)로 전환
- 앙상블
- `core/predictor.py`/`core/ev_calculator.py`/`core/investor.py` 연결
- ROI/EV 계산
- `build_classifier_pair_table`/`_load_predictions_with_probs` 자체 수정
