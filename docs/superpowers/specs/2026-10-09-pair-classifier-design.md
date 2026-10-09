# 복연승 페어 분류기 설계 (후보 1)

## 배경

2026-10-09 페어 상호작용 스크리닝(`scripts/diagnose_pair_interaction.py`,
오프셋 로지스틱 회귀 + 경주 단위 클러스터 부트스트랩)에서 "두 말이 동시에
top3에 들 확률"이 독립가정(top3_prob_i × top3_prob_j)만으로 설명되지 않고,
초반 페이스 상호작용으로 설명되는 잔차가 통계적으로 유의미하게 남는다는
결과가 나왔다(b1 both_early_leader [0.06, 0.46], b2 entry_gap
[-0.05, -0.03], b3 pace_pressure [-0.06, -0.02], 전부 95% CI가 0 배제).
test 최고 페어 픽 적중률도 40.0%(408경주)로 기존 휴리스틱(38.5%)·
Harville(36.5%)·무작위(7.2%)를 전부 상회했다. 자세한 내용은
[[project_multi_bet_probability_model]] 참고.

이 결과는 이 프로젝트가 지금까지 시도한 모든 모델 개선 방향(Harville,
경주 확신도 필터, 신규 신호 피처 3종, 누수 제거 재학습) 중 처음으로
"개선 신호 있음"으로 나온 결과다. 다만 오프셋 로지스틱은 세 보정항을
**더하기만** 하는 선형 구조라 상호작용을 단순하게만 반영한다. 이 신호를
더 유연한 모델(그레이디언트 부스팅)로 최대한 뽑아낼 수 있는지 확인하는
것이 이번 설계의 목표다(quant-model-expert가 2026-09-29에 제안한 "후보
1: 페어 전용 이진분류기").

## 목표

경주 내 모든 유효 말 쌍(i, j)을 한 행으로 재구성하고, 라벨(둘 다 top3)을
직접 학습하는 이진분류기를 만들어, 스크리닝에서 이미 확인된 40.0%(test
최고 페어 픽 적중률)를 넘어서는지 확인한다.

## 결정된 사항

- **새 의존성 추가 안 함.** `xgboost`는 이미 프로젝트 의존성(`core/trainer.py`가
  씀). `sklearn.model_selection.GroupKFold`도 이미 설치된 scikit-learn에
  포함.
- **기존 스크리닝 스크립트(`scripts/diagnose_pair_interaction.py`)는
  수정하지 않는다.** 이미 승인·실행 완료된 코드다. 거기 있는 leak-safe
  헬퍼(`fetch_avg_early_ord_asof`)는 **import해서 재사용**한다(수정 아님,
  읽기 전용 참조).
- **`scripts/_backtest_common.py`, `core/feature_builder.py`도 수정하지
  않는다.** 기존 패턴 그대로 — 모델/데이터 로딩 로직은 이 작업 전용으로
  독립 재구현한다(코드 몇 줄 중복되지만 공유 파일 훼손 위험을 없앤다 —
  이 파일들을 서브에이전트가 실수로 훼손한 전례가 이 프로젝트에 여러
  번 있었다).
- **하이퍼파라미터 튜닝(Optuna) 안 함.** 고정값을 쓴다(아래 "모델"
  절). 이 프로젝트의 기존 원칙("피처 중요도/안정성 분석처럼 재학습이
  필요한 진단은 무거운 단계를 생략하고 가볍게 돌려도 된다")을 따른다.
- **앙상블 안 함.** XGBoost 단일 모델만. 프로덕션 모델(RF+LR+XGB+LGBM
  4종 앙상블)과 다르다 — 이번은 "신호가 있는지, 얼마나 있는지"를 보는
  진단이라 앙상블의 안정화 효과가 필요 없고, 속도가 더 중요하다.
- **프로덕션 연결 없음.** `core/predictor.py`/`core/ev_calculator.py`/
  `core/investor.py` 수정 없음. ROI/EV 계산 없음.
- **검증 창은 스크리닝과 동일하게 유지한다.** dev는
  `CALIBRATION_EVAL_CUTOFF`(2026-05-01) 이전, test는 기존 408경주
  벤치마크 창(2026-05-01~06-25). 그래야 이번 결과(목표: 40.0% 돌파
  여부)를 스크리닝 결과와 직접 비교할 수 있다.

## 신규 파일

| 파일 | 내용 |
|---|---|
| `scripts/diagnose_pair_classifier.py` (신규) | 아래 전체 파이프라인을 담는 단일 스크립트. `scripts/diagnose_pair_interaction.py`와 같은 "진단용, 프로덕션 미연결" 관례를 따른다. |

## 데이터 파이프라인

1. **예측값 로드** (`_load_predictions_with_probs(start_date, end_date)`):
   `scripts/diagnose_pair_interaction.py`의 `_load_predictions_with_ids`와
   동일한 순서(모델 로드 → `load_historical_data` → `build_for_training`
   → `preprocessor.transform`)를 따르되, **두 확률을 모두** 계산해
   보존한다: `trainer.predict_proba(X)`(캘리브레이션된 win_prob)와
   `trainer.predict_proba_top3(X, raw=True)`(raw top3_prob,
   [[project_top3_no_blend]] 원칙 유지). `horse_id`/`entry_number`도
   유지한다(이미 Task 5에서 `build_for_training` 이후에도 두 컬럼이
   살아있음이 실측으로 확인됨, 재확인 불필요).

   반환 컬럼: `race_id`, `race_date`, `horse_id`, `entry_number`,
   `final_rank`, `win_prob`, `top3_prob`.

2. **과거 평균 초반통과순위**: `scripts.diagnose_pair_interaction`에서
   `fetch_avg_early_ord_asof`를 import해서 그대로 재사용한다(leak-safe
   로직 중복 구현하지 않음).

3. **페어 피처 테이블 생성** (`build_classifier_pair_table(df,
   avg_early_ord, early_leader_threshold=3.0)`):
   경주별로 유효 착순 말(`Utils.ranks.valid_rank_mask` 재사용)만 대상으로
   3두 이상인 경주의 모든 쌍(i, j)을 만들어 다음 피처를 계산한다:

   | 피처 | 정의 |
   |---|---|
   | `win_prob_i`, `win_prob_j` | 각 말의 캘리브레이션된 단승확률 |
   | `top3_prob_i`, `top3_prob_j` | 각 말의 raw 연승확률 |
   | `p_indep` | `top3_prob_i * top3_prob_j` |
   | `win_prob_diff` | `abs(win_prob_i - win_prob_j)` |
   | `win_prob_ratio` | `min(win_prob_i, win_prob_j) / max(win_prob_i, win_prob_j, 1e-6)` (0~1, 대칭 비율 — 어느 쪽이 i/j인지에 관계없이 "얼마나 비슷한 세기인가"를 나타냄) |
   | `both_early_leader` | 둘 다 과거 평균 초반순위 ≤ 3 |
   | `entry_gap` | `abs(entry_number_i - entry_number_j)` |
   | `pace_pressure` | 그 경주에서 "초반형"으로 분류되는 말의 수 |
   | `label` | `final_rank_i<=3 and final_rank_j<=3` |

   3두 미만인 경주는 페어가 생성되지 않는다(스크리닝과 동일 기준,
   자동 스킵·예외 없음). 과거 기록 없는 말(avg_early_ord=NaN)은
   `both_early_leader=False`로 중립 처리(스크리닝과 동일 관례).

## 모델

```
XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)
```

고정값이며 튜닝하지 않는다(값은 이 프로젝트의 기존 튜닝된 XGBoost
하이퍼파라미터(`models/best_params.json`)와 비슷한 규모로 잡은 것으로,
임의로 고른 값이 아니다 — 다만 이번 피처 공간에 맞춰 재튜닝한 것은
아니라는 점은 명시한다).

## 검증

- **dev/test 분리**: dev(≤2026-04-30)로 학습, test(2026-05-01~06-25,
  408경주)로 딱 한 번 평가.
- **dev 내 교차검증**: `GroupKFold(n_splits=5)`를 `race_id` 기준으로
  적용해 "페어 단위로 무작위 분할하면 같은 경주의 다른 페어가 양쪽에
  섞여 들어가 과적합을 숨긴다"는 문제를 피한다(Task 4의 경주 단위
  부트스트랩과 같은 원칙). 폴드별 정확도/로그손실을 보고한다.
- **최종 평가**: dev 전체로 재학습한 모델을 test에 적용해, 경주별로
  `predict_proba`가 가장 높은 페어를 픽으로 삼아 적중률을 낸다. 기존
  기준(무작위 7.2%, 휴리스틱 38.5%, Harville 36.5%, 오프셋 로지스틱
  40.0%)과 나란히 비교한다.
- **피처 중요도**: `model.feature_importances_`를 출력한다. 추가
  실험 없이 "어떤 피처가 실제로 기여하는가"를 바로 확인한다.

## 에러 처리/엣지케이스

`scripts/diagnose_pair_interaction.py`와 동일한 가드를 그대로 적용한다:
3두 미만 경주 자동 스킵, `win_prob_ratio` 계산 시 0으로 나누기 방지
(`max(..., 1e-6)`), 모델/데이터 없으면 조기 반환 + 한국어 오류 메시지,
dev 또는 test가 비면 조기 반환.

## 테스트 방침

- `build_classifier_pair_table`는 순수 함수(DB 호출 없음)이므로
  `tests/test_diagnose_pair_classifier.py`에 합성 데이터로 단위테스트를
  작성한다(`build_pair_table`의 기존 테스트 패턴을 따름): 페어 수가
  C(유효말수,2)와 일치하는지, `win_prob_ratio`가 0~1 범위이자 대칭인지
  (i/j를 바꿔도 같은 값), `win_prob_diff`가 절댓값인지, 3두 미만 경주가
  제외되는지, NaN `avg_early_ord`가 중립 처리되는지.
- 모델 로드·DB 접근이 필요한 로더와 학습·평가 파이프라인은 기존
  관례대로 단위테스트 대상이 아니다. 실DB 스모크 테스트로 검증한다.

## 판정 기준

- **test 최고 페어 픽 적중률이 40.0%(기존 오프셋 로지스틱 결과)를
  넘으면** → 더 유연한 모델이 선형 보정 이상의 신호를 더 뽑아낸 것.
  다음 단계(배당 데이터 확보 후 ROI 검증, 또는 캘리브레이션)를 고려할
  근거 확보.
- **40.0% 근처거나 못 미치면** → 오프셋 로지스틱이 이미 신호를 거의 다
  뽑아낸 것. 이 방향에서 추가 모델 복잡도는 불필요 — "개선 없음(오프셋
  로지스틱 수준이 한계)"으로 기록.
- 두 경우 모두 피처 중요도를 같이 보고해, 어떤 피처가 실제로 기여하는지
  기록한다.

## 범위 외 (이번에 하지 않음)

- 하이퍼파라미터 튜닝(Optuna), 앙상블
- `core/predictor.py`/`core/ev_calculator.py`/`core/investor.py` 연결
- ROI/EV 계산, 복연승 배당 데이터 수집
- `scripts/diagnose_pair_interaction.py`/`scripts/_backtest_common.py`/
  `core/feature_builder.py` 수정
