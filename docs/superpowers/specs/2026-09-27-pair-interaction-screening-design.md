# 복연승 페어 상호작용 신호 스크리닝 설계

## 배경

Harville 조합확률(연승 36.5%)과 기존 임시 휴리스틱(top3_prob 상위 2마리
묶기, 38.5%)은 둘 다 이미 검증된 개별 말 확률(win_probability/
top3_probability)을 재조합만 한 것이라, 원재료 이상의 정보를 만들지
못한다([[project_multi_bet_probability_model]]). "말 간 상호작용(예:
같은 선행형 말 두 마리가 초반에 서로 견제해서 둘 다 손해)"을 반영하면
이 한계를 넘을 수 있는지가 이번 세션의 논의 주제였다.

quant-model-expert가 브레인스토밍한 3개 후보 중, 사용자가 **후보
2(잔차보정/오프셋 모델)를 먼저 진행**하기로 결정했다. 이건 본격적인
페어 전용 분류기(후보 1)를 만들기 전에, "상호작용 신호가 통계적으로
존재하는가"를 저비용으로 먼저 스크리닝하는 단계다.

## 목표

"두 말이 동시에 top3에 들 확률"이 독립가정
(`p_indep = top3_probability_i × top3_probability_j`)만으로 설명되는지,
아니면 초반 페이스 경쟁 관련 상호작용으로 설명되는 잔차가 통계적으로
유의미하게 남는지 확인한다. 유의미한 잔차가 있으면 후보 1(페어 전용
분류기)에 착수할 근거가 되고, 없으면 이 방향 전체를 접고 "개선 없음"으로
기록한다.

## 결정된 사항

- **새 의존성 추가 안 함.** 원래 통계 라이브러리(statsmodels)로 오프셋
  로지스틱 회귀를 적합할 계획이었으나, 이 프로젝트에 설치돼 있지 않다.
  대신 이미 이 프로젝트 전체에서 일관되게 써온 방식(경주 단위 클러스터
  부트스트랩으로 신뢰구간을 내는 것)을 그대로 적용해, `scipy.optimize`로
  직접 로그우도를 최소화하는 방식으로 적합한다.
- **`scripts/_backtest_common.py`는 수정하지 않는다.** 이 파일은
  사용자의 별개 진행 중 작업(미커밋)이 있고, 과거 서브에이전트가 이
  파일을 두 번 실수로 훼손한 전례가 있다. 필요한 로딩 로직(모델
  예측값을 `horse_id`/`entry_number`와 함께 유지)은 새 스크립트 안에
  독립적으로 둔다 — 코드 몇 줄이 중복되지만 위험을 없앤다.
- **`core/feature_builder.py`도 수정하지 않는다.** 기존
  `_fetch_running_style_for_training`과 동일한 leak-safe as-of 집계
  로직(말별로 해당 경주 이전 기록만 사용)을 새 스크립트에서 재구현하되,
  "델타"(초반순위-최종순위)가 아니라 "초반순위 원값의 과거 평균"을
  계산한다는 점만 다르다.
- **프로덕션 연결·`config.py` 수정·ROI 계산 없음.** 이번 라운드는 순수
  스크리닝이다.

## 신규 파일

| 파일 | 내용 |
|---|---|
| `scripts/diagnose_pair_interaction.py` (신규) | 아래 파이프라인 전체를 담는 단일 스크립트. 기존 `scripts/diagnose_calibration.py`와 같은 "진단용, 프로덕션 미연결" 관례를 따른다. |

## 데이터 파이프라인

1. **예측값 로드** (`_load_predictions_with_ids(start_date, end_date)`):
   `scripts/_backtest_common.py`의 `run_predictions()`와 동일한 순서
   (모델 로드 → `load_historical_data` → `build_for_training` →
   `preprocessor.transform` → `predict_proba`/`predict_proba_top3(raw=True)`)를
   따르지만, 결과 DataFrame에 `horse_id`와 `entry_number`를 추가로
   보존한다.
2. **과거 평균 초반순위** (`_fetch_avg_early_ord_asof(horse_ids, race_dates)`):
   `race_entry_detail.sj_1c_ord`(없으면 `sj_3c_ord`로 대체)와
   `race_entries.final_rank`(`valid_rank_mask`로 90번대/0 제외)를 조회해,
   말별로 **해당 경주 날짜 이전 기록만** 써서 초반순위 원값의 평균을
   계산한다. 과거 기록이 없으면 `None`(중립, "초반형 아님"으로 취급).
3. **페어 테이블 생성** (`build_pair_table(df, early_ord_map)`):
   경주별로 유효 착순 말(`valid_rank_mask`)만 대상으로 모든 쌍(i,j)을
   만들어:
   - `p_indep = top3_prob_i * top3_prob_j`
   - `both_early_leader`: 둘 다 과거 평균 초반순위 ≤ 3인지(불리언, EDA에서
     확인한 임계값과 동일)
   - `entry_gap = |entry_number_i - entry_number_j|`
   - `pace_pressure`: 그 경주에서 "초반형"(과거 평균 초반순위 ≤ 3)으로
     분류되는 말의 총 수(경주 레벨 공변량, 같은 경주의 모든 페어 행에
     동일 값 반복)
   - `label = valid_rank_mask 통과 && final_rank_i<=3 && final_rank_j<=3`
   3두 미만 경주는 페어가 없으므로 자동으로 제외된다.

## 모델

```
logit(p_joint) = logit(p_indep) + b0 + b1*both_early_leader + b2*entry_gap + b3*pace_pressure
```

`p_indep`은 고정 오프셋(학습 대상 아님), `b0~b3`만 `scipy.optimize.minimize`로
음의 로그우도를 최소화해 추정한다. 파라미터 수가 4개뿐이라 최적화는
가볍다.

## 검증

- **dev/test 분리**: dev = `CALIBRATION_EVAL_CUTOFF`(2026-05-01) 이전,
  test = 기존 408경주 벤치마크 창(2026-05-01~06-25). dev에서 `b0~b3`를
  적합하고, test에서는 그 고정된 계수로 예측만 한다(재적합 없음).
- **유의성 검정**: dev 데이터를 **경주 단위로** 재표본추출하는 부트스트랩
  (n=1000~2000회, 기존 프로젝트 백테스트들과 동일한 방식)으로 `b1~b3`
  각각의 신뢰구간을 낸다. 신뢰구간이 0을 포함하면 "유의하지 않음"으로
  판정.
- **정확도 비교**: test에서 `p_joint`가 가장 높은 페어를 픽으로 삼아,
  기존 벤치마크(무작위 7.2%, 휴리스틱 38.5%, Harville 36.5%)와 동일한
  방식(적중률, 임계값 스윕)으로 나란히 비교한다.

## 에러 처리/엣지케이스

- 출전마 3두 미만 경주: 페어 자체가 없으므로 자동 제외(예외 없음).
- 과거 기록이 없는 말(신마 등): `both_early_leader=False`로 중립 처리
  (기존 `running_style_score`의 `fillna(0.0)` 관례와 같은 방향).
- `p_indep`이 0 또는 1에 가까워 `logit` 계산이 불안정한 경우:
  `p_indep`을 `[1e-6, 1-1e-6]`으로 클리핑한다.

## 테스트 방침

이 스크립트는 기존 `scripts/backtest_*.py`/`diagnose_*.py`와 같은
진단용 스크립트로, 그 관례를 따라 **단위테스트 대상이 아니다**. 다만
페어 테이블 생성 직후 스크립트 자체에서 가벼운 정합성 assert(경주별
페어 수 == `C(유효말수, 2)`)를 두어, 조용한 인덱싱 버그를 실행 시점에
바로 잡아낸다.

## 판정 기준

- `b1~b3` 중 하나 이상이 부트스트랩 95% 신뢰구간에서 0을 배제하면 →
  "상호작용 신호 있음, 후보 1(페어 전용 분류기) 착수 근거 확보"로 결론.
- 전부 0을 포함하면 → "개선 없음"으로 기록하고 이 방향 종결, 다음
  세션에서 배당 데이터 축적/ROI 쪽으로 리소스 전환을 제안.

## 범위 외 (이번에 하지 않음)

- 후보 1(페어 전용 분류기), 후보 3(contextual Bradley-Terry)
- `core/predictor.py`/`core/ev_calculator.py`/`core/investor.py` 연결
- `config.py`/`core/feature_builder.py` 수정
- ROI/EV 계산
- statsmodels 등 신규 의존성 추가
