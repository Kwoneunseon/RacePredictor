# Harville 연승/복연승 비교 백테스트 설계

## 배경

[[project_multi_bet_probability_model]](`docs/superpowers/specs/2026-07-25-multi-bet-probability-model-design.md`)에서
확정한 Phase 2 롤아웃 절차 중 1단계(`core/rank_distribution.py` — Harville
순차소거 방식으로 연승/복연승 확률 유도)는 이미 구현되어 있다
(2026-08-05, brute-force 대조 검증 포함 15개 단위테스트 통과). 그러나 이
모듈은 아직 어디에도 연결되지 않았고, 원래 절차의 2~3단계(기존 연승
모델과의 비교, 복연승 자체 정확도 검증)가 미착수 상태였다.

이 문서는 2026-09-17 브레인스토밍 세션에서 그 2~3단계만 구체화한 것이다.
4단계(복연승 배당 확보 후 EV/켈리 활성화)와 `predictor.py`/`ev_calculator.py`/
`investor.py` 연결은 이번 범위에 포함하지 않는다 — 결과를 보고 별도
세션에서 결정한다.

## 목표

1. Harville로 유도한 연승확률이 기존에 검증된 연승 전용 classifier(408경주
   백테스트, `TOP3_PROB_THRESHOLD=0.65`)와 **같거나 나은지** 동일 벤치마크
   창에서 비교한다.
2. 지금까지 임시 휴리스틱(`scripts/backtest_quinella_place_heuristic.py` —
   top3_prob/win_prob 상위 2마리 픽)으로만 검증됐던 복연승 예측을, Harville
   조합확률로 대체했을 때 정확도가 더 나은지 확인한다.

두 경우 모두 **적중률/커버리지 비교만 하며, 배당(odds) 기반 EV/ROI는
계산하지 않는다** — 필요한 입력(과거 경주 결과, 기존 모델의 win_prob)이
이미 DB와 학습된 모델에 있으므로 추가 데이터 수집이 필요 없다.

## 결정된 사항 (2026-09-17 논의)

- **캘리브레이션 생략.** 원 스펙의 방법론에는 Harville 확률에 isotonic
  보정을 적용하라고 되어 있지만, 이는 EV 계산(확률의 절대값 정확도가
  중요) 목적이다. 이번 라운드는 임계값 스윕을 통한 순서/커버리지 비교만
  하므로 확률의 상대적 순서만 유효하면 충분하다. `core/trainer.py` 수정과
  재학습 없이 진행한다. 캘리브레이션은 4단계(EV 연결) 착수 시점으로
  미룬다.
- **강도 점수는 `win_prob`(캘리브레이션 적용된 기존 값) 그대로 사용.**
  `scripts/_backtest_common.py`의 `run_predictions()`가 반환하는 값을
  그대로 쓰며, `core/predictor.py`의 랭커 블렌딩·softmax 정규화 파이프라인은
  거치지 않는다 — 기존 `backtest_top3.py`/`backtest_quinella_place_heuristic.py`와
  동일한 관례를 유지해 기존 408경주 벤치마크 수치와 직접 비교 가능하게
  한다.
- **벤치마크 창은 새로 정의하지 않는다.** 기존 TEST 구간
  (`2026-05-01~2026-06-25`, 408경주)을 그대로 재사용한다.

## 신규/수정 컴포넌트

| 파일 | 변경 내용 |
|---|---|
| `scripts/_backtest_common.py` | `attach_harville_top3(result_df) -> pd.DataFrame`: 경주별(`race_id`) group의 `win_prob`을 강도점수로 `core.rank_distribution.top3_marginal_probabilities()` 호출, 말별 `harville_top3_prob` 컬럼을 추가해 반환. `build_harville_pair_table(result_df) -> pd.DataFrame`: 경주별로 `pair_top3_probabilities()`를 호출해 확률이 가장 높은 페어를 선정하고 적중 여부·확률을 담은 경주 단위 테이블 반환(`build_quinella_place_heuristic.py`의 `build_quinella_table`과 동일 골격, 픽 방식만 휴리스틱 대신 Harville 조합확률 최댓값으로 대체). |
| `scripts/backtest_harville_top3.py` (신규) | `backtest_top3.py`와 동일한 리포트 구조(두수 필터별/임계값별/조합별 적중률표, 80% 달성 조건 탐색)를 `harville_top3_prob` 기준으로 재실행. 콘솔 출력에 기존 raw classifier 벤치마크(0.65 임계값 → 약 80% 적중/61.6% 커버리지, 408경주, [[project_next_task]] 기록값)를 나란히 인용해 비교. |
| `scripts/backtest_harville_pair_top3.py` (신규) | `backtest_quinella_place_heuristic.py`와 동일 구조(dev/test 분리, 두수 분포, 무작위 베이스라인, 임계값 스윕)로 Harville pair 확률을 검증. 기존 휴리스틱 결과(무작위 대비 5.36배, 38.5% 적중, 408경주, [[project_multi_bet_probability_model]] 기록값)와 3자 비교(무작위/기존 휴리스틱/Harville). |
| `tests/test_backtest_common.py` (신규 또는 기존에 추가) | `attach_harville_top3`/`build_harville_pair_table`에 대해 합성 소규모 경주(3~5두)로 불변식 검증: 경주 내 `harville_top3_prob` 합 ≈ 3, pair 확률 행렬 대칭성, 3두 미만 경주 스킵 동작. |

## 데이터 흐름

```
run_predictions(start, end)          # 기존 함수, 무수정
  → 말 단위 DataFrame
    (race_id, race_date, win_prob, top3_prob, final_rank, n_horses, ...)
  → attach_harville_top3()           # race_id group → rank_distribution 호출
    → harville_top3_prob 컬럼 추가
  → backtest_harville_top3.py: 기존 backtest_top3.py와 동일한 집계/스윕

run_predictions(dev_start, test_end) # 위와 동일 소스
  → build_harville_pair_table()      # race_id group → pair_top3_probabilities
    → 경주 단위 최적 페어/적중여부 테이블
  → backtest_harville_pair_top3.py: 기존 heuristic 스크립트와 동일한 리포트 구조
```

## 판정 기준

- **연승**: 동일 408경주 창에서 Harville 유도 확률이 기존 raw classifier와
  같거나 나은 적중률/커버리지 트레이드오프를 보이면, `core/predictor.py`의
  `top3_probability` 산출을 Harville 기반으로 교체하는 것을 다음 라운드
  작업으로 제안한다. 열등하면 기존 classifier를 유지하고 그 결과를
  memory에 기록한다(이 프로젝트는 "개선 없음"도 유효한 결과로 취급하고
  기록해왔다, [[project_new_signal_features_rollout]] 등).
- **복연승**: 기존 검증된 모델이 없으므로, Harville이 무작위 베이스라인과
  기존 휴리스틱보다 나은 정확도를 보이면 향후(4단계) EV 연결 대상으로
  채택 후보가 된다. 이번 라운드에서는 채택 여부만 결론 내리고 실제 연결은
  하지 않는다.

## 에러 처리/엣지케이스

- 출전마 3두 미만 경주: 연승/복연승 정의 자체가 불가능 → 스킵하고 스킵
  건수를 로그로 남긴다(`backtest_quinella_place_heuristic.py`의
  `skipped_small` 패턴 재사용).
- 그룹 내 `win_prob` 합이 0 이하인 경주(이론상 극히 드묾): `rank_distribution
  ._normalize()`가 `ValueError`를 던지므로 해당 경주만 try/except로 스킵하고
  경고 카운트를 남긴다 — 전체 백테스트를 중단시키지 않는다.

## 테스트 방침

- `core/rank_distribution.py` 자체는 기존 15개 단위테스트로 충분(수정 없음).
- 신규 헬퍼 2개(`attach_harville_top3`, `build_harville_pair_table`)만
  합성 데이터로 가볍게 단위테스트.
- 백테스트 스크립트 자체(리포트 출력 로직)는 기존 관례대로 단위테스트
  대상이 아니다 — 실행 결과를 사람이 검토한다.

## 범위 외 (이번 세션에 포함하지 않음)

- `core/predictor.py`/`core/ev_calculator.py`/`core/investor.py` 연결
- Harville 확률에 대한 isotonic 캘리브레이션 (4단계로 이연)
- 복연승 배당 데이터 확보/EV 활성화 (별도 세션, 서울 연승 배당 소스
  미해결 문제와도 연동됨, [[project_race_payouts_scraper]])
- 신규 데이터 수집 (기존 DB 데이터로 충분)
