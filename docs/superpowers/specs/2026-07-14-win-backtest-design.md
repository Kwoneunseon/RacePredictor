# 단승(win) 백테스트 스크립트 — 설계

## 배경

연승(top3) 예측은 `scripts/backtest_top3.py`로 정기적으로 검증되고 있고 (`TOP3_PROB_THRESHOLD=0.65` 기준 79.9%/69.6% 커버리지, [[project_next_task]]), 이 결과를 근거로 임계값을 튜닝해왔다.

반면 단승(win) 예측은 정식 백테스트가 없다. `docs/progress/2026-06-08-prediction-check.md`의 4경주 수기 점검(0/4 적중)이 유일한 기록이며, 표본이 너무 작아 신뢰할 수 없다. 사용자가 "단승·연승 정확도를 동등하게 개선하고 싶다"고 확인했고, 개선 여부를 판단하려면 먼저 측정 가능한 기준선이 필요하다.

이 설계는 정확도 개선 로드맵의 **1단계(측정 기반 마련)**만 다룬다. 새 피처(코너별 통과순위 등) 추가나 모델 캘리브레이션 점검은 이 결과를 본 뒤 별도 설계로 진행한다.

## 범위

- 신규 스크립트 `scripts/backtest_win.py` 추가
- `scripts/backtest_top3.py`와 중복되는 로딩/예측 로직(모델 로드 → 과거 데이터 로드 → 피처 빌드 → 전처리 → 예측)을 `scripts/_backtest_common.py`로 추출해 공유
- 기존 `backtest_top3.py`는 동일한 출력을 내도록 공유 모듈을 사용하게 최소 수정

## 컴포넌트

### `scripts/_backtest_common.py` (신규)

- `load_historical_data(start_date, end_date) -> pd.DataFrame`
  기존 `backtest_top3.py`의 동일 함수를 그대로 이동.
- `run_predictions(start_date, end_date) -> pd.DataFrame | None`
  모델 로드 → 데이터 로드 → 피처 빌드 → 전처리 → `win_prob`/`top3_prob` 예측까지 수행하고, `race_id, race_date, final_rank, top3_prob, win_prob, n_horses, prev_win_odds`를 담은 결과 DataFrame을 반환. 모델이 없거나 데이터가 없으면 `None` 반환(에러 메시지는 내부에서 출력).
  - `prev_win_odds`를 결과에 포함하는 이유: 단승 백테스트의 "인기마 기준선" 비교에 필요.

### `scripts/backtest_top3.py` (수정)

- 자체 정의했던 `load_historical_data`와 1~6단계 로직을 `_backtest_common.run_predictions()` 호출로 교체.
- 7단계 이후(두수 필터, 임계값 스윕, 조합 분석, 80% 권고)는 기존 그대로 유지 — 순수 출력 로직이라 변경 없음.

### `scripts/backtest_win.py` (신규)

`run_predictions()` 결과를 받아 경주별로 `win_prob` 최댓값 픽 선정 후:

1. **전체 Precision@1**: `final_rank == 1`인 비율
2. **`win_prob` 임계값별 적중률**: `[0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]` 스윕, 각 구간 경주수·적중률·전체비율 (기존 `BET_PROB_THRESHOLD=0.50` 위치 확인용)
3. **두수 필터별 적중률**: top3 스크립트와 동일 포맷 (`[999, 12, 10, 9, 8, 7, 6]`)
4. **인기마 기준선 비교**: 같은 경주 집합에서 `prev_win_odds`가 가장 낮은(가장 인기있는) 말을 찍었을 때의 적중률과 나란히 출력 — 모델이 "그냥 인기마 찍기"보다 나은지 확인
5. CLI: `--start`(기본 2025-01-01) `--end`(기본 오늘 날짜 대응 위해 top3 스크립트와 동일하게 인자로 받음)

## 데이터 흐름

```
race_entries(final_rank 있음) → RaceFeatureBuilder.build_for_training()
    → RacePreprocessor.transform() → trainer.predict_proba() / predict_proba_top3()
    → result_df (race_id, final_rank, win_prob, top3_prob, n_horses, prev_win_odds)
    → [backtest_top3.py 요약] / [backtest_win.py 요약]
```

## 에러 처리

기존 `backtest_top3.py`와 동일한 수준 유지: 모델 없음/데이터 없음이면 메시지 출력 후 조기 반환. 백테스트 스크립트는 대화식 분석 도구이므로 예외 처리를 과도하게 추가하지 않는다.

## 테스트

`backtest_top3.py`와 동일하게 실제 DB·저장된 모델에 의존하는 분석 스크립트라 단위 테스트 대상이 아니다 (기존 스크립트들도 `tests/`에 대응 테스트 없음). 실행 결과를 사람이 직접 확인하는 방식을 유지한다.

## 완료 기준

- `python scripts/backtest_win.py` 실행 시 전체 Precision@1, 임계값별/두수별 적중률, 인기마 기준선 비교가 출력된다.
- `python scripts/backtest_top3.py` 실행 결과가 리팩터링 전후 동일하다(회귀 없음).
