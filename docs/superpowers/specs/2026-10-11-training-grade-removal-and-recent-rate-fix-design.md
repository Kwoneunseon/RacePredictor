# training_grade_enc 제거 + recent_win_rate 학습경로 날짜정합 수정 설계

## 배경

2026-10-11 FEATURE_COLUMNS 27개 전수 점검(db-expert)에서 두 가지 결함을
발견했다(상세: `docs/model-data-reference-2026-10-11.md`,
`db_collector_api_outage_and_pitfalls.md`):

1. **`training_grade_enc`가 사실상 죽은 피처다.** 이 피처의 원천 테이블
   `horse_training_sessions`를 채우는 수집기가 코드베이스에 없다
   (`migrations/002_retroactive_new_signal_tables.sql`로 한 번 소급
   백필만 된 테이블). 최신 데이터는 2026-02-26이고 그 뒤로 0건 —
   dev 구간 후반부와 test 구간(2026-05~06) 전체가 항상 기본값
   1.0("정보없음")만 받는다. 조인 로직(말이름 텍스트 매칭) 자체는
   건강하다(96.1% 매칭) — 문제는 조인 대상 데이터가 없다는 것.

2. **`jockey_recent_win_rate`/`trainer_recent_win_rate`의 학습 경로와
   예측 경로가 서로 다른 기준으로 "과거"를 자른다.** 학습 경로
   (`core/preprocessor.py::_add_rolling_recent_rates`)는 날짜순 정렬 후
   **행 위치**(`wins[:i]`)로 자르는데, 같은 날 여러 경주가 있으면 같은
   날 안에서 먼저 정렬된 경주까지 "과거"로 포함시킨다. 예측 경로
   (`core/feature_builder.py::_fetch_recent_rates`)는 `race_date < 당일`
   로 그날 전체를 제외한다. test 창(408경주, 4,019행)에서 직접 대조한
   결과 65.3%가 값이 다르고 26.3%는 차이가 0.01을 초과한다(train-serve
   skew).

## 결정된 사항

- **범위는 코드 수정만.** 재학습(`python main.py --mode train`)은 이번
  범위에 포함하지 않는다(사용자 지시). deleted 컬럼 연결 작업은 이미
  완료되어 있으나, 재학습·재평가는 별도 후속 작업으로 분리한다.
- **① `training_grade_enc`는 완전히 제거한다**(결측 플래그로 남기거나
  수집기를 새로 만드는 대안은 채택하지 않음). `FEATURE_COLUMNS`에서
  빼고, 이 피처를 채우던 모든 코드(함수·호출부·테스트)를 삭제한다.
  `horse_training_sessions` 테이블 자체나 스키마는 건드리지 않는다
  (테이블은 남아있지만 더 이상 어떤 코드도 참조하지 않게 된다).
- **② 학습 경로를 예측 경로와 동일한 날짜 기준으로 통일한다**(반대
  방향, 즉 예측 경로를 학습 경로 방식에 맞추는 것은 채택하지 않음 —
  "당일 다른 경주 결과까지 보고 계산"하는 쪽이 미래정보 누수에 더
  가깝기 때문). `_fetch_recent_rates`(예측 경로) 자체는 이미 올바르므로
  수정하지 않는다.
- 다른 25개 피처(이미 정상 확인됐거나 별도 후속 작업으로 분리된 것들)
  는 이번 범위에 포함하지 않는다.

## ① `training_grade_enc` 제거

**Files:**
- Modify: `config.py` — `FEATURE_COLUMNS`에서 `"training_grade_enc",` 삭제
- Modify: `core/feature_builder.py` — 아래 전부 삭제:
  - `_training_grade(remk_txt)` 함수(45~54행)
  - `_fetch_training_grades(self, df, race_date, client)` 메서드(193~215행)
  - `_fetch_training_grades_for_training(self, df, client)` 메서드(516~557행)
  - `build_for_prediction` 안의 `df = self._fetch_training_grades(df, race_date, client)` 호출 줄(90행)
  - `build_for_training` 안의 `df = self._fetch_training_grades_for_training(df, client)` 호출 줄(107행)
- Modify: `core/preprocessor.py` — `fit_transform`의 27~28행
  (`if "training_grade_enc" not in df.columns: df["training_grade_enc"] = 0.0`)과
  `transform`의 42~43행(동일 패턴) 삭제
- Modify: `core/feature_importance_report.py` — 40행
  (`"training_grade_enc": "조교 등급(인코딩)",`) 삭제
- Modify: `tests/test_feature_builder.py` — `_training_grade` 단위테스트
  4개(`test_training_grade_양호`/`_불량`/`_빈문자`/`_일반`) 삭제,
  `training_grade_enc`를 검증/포함하는 다른 테스트(128행 근처 컬럼
  목록, 150행 `result["training_grade_enc"].iloc[0] == 1.0` assertion)
  수정(해당 피처 관련 부분만 제거, 나머지 assertion은 유지)
- Modify: `tests/test_trainer.py` — 합성 데이터 생성 부분(133, 200,
  338행 근처) 중 `"training_grade_enc": np.random.uniform(0, 3, n),`
  줄 삭제

`horse_training_sessions` 테이블/스키마, `remk_txt` 관련 다른 코드는
건드리지 않는다(이 테이블을 참조하는 코드가 이제 하나도 남지 않게
된다).

## ② `_add_rolling_recent_rates` 날짜 정합 수정

**Files:**
- Modify: `core/preprocessor.py` — `_add_rolling_recent_rates`(145~177행)
  안의 `_rolling_rate` 내부 함수

현재 코드:
```python
for i in range(len(wins)):
    cutoff = dates[i] - np.timedelta64(30, "D")
    past = wins[:i][dates[:i] >= cutoff]
    vals.append(float(past.mean()) if len(past) > 0 else 0.0)
```

수정:
```python
for i in range(len(wins)):
    current_date = dates[i]
    cutoff = current_date - np.timedelta64(30, "D")
    mask = (dates < current_date) & (dates >= cutoff)
    past = wins[mask]
    vals.append(float(past.mean()) if len(past) > 0 else 0.0)
```

`wins[:i]`/`dates[:i]`(행 위치 슬라이싱)를 `dates < current_date`(날짜
비교)로 바꾼다 — 같은 날짜의 다른 행은 정렬 순서와 무관하게 전부
제외된다. 그룹(`groupby(id_col)`) 단위로 순회하는 바깥 구조는 그대로
유지한다(이미 `race_entries`가 `_drop_non_finish_rows`로 유효 착순만
남은 뒤 호출되므로, 예측 경로의 `final_rank > 0 AND final_rank <
NON_FINISH_RANK_MIN` 필터와 동등하다 — 추가 필터 불필요).

- Test: `tests/test_preprocessor.py`(또는 해당 함수를 다루는 기존
  테스트 파일)에 새 테스트 추가 — 같은 날짜에 같은 `jk_no`가 여러
  행으로 등장하는 합성 데이터를 만들어, 그중 나중 정렬된 행의
  `jockey_recent_win_rate`가 같은 날 앞선 행의 승패를 반영하지
  **않는지** 검증한다.

## 에러 처리/엣지케이스

- ①: 제거 후 `FEATURE_COLUMNS`가 26개가 된다. `core/trainer.py`는
  `trainer.feature_columns`를 모델 저장 시점 값으로 자체 보관하므로
  (`_load_predictions_with_probs` 등 기존 코드가 `config.FEATURE_COLUMNS`
  를 직접 재조회하지 않고 `trainer.feature_columns`를 씀), 기존에 27개
  피처로 학습된 저장 모델은 영향받지 않는다 — 다음 재학습부터 26개로
  저장된다.
- ②: 그룹 내 행이 1개뿐이거나 과거 30일 내 기록이 전혀 없는 경우
  (`len(past) == 0`)는 기존과 동일하게 0.0을 반환한다(동작 변화 없음).

## 테스트 방침

- 삭제된 함수를 대상으로 하던 기존 테스트는 제거한다(①).
- `_add_rolling_recent_rates`에 새 회귀 테스트를 추가해, 수정 전 코드
  였다면 실패했을 "같은 날 여러 경주" 시나리오를 명시적으로 커버한다(②).
- 기존 전체 테스트 스위트가 수정 후에도 전부 통과해야 한다.

## 범위 외 (이번에 하지 않음)

- 재학습(`python main.py --mode train`) — 사용자 지시로 이번 범위에서
  제외, 별도 후속 작업으로 진행
- 1~3차 복연승 백테스트 재평가, 연승/단승 성능 재확인
- `horse_training_sessions`를 채우는 신규 수집기 개발
- `_fetch_recent_rates`(예측 경로) 자체의 수정 — 이미 올바름
- 다른 FEATURE_COLUMNS 피처(이미 정상 확인됐거나 별도 이슈로 분리된 것)
