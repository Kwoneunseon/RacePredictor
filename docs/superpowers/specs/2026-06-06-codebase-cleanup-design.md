# 코드베이스 정리 설계 스펙
**날짜**: 2026-06-06  
**범위**: 파일 정리 + 코드 구조 개선 (기능 동작 변경 없음)

---

## 목표

현재 프로젝트의 두 가지 핵심 문제를 해결한다:
1. 루트 디렉토리의 debug/임시 파일 혼재 → 삭제·이동
2. `predict_from_db()` 330줄 단일 메서드, `cmd_train()` ETL 중복 → `FeatureBuilder` 분리로 해소

기능 동작은 그대로 유지한다.

---

## 아키텍처

### 신규: `core/feature_builder.py`

현재 `predictor.py:predict_from_db()` 안에 인라인으로 구현된 DB 쿼리 + 피처 계산 12종을 한 클래스로 추출한다.

```
RaceFeatureBuilder
├── build_for_prediction(race_date, meet_code=None) → pd.DataFrame
│     목적: 예측용 - DB에서 race_entries 조회 후 모든 피처를 붙인 DataFrame 반환
│     호출처: RacePredictor.predict_from_db()
│
└── build_for_training(df, end_date) → pd.DataFrame
      목적: 학습용 - 이미 조회된 race_entries df에 피처 조인
      호출처: main.cmd_train()

private 메서드 (피처별 분리):
  _fetch_weights(df, race_date)            # 마체중 스크래핑
  _fetch_recent_rates(df, race_date)       # 기수/조교사 최근 30일 승률
  _fetch_training_grades(df, race_date)   # 조교현황 training_grade_enc
  _fetch_history(df, race_date)           # horse_race_history 조인
  _fetch_last_race_features(df, race_date) # days_since_last_race, horse_weight_diff
  _fetch_prev_odds(df, race_date)         # prev_win_odds
  _fetch_speed_features(df, race_date)    # prev_speed_figure, speed_figure_trend
  _fetch_distance_rates(df, race_date)    # horse_win_rate_at_distance
  _fetch_combo_features(df, race_date)    # horse_track_win_rate, jockey_horse_win_rate 등 5종
```

### 수정: `core/predictor.py`

`predict_from_db()` 330줄 → ~15줄:

```python
def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
    fb = RaceFeatureBuilder()
    df = fb.build_for_prediction(race_date, meet_code)
    if df.empty:
        return pd.DataFrame()
    df = self.preprocessor.transform(df)
    return self.predict(df)
```

`predict_today()`, `format_kakao_message()` 모듈 레벨 함수는 그대로 유지.

### 수정: `main.py:cmd_train()`

현재 150줄 인라인 ETL → FeatureBuilder 재사용:

```python
def cmd_train(args):
    df = _fetch_race_entries(args)          # DB 조회만 (신규 private 함수, ~40줄)
    fb = RaceFeatureBuilder()
    df = fb.build_for_training(df, args.end_date)
    preprocessor = RacePreprocessor()
    df_processed = preprocessor.fit_transform(df)
    trainer = RaceModelTrainer()
    metrics = trainer.train(df_processed)
    mm = ModelManager()
    mm.save(trainer, preprocessor, metrics=metrics["metrics"])
```

### 이동: `rebuild_horse_history.py` → `core/history_rebuilder.py`

- `main()` 함수 시그니처를 `rebuild(start_date: str, end_date: str)` 로 변경
- `cmd_update_results()`의 sys.argv 조작 제거:

```python
# 기존 (제거)
sys.argv = ["rebuild_horse_history.py", date, date]
rebuild_history()

# 변경
from core.history_rebuilder import rebuild
rebuild(date, date)
```

---

## 파일 정리

### 삭제 (복구 불필요한 임시/디버그 파일)

| 파일 | 이유 |
|---|---|
| `debug_compare.py` | 일회성 디버그 |
| `debug_data.py` | 일회성 디버그 |
| `debug_features.py` | 일회성 디버그 |
| `debug_leak.py` | 일회성 디버그 |
| `debug_result.py` | 일회성 디버그 |
| `api_fields.txt` | API 탐색 메모 |
| `api_search.txt` | API 탐색 메모 |
| `api_search2.txt` | API 탐색 메모 |
| `compare_result.txt` | 임시 출력 |
| `leak_result.txt` | 임시 출력 |
| `pred_0524.txt` | 임시 출력 |
| `pred_0524_v2.txt` | 임시 출력 |
| `pred_result.txt` | 임시 출력 |
| `tmp_out.txt` | 임시 출력 |
| `morning_test.txt` | 임시 메모 |

### `_archive/`로 이동 (참고용 보존)

| 파일 | 이유 |
|---|---|
| `check_today_api.py` | API 탐색 스크립트, 현재 미사용 |
| `find_entry_api.py` | API 탐색 스크립트, 현재 미사용 |
| `find_entry_api2.py` | API 탐색 스크립트, 현재 미사용 |
| `analyze_accuracy.py` | 일회성 정확도 분석 |
| `accuracy_report.csv` | 분석 결과물 |
| `run_predict.py` | main.py로 대체된 구 엔트리포인트 |

### `Logs/`로 이동

| 파일 | 이유 |
|---|---|
| `backfill_log.txt` | 로그 파일 (2MB) |

### `Utils/`로 이동

| 파일 | 이유 |
|---|---|
| `kakao_auth.py` | 카카오 유틸리티, Utils에 속함 |

### 루트 삭제 (이동 후)

- `rebuild_horse_history.py` → `core/history_rebuilder.py` 이동 완료 후 루트 파일 삭제

---

## 데이터 플로우 (변경 후)

```
[예측 모드]
main.cmd_predict()
  └─ RacePredictor.predict_from_db(date)
        └─ RaceFeatureBuilder.build_for_prediction(date)  ← 신규
              ├─ DB: race_entries 조회
              ├─ DB: 피처 9종 조인 (각 private 메서드)
              └─ DataFrame 반환
        └─ preprocessor.transform()
        └─ predict()

[학습 모드]
main.cmd_train()
  ├─ _fetch_race_entries()                   ← 신규 private 함수
  └─ RaceFeatureBuilder.build_for_training() ← 신규, 재사용
        └─ 동일한 피처 조인 로직
  └─ preprocessor.fit_transform()
  └─ trainer.train()
```

---

## 비기능 요건

- 기존 테스트 (`tests/`) 수정 최소화 — `predictor.py` 인터페이스 변경 없음
- `launch.json` 설정 변경 없음
- `.env`, `config.py` 변경 없음
- 기존 모델 파일 (`models/*.pkl`) 호환성 유지

---

## 작업 범위 요약

| 작업 | 파일 |
|---|---|
| 신규 생성 | `core/feature_builder.py` |
| 대폭 수정 | `core/predictor.py`, `main.py` |
| 이동+수정 | `rebuild_horse_history.py` → `core/history_rebuilder.py` |
| 이동 | `kakao_auth.py` → `Utils/kakao_auth.py` |
| 이동 | `backfill_log.txt` → `Logs/` |
| 아카이브 | 6개 파일 → `_archive/` |
| 삭제 | 15개 파일 (debug/임시) |
