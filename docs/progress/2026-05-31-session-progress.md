# 경마 자동 투자 봇 — 진행 현황 (2026-05-31 세션)

## 이 세션에서 해결한 문제들

### 1. 당일 예측 확률이 비정상적으로 낮았던 원인 → 전부 수정 완료

| 문제 | 원인 | 수정 파일 |
|---|---|---|
| 배당 기반 피처 | win_odds는 경주 완료 후 데이터 → data leakage | `config.py`, `preprocessor.py` |
| track_condition / weather 불일치 | 당일은 `""` / `"-"` → 학습 시 본 적 없는 값 → 첫 번째 클래스로 강제 매핑 | `preprocessor.py` |
| horse_weight = 0 | 미계체 말 62%가 0 → fallback 없음 | `predictor.py` |
| final_rank = 0 학습 오염 | 경주 미완료 엔트리 432건이 학습 데이터에 섞임 | `main.py` |
| horse_race_history 당일 누수 | as_of_date = 경주일 레코드가 경주 후 통계 포함 → lte → lt 변경 | `predictor.py`, `main.py` |
| training_grade_enc 항상 0 | 학습 시 horse_training_sessions 조인 없음 | `main.py` |
| track_condition / weather 제거 | 당일 예측 시 항상 unknown → 제거 | `config.py` |

### 2. 새로 추가된 피처 3개

| 피처 | 의미 | 비고 |
|---|---|---|
| `days_since_last_race` | 직전 경주로부터 경과일 | 피로/복귀 상태 |
| `horse_weight_diff` | 직전 대비 마체중 증감 | horse_weight=0 fallback 포함 |
| `horse_win_rate_at_distance` | 이 거리에서 누적 승률 | leakage 방지(cumsum 방식) |

### 3. 조교 데이터 개선

- `training_grade_enc`: 양호→2, 기록없음→1, 불량계열→0 (기존 0/1 이진 → 삼단계 등급)
- 학습 기간(2025-06~2026-05) 조교 데이터 백필 완료: **26,464건** 수집
- `cmd_train`에 배치 조인(100개 단위) 추가

### 4. 과거 날짜 예측이 70~87%로 비정상 높았던 원인 분석

- `horse_race_history.as_of_date = 경주일` 레코드가 **경주 후 통계**를 담고 있음
- `lte(as_of_date, race_date)` → 당일 결과 포함한 stats 사용 → data leakage
- 수정: `lt(as_of_date, race_date)` + `merge_asof(allow_exact_matches=False)`
- 단, 여전히 과거 완료 경주는 track_condition/weather가 실제값 → 당일 대비 feature 품질 차이 있음

### 5. 출전정보 API 탐색 결과

- 30여 개 API 후보 테스트 → 모두 404/500/403
- **결론**: 현재 API KEY로 접근 가능한 출전정보 전용 API 없음
- 기존 결과 API(SeoulRace_1 등)가 당일 출전 데이터를 반환하나 wgHr/track/weather = 0/"-"
- 별도 출전정보 API 사용하려면 data.go.kr에서 추가 이용 신청 필요

---

## 현재 FEATURE_COLUMNS (24개)

```python
FEATURE_COLUMNS = [
    "horse_age", "is_male", "horse_class_num", "race_distance", "total_horses",
    "horse_weight", "race_grade_enc",
    "prev_total_races", "prev_5_avg_rank", "prev_total_avg_rank",
    "jockey_win_rate", "trainer_win_rate", "horse_win_rate",
    "horse_top3_rate", "experience_score", "recent_form",
    "horse_class_vs_mean", "win_rate_vs_mean",
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    "training_grade_enc",
    "days_since_last_race", "horse_weight_diff", "horse_win_rate_at_distance",
]
```

**제거된 피처**: `win_odds`, `implied_prob`, `race_implied_prob_rank`, `implied_prob_vs_mean`, `track_condition_enc`, `weather_enc`

---

## 현재 모델 성능 (2026-05-31 재학습)

| 모델 | AUC | Acc |
|---|---|---|
| RandomForest | 0.879 | 0.915 |
| XGBoost | 0.853 | 0.917 |
| LightGBM | 0.876 | 0.906 |
| LogisticRegression | 0.878 | 0.893 |

학습 데이터: 11,591건 (2025-06-01 ~ 2026-05-31, final_rank > 0)

---

## 다음 세션에서 구현할 것

### 기능: 매일 10시 자동 예측 + 카톡 알림

**설계 확정 내용:**
- 매일 10:00 (Asia/Seoul) 실행
- 오늘 경주 데이터 없으면 → 종료 (카톡 없음)
- 있으면 → 전체 경주 예측 후 카톡 전송
- 필터 통과 여부도 메시지에 표시

**카톡 메시지 형식:**
```
[경마 예측] 2026-05-31

■ 서울 (10경주)
 1경주 1000m | 럭키링맨     18.3%  [필터제외]
 4경주 1200m | 오라토리움   19.2%  ★베팅대상
 ...

■ 부산 (7경주)
...

총 17경주 / 10:00
```

**구현 대상 파일:**
1. `Utils/kakaotalk.py` — 토큰 자동 갱신 + refresh_token 저장 방식으로 리팩토링
2. `core/predictor.py` — `predict_today()` 함수 추가 (스케줄러에서 재사용)
3. `scheduler.py` — `morning_task()` + CronTrigger(10:00) 추가

### 카카오톡 토큰 발급 (아래 별도 설명)

---

## 카카오톡 토큰 발급 방법

### 1단계: 인증 URL 접속

브라우저에서 아래 URL 접속 (REST API KEY 적용됨):

```
https://kauth.kakao.com/oauth/authorize?client_id=1f69fda13909927ed4aecb7920851877&redirect_uri=https://localhost&response_type=code
```

카카오 로그인 후 → 허용 클릭 → 브라우저 주소창이 아래처럼 바뀜:

```
https://localhost/?code=XXXXXXXXXXXXXXXXXXXXXXXXXX
```

`code=` 뒤의 값을 복사.

### 2단계: 토큰 발급 스크립트 실행

```bash
python kakao_auth.py
```

또는 터미널에서:

```bash
! python kakao_auth.py
```

코드 입력창에 위에서 복사한 code 붙여넣기 → `.env`에 자동 저장.

### 3단계: .env 확인

```
KAKAO_ACCESS_TOKEN=...
KAKAO_REFRESH_TOKEN=...
```

두 항목이 채워지면 완료. 이후 access_token은 코드에서 자동 갱신.

---

## 파일별 현재 상태 요약

```
config.py               — FEATURE_COLUMNS 24개 (배당/날씨 제거)
core/preprocessor.py    — _normalize_raw_fields, _add_historical_performance_features 추가
core/predictor.py       — days_since_last_race, horse_weight_diff, horse_win_rate_at_distance 쿼리 추가
                          horse_weight=0 fallback, lt(as_of_date) 수정
                          training_grade_enc 삼단계 등급 적용
main.py                 — final_rank > 0 필터, horse_name 추가
                          horse_race_history merge_asof(allow_exact_matches=False)
                          training_grade_enc 배치 조인(100개 단위)
library/collector_training.py — trngDt int→str 파싱 버그 수정
Utils/kakaotalk.py      — 미수정 (다음 세션 개선 예정)
scheduler.py            — 미수정 (다음 세션 추가 예정)
```

---

## DB 현황

- Supabase: `kjctbohokxdzxrsrmwlh`
- race_entries: ~12,396건 (completed)
- horse_race_history: 12,309건
- horse_training_sessions: 26,464건 (백필 완료)
- 최신 모델: `models/model_20260531_011457.pkl`
