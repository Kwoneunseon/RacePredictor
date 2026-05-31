# 경마 자동 투자 봇 전면 재설계 설계 문서

**날짜**: 2026-05-25  
**상태**: 승인됨

---

## 개요

경마 자동 투자 봇을 전면 재설계한다. 기존 ML 로직은 참고하되, 새 Supabase 스키마·코드 구조·스케줄러·ML 개선을 포함한 완전한 재작성을 목표로 한다.

### 목표
- 새 Supabase 프로젝트에 최적화된 스키마 구축
- 하드코딩된 키 제거 → 환경변수(.env) 관리
- 중복 코드 통합, 모듈 분리
- 마사회 데이터 자동 수집 스케줄러 (경주일 1시간 주기, 평일 1회)
- ML 모델 개선 (XGBoost/LightGBM 추가)
- 예측·투자 기록 DB 저장 및 수익 추적

---

## 아키텍처

```
[공공데이터 API (apis.data.go.kr)]
          ↓
[Collector 레이어]  library/
  ├─ collector_races.py
  ├─ collector_horse.py
  ├─ collector_jockeys.py
  └─ collector_trainers.py
          ↓
[Scheduler]  scheduler.py
  ├─ 경주일: 매 1시간 수집
  └─ 평일: 1일 1회 수집
          ↓
[Supabase DB]  (kjctbohokxdzxrsrmwlh.supabase.co)
          ↑ ↓
[ML 파이프라인]  core/
  ├─ preprocessor.py   (전처리·특성 엔지니어링)
  ├─ trainer.py        (모델 학습: RF + XGBoost + LightGBM)
  ├─ predictor.py      (앙상블 예측)
  └─ backtester.py     (백테스팅)
          ↓
[투자 로직]  core/investor.py
  ├─ 배팅 전략 (Kelly Criterion 기반)
  ├─ 수익 계산
  └─ 카카오톡 알림 (Utils/kakaotalk.py)
```

---

## Supabase 스키마 (신규)

### 핵심 테이블

#### `horses`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| horse_id | text PK | 마사회 말 ID |
| name | text | 말 이름 |
| age | int | 나이 |
| gender | text | 성별 (수컷/암컷) |
| rank | text | 등급 |
| updated_at | timestamptz | 최신 업데이트 시각 |

#### `races`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| race_id | text PK | 경주 ID |
| race_date | date | 경주 날짜 |
| meet_code | int | 경기장 코드 (1=서울, 2=부산, 3=제주) |
| race_no | int | 경주 번호 |
| race_distance | int | 거리 (m) |
| race_grade | text | 등급 |
| track_condition | text | 노면 상태 |
| weather | text | 날씨 |
| total_horses | int | 출전 두수 |

#### `race_entries`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | bigserial PK | - |
| race_id | text FK | 경주 ID |
| horse_id | text FK | 말 ID |
| race_date | date | 경주 날짜 (조회 최적화) |
| entry_number | int | 출전 번호 |
| horse_weight | int | 마체중 (kg) |
| jk_no | text FK | 기수 번호 |
| trainer_id | text FK | 조교사 ID |
| final_rank | int | 최종 순위 (NULL=경주 전) |
| win_odds | float | 단승 배당률 |

#### `jockeys`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| jk_no | text PK | 기수 번호 |
| name | text | 기수 이름 |
| total_races | int | 누적 출전 |
| total_wins | int | 누적 1위 |
| updated_at | timestamptz | - |

#### `trainers`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| trainer_id | text PK | 조교사 ID |
| name | text | 조교사 이름 |
| rc_cnt_t | int | 누적 출전 |
| ord1_cnt_t | int | 누적 1위 |
| updated_at | timestamptz | - |

### 추가 테이블

#### `horse_race_history` (성능 캐시)
말별 과거 성적 집계를 미리 계산해 예측 시 조인 부담을 줄임.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| horse_id | text FK | - |
| as_of_date | date | 기준 날짜 |
| prev_total_races | int | 기준일 이전 총 출전 수 |
| prev_wins | int | 기준일 이전 1위 횟수 |
| prev_top3 | int | 기준일 이전 3위 이내 횟수 |
| prev_5_avg_rank | float | 최근 5경주 평균 순위 |
| prev_total_avg_rank | float | 전체 평균 순위 |

#### `predictions`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | bigserial PK | - |
| race_id | text FK | - |
| horse_id | text FK | - |
| predicted_rank | int | 예측 순위 |
| win_probability | float | 1위 확률 |
| model_version | text | 사용 모델 버전 |
| created_at | timestamptz | - |
| actual_rank | int | 실제 순위 (경주 후 업데이트) |

#### `betting_records`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | bigserial PK | - |
| race_id | text FK | - |
| horse_id | text FK | - |
| bet_amount | int | 베팅 금액 |
| bet_type | text | 단승/연승/복승 |
| odds | float | 배당률 |
| profit | int | 수익 (음수=손실) |
| created_at | timestamptz | - |

#### `scheduler_logs`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | bigserial PK | - |
| job_type | text | collect/predict/bet |
| started_at | timestamptz | - |
| finished_at | timestamptz | - |
| status | text | success/error |
| records_processed | int | 처리 건수 |
| error_message | text | 오류 메시지 |

---

## 코드 구조 (재설계)

```
경마 자동 투자 봇/
├─ .env                      # API 키, Supabase URL/KEY (git 제외)
├─ .env.example              # 예시 (git 포함)
├─ config.py                 # .env 로드, 설정값 노출
├─ main.py                   # 진입점
├─ scheduler.py              # APScheduler 기반 자동화
│
├─ library/                  # 데이터 수집
│   ├─ collector_races.py
│   ├─ collector_horse.py
│   ├─ collector_jockeys.py
│   └─ collector_trainers.py
│
├─ core/                     # ML 파이프라인
│   ├─ preprocessor.py       (전처리·특성 엔지니어링)
│   ├─ trainer.py            (모델 학습)
│   ├─ predictor.py          (예측)
│   ├─ backtester.py         (백테스팅)
│   └─ investor.py           (투자 로직)
│
├─ Utils/
│   ├─ database.py           (Supabase 연결 싱글톤)
│   ├─ kakaotalk.py          (알림)
│   └─ logger.py             (로깅 설정)
│
├─ migrations/               (DB 마이그레이션 SQL)
│   └─ 001_init_schema.sql
│
└─ models/                   (학습된 모델 pkl)
```

---

## 스케줄러 설계

```python
# APScheduler 사용
# 경주일 (토/일): 매 1시간마다 데이터 수집
# 평일: 1일 1회 수집
# 매주 월요일: 모델 재학습 여부 검토
```

경주 유무는 Supabase의 `races` 테이블에서 오늘 날짜 경주가 있는지 확인.

---

## ML 개선 계획

**기존**: RandomForest + GradientBoosting + LogisticRegression 앙상블  
**신규**: XGBoost + LightGBM 추가, 소프트 보팅 앙상블

**추가 특성**:
- 마필 최근 5경주 평균 배당률
- 기수·조교사 최근 30일 승률 (누적이 아닌 최신)
- 경주 거리 × 말 체중 상호작용 특성

---

## 환경변수 관리

```
# .env
SUPABASE_URL=https://kjctbohokxdzxrsrmwlh.supabase.co
SUPABASE_KEY=<service_role_key>
API_KEY=<공공데이터포털_API키>
KAKAO_TOKEN=<카카오토큰>
```

config.py는 python-dotenv로 .env를 로드하고 값을 노출.  
모든 파일에서 직접 키를 하드코딩하지 않음.
