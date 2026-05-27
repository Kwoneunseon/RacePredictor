# EV 기반 베팅 시스템 + LTR 모델 연결 설계

**작성일:** 2026-05-27  
**구현 순서:** B (EV+배당 스크래핑) → A (필터링+베팅타입 최적화)

---

## 1. 배경 및 목표

현재 봇은 4모델 앙상블(RF+XGBoost+LightGBM+LR)로 예측 후 확률 0.6 이상이면 1,000원 고정 베팅한다. 문제는 모델이 인기마(배당 1.1~1.3배)를 맞춰도 수수료(25%)를 감안하면 EV가 음수라는 점이다.

**목표:**
1. 실시간 배당률 기반 EV > 0 인 경주만 베팅 (수익 구조 개선)
2. CatBoost LTR 모델(단승 승률 20.9%) 연결 (예측 정확도 개선)
3. 비서울 경마장 API 버그 수정 (제주·부산경남 표출 오류)

**손익분기:** 단승 승률 20.9% → 평균 배당 ≥ 4.78배 필요. EV 필터로 저배당 인기마 제외.

---

## 2. OddsCollector (`core/odds_collector.py`)

실시간 배당률을 수집하고, 실패 시 DB 통계로 폴백한다.

```python
class OddsCollector:
    def get_odds(race_date, meet_code) -> dict
    # 반환: {race_id: {horse_id: odds, ..., "source": "scraping"|"fallback"}}

    def _scrape_kra(race_date, meet_code) -> dict | None
    # Playwright headless Chromium으로 race.kra.co.kr 스크래핑
    # 레이스 15분 전 수집, 30초 타임아웃

    def _fallback_from_db(race_date, meet_code) -> dict
    # DB의 win_odds 중앙값: (meet_code, race_grade, distance_bucket) 기준
    # 이 경우 EV 임계값을 0.25로 상향 적용
```

**2-Track 아키텍처:**
- Primary: Playwright 스크래핑 → EV 임계값 0.10
- Fallback: DB 통계 중앙값 → EV 임계값 0.25
- Emergency: 배당 확보 불가 시 해당 경주 베팅 전면 생략

---

## 3. EVCalculator (`core/ev_calculator.py`)

모델 확률과 시장 배당을 받아 기댓값과 Kelly 베팅 비율을 계산한다.

```python
class EVCalculator:
    def calculate_ev(model_prob: float, odds: float) -> float
    # EV = model_prob × odds - 1

    def kelly_fraction(p: float, b: float) -> float
    # f = (p × b - q) / b  (b = odds - 1, q = 1 - p)

    def bet_size(bankroll: float, kelly_f: float) -> float
    # 실제 베팅 = bankroll × kelly_f × 0.25  (¼ Kelly)

    def recommend(model_prob, odds, bankroll, source) -> BetRecommendation | None
    # source별 EV 임계값 적용 후 BetRecommendation 반환
```

**베팅 타입 선택:**

| 조건 | 타입 |
|------|------|
| EV > 0.1, 모델확률 > 0.3 | 단승 |
| 0.1 < EV ≤ 0.2, 확률 0.2~0.3 | 연승 |
| EV ≤ 0 | 베팅 없음 |

**Favorite-Longshot Bias 필터:**
- 모델 순위 1위 + 시장 배당 순위 3~8위 → EV 임계값 0.05 완화 (시장 저평가)
- 시장 배당 1위(최저 배당) + 모델 확률 < 0.35 → 베팅 제외 (시장 고평가)

---

## 4. Investor 개편 (`core/investor.py`)

기존 고정 임계값·고정 금액 방식을 EV+Kelly 기반으로 전면 교체한다.

```python
class Investor:
    def __init__(self, bankroll: float):
        self.bankroll = bankroll          # config.py: BANKROLL = 100_000
        self.odds_collector = OddsCollector()
        self.ev_calc = EVCalculator()

    def decide(self, predictions, race_date, meet_code) -> list[BetDecision]
    # predictions → odds_map → EVCalculator.recommend() → BetDecision 목록

    def record_result(self, bet_decision, actual_rank)
    # 실제 결과 DB 기록
```

**bankroll 관리 원칙:**
- 레이스별 최대 베팅 = bankroll × 5% (Kelly 초과 시 캡 적용)
- 최소 베팅 단위 = 100원 (KRA 규정)

**출력 예시 (main.py):**
```
[서울 3경주] 홍길동 → 단승 2,300원 (EV=+0.18, Kelly=9.2%)
[서울 7경주] 베팅 없음 (EV=-0.05, 배당 출처: 폴백)
```

---

## 5. RaceFilter (`core/race_filter.py`)

베팅 대상 경주를 사전 필터링한다. 모든 조건은 AND로 적용된다.

```python
class RaceFilter:
    def is_eligible(self, race_info: dict) -> bool
    def filter_races(self, races: list[dict]) -> list[dict]
```

**필터 규칙:**

| 필터 | 기준 | 이유 |
|------|------|------|
| 거리 | 1200m 또는 1400m | 단거리가 예측 안정성 높음 |
| 등급 | 4급 또는 5급 | 상위급은 이변 빈번 |
| 초출마 비율 | 경주 내 30% 미만 | 미지수 과다 시 확률 신뢰도 저하 |
| 출전두수 | 5두 이상 | 소규모 경주 배당 낮음 |

- 초출마 판별: `race_entries.prev_total_races == 0`
- 서울 주말 기준 12경주 중 약 3~5경주 통과 예상
- 필터 파라미터는 `config.py`에서 조정 가능

---

## 6. LTR 모델 연결 (`core/ltr_predictor.py`)

`models/my_ltr_model_individual/`의 CatBoost LTR 모델(단승 승률 20.9%)을 `main.py`에 연결한다.

```python
class LTRPredictor:
    def __init__(self, model_dir: str):
        # CatBoost 모델 + metadata.json 로드
        # feature_columns는 metadata에서 읽음

    def predict_from_db(self, race_date, meet_code) -> pd.DataFrame
    # DB 조회 → LTR 피처 구성 → 예측 → 확률 캘리브레이션

    def _calibrate(self, raw_scores, race_groups) -> np.ndarray
    # 경주별 Softmax: LTR 점수 → 합계=1인 확률 분포
```

**피처 매핑:**
- `metadata.json`의 `feature_columns` 기준 자동 매핑
- DB에 없는 피처(예: `humidity_percent`)는 0으로 대체 + 경고 로그

**main.py 변경 (cmd_predict 내 3줄 교체):**
```python
# 변경 전
trainer = RaceModelTrainer()
mm = ModelManager()
mm.load(trainer, preprocessor)
predictor = RacePredictor(trainer, preprocessor)

# 변경 후
from core.ltr_predictor import LTRPredictor
predictor = LTRPredictor(model_dir="models/my_ltr_model_individual")
```

---

## 7. 비서울 경마장 API 버그 수정 (`library/collector_races.py`)

**버그:** `fetch_race_data_all`이 모든 경마장에 부산경남 전용 엔드포인트를 사용.

**수정:**
```python
MEET_RACE_ENDPOINTS = {
    1: "https://apis.data.go.kr/B551015/API186_1/SeoulRace_1",   # 서울
    2: "https://apis.data.go.kr/B551015/API4_2/raceResult_2",    # 제주
    3: "http://apis.data.go.kr/B551015/API4_3/raceResult_3",     # 부산경남
}
# meet_code별 엔드포인트 + 파라미터 형식(서울: rc_date_to/fr, 제주·부산: rc_month) 분기
# 재시도 시 meet_code 누락 버그 함께 수정
```

---

## 8. 구현 순서 (B → A)

1. 비서울 API 버그 수정 (`collector_races.py`) — 독립, 즉시 적용
2. LTR 모델 연결 (`ltr_predictor.py` + `main.py`) — 예측 정확도 기반 확보
3. OddsCollector (`odds_collector.py`) — Playwright 환경 구성 포함
4. EVCalculator (`ev_calculator.py`)
5. RaceFilter (`race_filter.py`)
6. Investor 개편 (`investor.py`) — 위 모듈 통합
7. main.py 전체 흐름 연결 및 출력 포맷 업데이트

---

## 9. 새 파일 목록

| 파일 | 역할 |
|------|------|
| `core/ltr_predictor.py` | CatBoost LTR 예측기 |
| `core/odds_collector.py` | 배당률 수집 (Playwright + DB 폴백) |
| `core/ev_calculator.py` | EV·Kelly 계산 |
| `core/race_filter.py` | 경주 필터링 |

**수정 파일:** `core/investor.py`, `main.py`, `library/collector_races.py`, `config.py`
