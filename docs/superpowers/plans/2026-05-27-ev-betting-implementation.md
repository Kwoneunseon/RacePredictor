# EV 기반 베팅 시스템 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 고정 임계값 베팅을 EV+Kelly 기반으로 교체하고, 경주 필터링으로 베팅 대상을 좁힌다.

**Architecture:** EVCalculator(순수 계산) → RaceFilter(경주 선별) → OddsCollector(배당 조회) → Investor(베팅 결정) 순서로 독립 모듈을 쌓고 main.py에서 연결한다.

**Tech Stack:** Python 3.11, pandas, numpy, supabase-py, pytest

**DB 실측 확인 사항:**
- `race_grade`: "일반" (대부분) / 특별 경주명 (소수)
- `races.race_distance`: integer (1200, 1400, ...)
- `horses.rank`: "국5", "제4", "외6" 형식 (prefix + 1~6)
- `race_entries` 컬럼: id, race_id, horse_id, race_date, entry_number, horse_weight, jk_no, trainer_id, final_rank, win_odds
- `win_odds`: float, post-race 확정 배당 (median 9.6~27.95)
- LTR 모델 파일 없음 → 4모델 앙상블 유지

---

## 파일 구조

| 파일 | 역할 | 상태 |
|------|------|------|
| `core/ev_calculator.py` | EV·Kelly 계산 | 신규 |
| `core/race_filter.py` | 경주 필터링 | 신규 |
| `core/odds_collector.py` | 배당률 수집 (DB 폴백) | 신규 |
| `core/investor.py` | 베팅 결정 (EV+Kelly 기반) | 수정 |
| `main.py` | 전체 흐름 연결 | 수정 |
| `config.py` | BANKROLL, 필터 파라미터 추가 | 수정 |
| `tests/test_ev_calculator.py` | EVCalculator 단위 테스트 | 신규 |
| `tests/test_race_filter.py` | RaceFilter 단위 테스트 | 신규 |

---

## Task 1: config.py에 상수 추가

**Files:**
- Modify: `config.py`

- [ ] **Step 1: config.py에 베팅 상수 추가**

```python
# 기존 코드 맨 아래에 추가
BANKROLL = 100_000          # 총 자산 (원)
MAX_BET_FRACTION = 0.05     # 레이스별 최대 베팅 비율
KELLY_FRACTION = 0.25       # ¼ Kelly 안전 계수
EV_THRESHOLD_SCRAPING = 0.10
EV_THRESHOLD_FALLBACK = 0.25
FILTER_DISTANCES = [1200, 1400]   # 허용 거리 (m)
FILTER_MIN_HORSES = 5             # 최소 출전두수
```

- [ ] **Step 2: 커밋**

```bash
git add config.py
git commit -m "feat: 베팅 전략 상수 추가 (BANKROLL, EV 임계값, 필터 파라미터)"
```

---

## Task 2: EVCalculator

**Files:**
- Create: `core/ev_calculator.py`
- Create: `tests/test_ev_calculator.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_ev_calculator.py`:
```python
import pytest
from core.ev_calculator import EVCalculator, BetRecommendation

def test_calculate_ev_positive():
    calc = EVCalculator()
    # EV = 0.25 * 5.0 - 1 = 0.25
    assert abs(calc.calculate_ev(0.25, 5.0) - 0.25) < 0.001

def test_calculate_ev_negative():
    calc = EVCalculator()
    # EV = 0.10 * 2.0 - 1 = -0.80
    assert abs(calc.calculate_ev(0.10, 2.0) - (-0.80)) < 0.001

def test_kelly_fraction():
    calc = EVCalculator()
    # p=0.25, b=4.0(odds=5.0), q=0.75
    # f = (0.25*4 - 0.75) / 4 = 0.25/4 = 0.0625
    assert abs(calc.kelly_fraction(0.25, 4.0) - 0.0625) < 0.001

def test_kelly_negative_clamps_to_zero():
    calc = EVCalculator()
    # p=0.1, b=1.0(odds=2.0), q=0.9
    # f = (0.1*1 - 0.9) / 1 = -0.8 → 0
    assert calc.kelly_fraction(0.10, 1.0) == 0.0

def test_bet_size_quarter_kelly():
    calc = EVCalculator()
    # kelly_f=0.10, bankroll=100000, ¼ Kelly → 2500
    assert calc.bet_size(100_000, 0.10) == 2500

def test_bet_size_capped_at_5pct():
    calc = EVCalculator()
    # kelly_f=0.30 → would be 7500, but cap=5000
    assert calc.bet_size(100_000, 0.30) == 5000

def test_bet_size_rounded_to_100():
    calc = EVCalculator()
    # kelly_f=0.013 → 100000 * 0.013 * 0.25 = 325 → rounds to 300
    result = calc.bet_size(100_000, 0.013)
    assert result % 100 == 0

def test_recommend_returns_none_below_ev_threshold():
    calc = EVCalculator()
    result = calc.recommend(model_prob=0.10, odds=2.0, bankroll=100_000, source="scraping")
    assert result is None  # EV = -0.80

def test_recommend_returns_bet_above_threshold():
    calc = EVCalculator()
    result = calc.recommend(model_prob=0.30, odds=5.0, bankroll=100_000, source="scraping")
    assert result is not None
    assert isinstance(result, BetRecommendation)
    assert result.ev > 0.10
    assert result.bet_type in ("단승", "연승")
    assert result.amount >= 100

def test_recommend_fallback_uses_higher_threshold():
    calc = EVCalculator()
    # EV = 0.15 → OK for scraping (>0.10) but NOT for fallback (>0.25)
    result_scraping = calc.recommend(0.25, 4.6, 100_000, source="scraping")
    result_fallback = calc.recommend(0.25, 4.6, 100_000, source="fallback")
    assert result_scraping is not None
    assert result_fallback is None

def test_single_bet_type_for_high_prob():
    calc = EVCalculator()
    result = calc.recommend(model_prob=0.40, odds=6.0, bankroll=100_000, source="scraping")
    assert result is not None
    assert result.bet_type == "단승"

def test_yeonseung_bet_type_for_medium_prob():
    calc = EVCalculator()
    result = calc.recommend(model_prob=0.22, odds=8.0, bankroll=100_000, source="scraping")
    assert result is not None
    assert result.bet_type == "연승"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd "F:\Project\경마 자동 투자 봇"
python -m pytest tests/test_ev_calculator.py -v 2>&1 | head -20
```
Expected: ImportError or ModuleNotFoundError

- [ ] **Step 3: EVCalculator 구현**

`core/ev_calculator.py`:
```python
from dataclasses import dataclass
from config import (
    KELLY_FRACTION, MAX_BET_FRACTION,
    EV_THRESHOLD_SCRAPING, EV_THRESHOLD_FALLBACK,
)


@dataclass
class BetRecommendation:
    bet_type: str   # "단승" | "연승"
    amount: int     # 원 단위, 100원 단위 반올림
    ev: float
    kelly_f: float


class EVCalculator:
    def calculate_ev(self, model_prob: float, odds: float) -> float:
        return model_prob * odds - 1

    def kelly_fraction(self, p: float, b: float) -> float:
        """b = odds - 1 (순이익 배율)"""
        q = 1 - p
        f = (p * b - q) / b
        return max(f, 0.0)

    def bet_size(self, bankroll: float, kelly_f: float) -> int:
        raw = bankroll * kelly_f * KELLY_FRACTION
        capped = min(raw, bankroll * MAX_BET_FRACTION)
        rounded = int(capped // 100) * 100
        return max(rounded, 100)

    def recommend(
        self,
        model_prob: float,
        odds: float,
        bankroll: float,
        source: str,
    ) -> BetRecommendation | None:
        ev = self.calculate_ev(model_prob, odds)
        threshold = EV_THRESHOLD_SCRAPING if source == "scraping" else EV_THRESHOLD_FALLBACK
        if ev <= threshold:
            return None

        b = odds - 1
        kelly_f = self.kelly_fraction(model_prob, b)
        if kelly_f == 0:
            return None

        amount = self.bet_size(bankroll, kelly_f)
        bet_type = "단승" if model_prob >= 0.30 else "연승"

        return BetRecommendation(
            bet_type=bet_type,
            amount=amount,
            ev=ev,
            kelly_f=kelly_f,
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_ev_calculator.py -v
```
Expected: 12 passed

- [ ] **Step 5: 커밋**

```bash
git add core/ev_calculator.py tests/test_ev_calculator.py
git commit -m "feat: EVCalculator - EV·Kelly 계산 모듈 (TDD)"
```

---

## Task 3: RaceFilter

**Files:**
- Create: `core/race_filter.py`
- Create: `tests/test_race_filter.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_race_filter.py`:
```python
import pytest
from core.race_filter import RaceFilter

def make_race(distance=1200, grade="일반", total_horses=8):
    return {"race_distance": distance, "race_grade": grade, "total_horses": total_horses}

def test_allows_1200m_ilban():
    f = RaceFilter()
    assert f.is_eligible(make_race(distance=1200, grade="일반"))

def test_allows_1400m_ilban():
    f = RaceFilter()
    assert f.is_eligible(make_race(distance=1400, grade="일ban"))

def test_rejects_1800m():
    f = RaceFilter()
    assert not f.is_eligible(make_race(distance=1800))

def test_rejects_1300m():
    f = RaceFilter()
    assert not f.is_eligible(make_race(distance=1300))

def test_rejects_special_race():
    f = RaceFilter()
    assert not f.is_eligible(make_race(grade="뚝섬배(G2)"))

def test_rejects_too_few_horses():
    f = RaceFilter()
    assert not f.is_eligible(make_race(total_horses=4))

def test_allows_exactly_5_horses():
    f = RaceFilter()
    assert f.is_eligible(make_race(total_horses=5))

def test_filter_races_returns_eligible_only():
    f = RaceFilter()
    races = [
        make_race(distance=1200),
        make_race(distance=1800),
        make_race(distance=1400),
        make_race(grade="JIBS배"),
    ]
    result = f.filter_races(races)
    assert len(result) == 2
    assert all(r["race_distance"] in [1200, 1400] for r in result)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python -m pytest tests/test_race_filter.py -v 2>&1 | head -10
```
Expected: ImportError

- [ ] **Step 3: RaceFilter 구현**

`core/race_filter.py`:
```python
from config import FILTER_DISTANCES, FILTER_MIN_HORSES


class RaceFilter:
    def is_eligible(self, race: dict) -> bool:
        if race.get("race_distance") not in FILTER_DISTANCES:
            return False
        if race.get("race_grade", "") != "일반":
            return False
        if (race.get("total_horses") or 0) < FILTER_MIN_HORSES:
            return False
        return True

    def filter_races(self, races: list[dict]) -> list[dict]:
        return [r for r in races if self.is_eligible(r)]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python -m pytest tests/test_race_filter.py -v
```
Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
git add core/race_filter.py tests/test_race_filter.py
git commit -m "feat: RaceFilter - 거리/등급/두수 기반 경주 필터 (TDD)"
```

---

## Task 4: OddsCollector (DB 폴백)

**Files:**
- Create: `core/odds_collector.py`

- [ ] **Step 1: OddsCollector 구현**

`core/odds_collector.py`:
```python
import numpy as np
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


class OddsCollector:
    def get_odds(self, race_date: str, meet_code: int) -> dict:
        """
        반환: {race_id: {horse_id: float, "_source": "scraping"|"fallback"}}
        스크래핑 미구현 → 항상 fallback
        """
        return self._fallback_from_db(race_date, meet_code)

    def _fallback_from_db(self, race_date: str, meet_code: int) -> dict:
        client = get_client()

        # 해당 날짜 경주 목록 조회
        races_res = client.table("races")\
            .select("race_id, race_distance")\
            .eq("race_date", race_date)\
            .eq("meet_code", meet_code)\
            .execute()
        if not races_res.data:
            return {}

        # 거리별 중앙값 배당 (과거 전체 기준)
        median_map = self._load_median_odds(meet_code)

        result = {}
        for race in races_res.data:
            race_id = race["race_id"]
            dist = race.get("race_distance")
            fallback_odds = median_map.get(dist, 15.0)

            entries_res = client.table("race_entries")\
                .select("horse_id")\
                .eq("race_id", race_id)\
                .execute()
            if not entries_res.data:
                continue

            result[race_id] = {
                e["horse_id"]: fallback_odds
                for e in entries_res.data
            }
            result[race_id]["_source"] = "fallback"

        logger.info(f"폴백 배당 로드: {len(result)}경주 (meet={meet_code})")
        return result

    def _load_median_odds(self, meet_code: int) -> dict[int, float]:
        client = get_client()
        res = client.rpc("median_odds_by_distance", {"p_meet_code": meet_code}).execute()
        if res.data:
            return {row["race_distance"]: row["median_odds"] for row in res.data}
        # RPC 없으면 SQL 직접 조회
        return self._load_median_odds_sql(meet_code)

    def _load_median_odds_sql(self, meet_code: int) -> dict[int, float]:
        from Utils.database import get_client
        client = get_client()
        res = client.table("race_entries")\
            .select("win_odds, races(race_distance)")\
            .gt("win_odds", 0)\
            .execute()
        if not res.data:
            return {}

        from collections import defaultdict
        grouped: dict[int, list] = defaultdict(list)
        for row in res.data:
            rc = row.get("races") or {}
            dist = rc.get("race_distance")
            odds = row.get("win_odds")
            if dist and odds:
                grouped[dist].append(odds)

        return {dist: float(np.median(vals)) for dist, vals in grouped.items()}
```

- [ ] **Step 2: 동작 확인 (DB 연결 필요)**

```bash
python -c "
from core.odds_collector import OddsCollector
oc = OddsCollector()
result = oc._load_median_odds_sql(1)
print('서울 거리별 중앙값:', result)
"
```
Expected: `{1200: 18.6, 1400: 15.35, ...}` 형식

- [ ] **Step 3: 커밋**

```bash
git add core/odds_collector.py
git commit -m "feat: OddsCollector - DB 폴백 배당 수집 (거리별 중앙값)"
```

---

## Task 5: Investor 개편

**Files:**
- Modify: `core/investor.py`

- [ ] **Step 1: Investor 전면 교체**

`core/investor.py`:
```python
import pandas as pd
from Utils.database import upsert, get_client
from Utils.logger import get_logger
from core.ev_calculator import EVCalculator
from core.odds_collector import OddsCollector
from config import BANKROLL

logger = get_logger(__name__)


class Investor:
    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.ev_calc = EVCalculator()
        self.odds_collector = OddsCollector()

    def decide(
        self,
        predictions: pd.DataFrame,
        race_date: str,
        meet_code: int,
    ) -> list[dict]:
        odds_map = self.odds_collector.get_odds(race_date, meet_code)
        decisions = []

        race_groups = predictions.groupby("race_id", sort=False)
        for race_id, group in race_groups:
            race_odds = odds_map.get(race_id, {})
            source = race_odds.get("_source", "fallback")

            top = group.sort_values("win_probability", ascending=False).iloc[0]
            horse_id = top.get("horse_id")
            model_prob = top["win_probability"]
            odds = race_odds.get(horse_id, 0)

            if odds <= 0:
                logger.debug(f"{race_id}: 배당 없음, 스킵")
                continue

            rec = self.ev_calc.recommend(model_prob, odds, self.bankroll, source)
            if rec is None:
                logger.debug(f"{race_id}: EV 미달, 스킵")
                continue

            decisions.append({
                "race_id": race_id,
                "horse_id": horse_id,
                "horse_name": top.get("horse_name", horse_id),
                "bet_type": rec.bet_type,
                "amount": rec.amount,
                "ev": rec.ev,
                "kelly_f": rec.kelly_f,
                "model_prob": model_prob,
                "odds": odds,
                "source": source,
            })

        logger.info(f"베팅 결정: {len(decisions)}건 / {len(race_groups)}경주")
        return decisions

    def save_bets(self, decisions: list[dict]) -> bool:
        if not decisions:
            return True
        rows = [
            {
                "race_id": d["race_id"],
                "horse_id": d["horse_id"],
                "bet_amount": d["amount"],
                "bet_type": d["bet_type"],
                "odds": d["odds"],
            }
            for d in decisions
        ]
        return upsert("betting_records", rows, "id")

    def calculate_profit(self, odds: float, won: bool, bet_amount: int) -> int:
        if won:
            return int(bet_amount * odds) - bet_amount
        return -bet_amount

    def update_results(self, race_id: str) -> None:
        client = get_client()
        try:
            entries = client.table("race_entries")\
                .select("horse_id, final_rank, win_odds")\
                .eq("race_id", race_id)\
                .execute().data or []
            info_map = {e["horse_id"]: e for e in entries}

            bets = client.table("betting_records")\
                .select("id, horse_id, bet_amount")\
                .eq("race_id", race_id)\
                .execute().data or []

            for bet in bets:
                info = info_map.get(bet["horse_id"], {})
                won = info.get("final_rank") == 1
                odds = info.get("win_odds") or 0
                profit = self.calculate_profit(odds, won, bet["bet_amount"])
                client.table("betting_records")\
                    .update({"profit": profit})\
                    .eq("id", bet["id"])\
                    .execute()
            logger.info(f"{race_id} 결과 업데이트 완료")
        except Exception as e:
            logger.error(f"결과 업데이트 실패: {e}")
```

- [ ] **Step 2: 기존 investor 테스트 확인**

```bash
python -m pytest tests/test_investor.py -v
```
Expected: 기존 테스트 중 일부 실패할 수 있음 (시그니처 변경) → 다음 Step에서 수정

- [ ] **Step 3: 테스트 업데이트**

`tests/test_investor.py`에서 `Investor(threshold=0.6, bet_amount=1000)` → `Investor()` 로 변경,
`decide(predictions)` → `decide(predictions, race_date, meet_code)` 시그니처 맞춤.

기존 테스트가 DB 연동 없이 동작하도록 mock 적용:
```python
from unittest.mock import patch, MagicMock

def test_decide_no_eligible_races():
    investor = Investor(bankroll=100_000)
    empty_df = pd.DataFrame(columns=["race_id", "horse_id", "win_probability", "horse_name"])
    with patch.object(investor.odds_collector, "get_odds", return_value={}):
        result = investor.decide(empty_df, "2026-05-31", 1)
    assert result == []
```

- [ ] **Step 4: 커밋**

```bash
git add core/investor.py tests/test_investor.py
git commit -m "feat: Investor 개편 - EV+Kelly 기반 베팅 결정"
```

---

## Task 6: main.py 연결

**Files:**
- Modify: `main.py`

- [ ] **Step 1: cmd_predict 수정**

`main.py`의 `cmd_predict` 함수를 아래로 교체:

```python
def cmd_predict(args):
    logger.info(f"=== 예측 시작: {args.date} ===")

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()

    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음. 먼저 --mode train 실행")
        return

    from core.race_filter import RaceFilter
    from core.investor import Investor

    predictor = RacePredictor(trainer, preprocessor)
    predictions = predictor.predict_from_db(args.date, meet_code=args.meet)

    if predictions.empty:
        logger.warning(f"{args.date} 예측 결과 없음")
        return

    # 경주 필터 적용
    race_filter = RaceFilter()
    if "race_id" in predictions.columns:
        from Utils.database import get_client
        client = get_client()
        races_res = client.table("races")\
            .select("race_id, race_distance, race_grade, total_horses")\
            .eq("race_date", args.date)\
            .execute()
        race_info = {r["race_id"]: r for r in (races_res.data or [])}
        eligible_ids = {
            rid for rid, info in race_info.items()
            if race_filter.is_eligible(info)
        }
        predictions = predictions[predictions["race_id"].isin(eligible_ids)]

    MEET_NAME = {1: "서울", 2: "제주", 3: "부산", 0: "미상"}

    def race_label(row):
        meet = MEET_NAME.get(row.get("meet_code"), "미상")
        race_no = row.get("race_no", "?")
        return f"[{meet} {race_no}경주]"

    print(f"\n=== 예측 결과 ({args.date}) ===")
    race_groups = predictions.groupby("race_id", sort=False)

    for race_id, group in sorted(race_groups, key=lambda x: (x[1].iloc[0].get("meet_code", 0),
                                                               x[1].iloc[0].get("race_no", 0))):
        top = group.sort_values("win_probability", ascending=False).iloc[0]
        label = race_label(top)
        name = top.get("horse_name") or top.get("horse_id", "?")
        prob = top["win_probability"]
        print(f"  {label} 1위 예측: {name} (확률: {prob:.3f})")

    # EV 기반 베팅 결정
    investor = Investor()
    meet_code = args.meet or 1
    decisions = investor.decide(predictions, args.date, meet_code)

    if decisions:
        print(f"\n베팅 추천: {len(decisions)}경주")
        for d in decisions:
            print(
                f"  [{d['race_id']}] {d['horse_name']} → "
                f"{d['bet_type']} {d['amount']:,}원 "
                f"(EV={d['ev']:+.2f}, 확률={d['model_prob']:.3f}, 배당={d['odds']:.1f}배, 출처={d['source']})"
            )
    else:
        print("\n베팅 추천: 없음 (EV 기준 미달)")
```

- [ ] **Step 2: 동작 테스트**

```bash
python main.py --mode predict --date 2025-11-01 --meet 1
```
Expected: 예측 결과 출력, 베팅 결정 출력 (EV 기반)

- [ ] **Step 3: 커밋**

```bash
git add main.py
git commit -m "feat: main.py - EV+Kelly 베팅 결정 및 경주 필터 통합"
```

---

## 전체 테스트 실행

```bash
python -m pytest tests/test_ev_calculator.py tests/test_race_filter.py -v
```
Expected: 20+ 테스트 통과
