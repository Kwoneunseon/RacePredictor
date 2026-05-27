import pandas as pd
import pytest
from unittest.mock import patch
from core.investor import Investor


def make_predictions(probs_by_horse: dict, race_id="R1"):
    """horse_id → win_probability 매핑으로 DataFrame 생성"""
    return pd.DataFrame([
        {"race_id": race_id, "horse_id": hid, "horse_name": f"말{hid}", "win_probability": p}
        for hid, p in probs_by_horse.items()
    ])


# fallback 시 경주 내 상대 배당 추정 검증
# sum_probs * 0.75 > 1.40 이면 베팅 → sum_probs > 1.867 필요

@pytest.fixture
def high_sum_predictions():
    # sum = 0.80+0.60+0.50+0.40 = 2.30 → EV = 2.30*0.75-1 = 0.725 > 0.40 → 베팅
    return make_predictions({"H1": 0.80, "H2": 0.60, "H3": 0.50, "H4": 0.40})


@pytest.fixture
def low_sum_predictions():
    # sum = 0.25+0.20+0.15+0.10+0.10 = 0.80 → EV = 0.80*0.75-1 = -0.40 → 베팅 없음
    return make_predictions({"H1": 0.25, "H2": 0.20, "H3": 0.15, "H4": 0.10, "H5": 0.10})


@pytest.fixture
def fallback_odds_map():
    return {"R1": {"H1": 15.0, "H2": 15.0, "H3": 15.0, "H4": 15.0, "_source": "fallback"}}


def test_fallback_uses_relative_odds_not_median(high_sum_predictions, fallback_odds_map):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=fallback_odds_map):
        decisions = investor.decide(high_sum_predictions, "2026-05-31", 1)
    assert len(decisions) == 1
    # 상대 배당 = (2.30/0.80)*0.75 = 2.16, 과거 중앙값 15.0이 아니어야 함
    assert decisions[0]["odds"] < 5.0, "폴백 배당은 중앙값(15배)이 아닌 상대 추정값이어야 함"


def test_fallback_bets_when_sum_probs_high(high_sum_predictions, fallback_odds_map):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=fallback_odds_map):
        decisions = investor.decide(high_sum_predictions, "2026-05-31", 1)
    assert len(decisions) == 1
    assert decisions[0]["horse_id"] == "H1"
    assert decisions[0]["ev"] > 0.40


def test_fallback_no_bet_when_sum_probs_low(low_sum_predictions, fallback_odds_map):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=fallback_odds_map):
        decisions = investor.decide(low_sum_predictions, "2026-05-31", 1)
    assert decisions == []


def test_high_prob_horse_gets_low_estimated_odds(high_sum_predictions, fallback_odds_map):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=fallback_odds_map):
        decisions = investor.decide(high_sum_predictions, "2026-05-31", 1)
    assert len(decisions) == 1
    # 확률 0.80 말의 추정 배당: (2.30/0.80)*0.75 ≈ 2.16배
    assert abs(decisions[0]["odds"] - 2.16) < 0.1


def test_decide_empty_predictions():
    investor = Investor(bankroll=100_000)
    empty = pd.DataFrame(columns=["race_id", "horse_id", "win_probability", "horse_name"])
    with patch.object(investor.odds_collector, "get_odds", return_value={}):
        assert investor.decide(empty, "2026-05-31", 1) == []


def test_decision_has_required_fields(high_sum_predictions, fallback_odds_map):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=fallback_odds_map):
        decisions = investor.decide(high_sum_predictions, "2026-05-31", 1)
    assert len(decisions) == 1
    for field in ("race_id", "horse_id", "horse_name", "bet_type", "amount", "ev", "odds", "source"):
        assert field in decisions[0], f"필드 누락: {field}"


def test_scraping_source_uses_actual_odds():
    investor = Investor(bankroll=100_000)
    preds = make_predictions({"H1": 0.30})
    scraping_odds = {"R1": {"H1": 8.0, "_source": "scraping"}}
    with patch.object(investor.odds_collector, "get_odds", return_value=scraping_odds):
        decisions = investor.decide(preds, "2026-05-31", 1)
    # EV = 0.30*8.0-1 = 1.40 > 0.10(scraping 임계값) → 베팅
    assert len(decisions) == 1
    assert decisions[0]["odds"] == 8.0  # 실시간 배당은 그대로 사용


def test_calculate_profit_on_win():
    investor = Investor()
    assert investor.calculate_profit(odds=3.5, won=True, bet_amount=1000) == int(1000 * 3.5) - 1000


def test_calculate_profit_on_loss():
    investor = Investor()
    assert investor.calculate_profit(odds=3.5, won=False, bet_amount=1000) == -1000
