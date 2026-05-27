import pandas as pd
import pytest
from unittest.mock import patch
from core.investor import Investor


@pytest.fixture
def predictions():
    return pd.DataFrame({
        "race_id": ["R1", "R1", "R1"],
        "horse_id": ["H1", "H2", "H3"],
        "horse_name": ["말A", "말B", "말C"],
        "win_probability": [0.72, 0.55, 0.30],
    })


@pytest.fixture
def high_ev_odds():
    # EV = 0.72 * 8.0 - 1 = 4.76 → 베팅 발생
    return {"R1": {"H1": 8.0, "H2": 8.0, "H3": 8.0, "_source": "fallback"}}


@pytest.fixture
def low_ev_odds():
    # EV = 0.72 * 1.2 - 1 = -0.136 → 베팅 없음
    return {"R1": {"H1": 1.2, "H2": 1.2, "H3": 1.2, "_source": "fallback"}}


def test_decide_returns_bet_when_ev_positive(predictions, high_ev_odds):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=high_ev_odds):
        decisions = investor.decide(predictions, "2026-05-31", 1)
    assert len(decisions) == 1
    assert decisions[0]["horse_id"] == "H1"
    assert decisions[0]["ev"] > 0.25  # fallback 임계값 초과


def test_decide_no_bet_when_ev_negative(predictions, low_ev_odds):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=low_ev_odds):
        decisions = investor.decide(predictions, "2026-05-31", 1)
    assert decisions == []


def test_decide_empty_predictions():
    investor = Investor(bankroll=100_000)
    empty = pd.DataFrame(columns=["race_id", "horse_id", "win_probability", "horse_name"])
    with patch.object(investor.odds_collector, "get_odds", return_value={}):
        decisions = investor.decide(empty, "2026-05-31", 1)
    assert decisions == []


def test_decide_skips_race_with_no_odds(predictions):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value={}):
        decisions = investor.decide(predictions, "2026-05-31", 1)
    assert decisions == []


def test_decision_has_required_fields(predictions, high_ev_odds):
    investor = Investor(bankroll=100_000)
    with patch.object(investor.odds_collector, "get_odds", return_value=high_ev_odds):
        decisions = investor.decide(predictions, "2026-05-31", 1)
    assert len(decisions) == 1
    d = decisions[0]
    for field in ("race_id", "horse_id", "horse_name", "bet_type", "amount", "ev", "odds", "source"):
        assert field in d, f"필드 누락: {field}"


def test_calculate_profit_on_win():
    investor = Investor()
    profit = investor.calculate_profit(odds=3.5, won=True, bet_amount=1000)
    assert profit == int(1000 * 3.5) - 1000


def test_calculate_profit_on_loss():
    investor = Investor()
    profit = investor.calculate_profit(odds=3.5, won=False, bet_amount=1000)
    assert profit == -1000
