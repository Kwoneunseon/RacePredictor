import pytest
import pandas as pd
from core.investor import Investor


@pytest.fixture
def predictions():
    return pd.DataFrame({
        "race_id": ["R1", "R1", "R1"],
        "horse_id": ["H1", "H2", "H3"],
        "horse_name": ["말A", "말B", "말C"],
        "entry_number": [1, 2, 3],
        "win_probability": [0.72, 0.55, 0.30],
        "predicted_rank": [1, 2, 3],
    })


def test_investor_recommends_high_probability(predictions):
    investor = Investor(threshold=0.6, bet_amount=1000)
    bets = investor.decide(predictions)
    assert len(bets) >= 1
    assert bets.iloc[0]["win_probability"] >= 0.6


def test_investor_no_bets_below_threshold(predictions):
    investor = Investor(threshold=0.9, bet_amount=1000)
    bets = investor.decide(predictions)
    assert len(bets) == 0


def test_investor_bet_amount_set(predictions):
    investor = Investor(threshold=0.5, bet_amount=2000)
    bets = investor.decide(predictions)
    assert (bets["bet_amount"] == 2000).all()


def test_investor_calculates_profit_on_win():
    investor = Investor(threshold=0.0, bet_amount=1000)
    profit = investor.calculate_profit(odds=3.5, won=True)
    assert profit == int(1000 * 3.5) - 1000


def test_investor_calculates_loss_on_lose():
    investor = Investor(threshold=0.0, bet_amount=1000)
    profit = investor.calculate_profit(odds=3.5, won=False)
    assert profit == -1000
