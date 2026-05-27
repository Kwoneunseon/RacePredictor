import pytest
from core.ev_calculator import EVCalculator, BetRecommendation


def test_calculate_ev_positive():
    calc = EVCalculator()
    assert abs(calc.calculate_ev(0.25, 5.0) - 0.25) < 0.001


def test_calculate_ev_negative():
    calc = EVCalculator()
    assert abs(calc.calculate_ev(0.10, 2.0) - (-0.80)) < 0.001


def test_kelly_fraction():
    calc = EVCalculator()
    # p=0.25, b=4.0, q=0.75 → f=(0.25*4-0.75)/4=0.0625
    assert abs(calc.kelly_fraction(0.25, 4.0) - 0.0625) < 0.001


def test_kelly_negative_clamps_to_zero():
    calc = EVCalculator()
    # p=0.1, b=1.0 → f=(0.1-0.9)/1=-0.8 → 0
    assert calc.kelly_fraction(0.10, 1.0) == 0.0


def test_bet_size_quarter_kelly():
    calc = EVCalculator()
    # kelly_f=0.10, bankroll=100000 → 100000*0.10*0.25=2500
    assert calc.bet_size(100_000, 0.10) == 2500


def test_bet_size_capped_at_5pct():
    calc = EVCalculator()
    # kelly_f=0.30 → 7500 but cap=5000
    assert calc.bet_size(100_000, 0.30) == 5000


def test_bet_size_rounded_to_100():
    calc = EVCalculator()
    # kelly_f=0.013 → 100000*0.013*0.25=325 → rounds down to 300
    result = calc.bet_size(100_000, 0.013)
    assert result % 100 == 0
    assert result == 300


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
    # EV = 0.25*4.6-1 = 0.15 → OK scraping(>0.10), NOT fallback(>0.25)
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
