import pytest
from core.race_filter import RaceFilter


def make_race(distance=1200, grade="일반", total_horses=8):
    return {"race_distance": distance, "race_grade": grade, "total_horses": total_horses}


def test_allows_1200m_ilban():
    f = RaceFilter()
    assert f.is_eligible(make_race(distance=1200, grade="일반"))


def test_allows_1400m_ilban():
    f = RaceFilter()
    assert f.is_eligible(make_race(distance=1400, grade="일반"))


def test_rejects_1800m():
    f = RaceFilter()
    assert not f.is_eligible(make_race(distance=1800))


def test_rejects_1300m():
    f = RaceFilter()
    assert not f.is_eligible(make_race(distance=1300))


def test_rejects_special_race():
    f = RaceFilter()
    assert not f.is_eligible(make_race(grade="뚝섬배(G2)"))


def test_rejects_jibs_race():
    f = RaceFilter()
    assert not f.is_eligible(make_race(grade="JIBS배"))


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
        make_race(distance=1200, total_horses=3),
    ]
    result = f.filter_races(races)
    assert len(result) == 2
    assert all(r["race_distance"] in [1200, 1400] for r in result)


def test_rejects_none_total_horses():
    f = RaceFilter()
    assert not f.is_eligible({"race_distance": 1200, "race_grade": "일반", "total_horses": None})
