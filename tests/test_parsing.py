from library.parsing import parse_date, safe_int, safe_float, safe_str


def test_parse_date_converts_yyyymmdd_to_iso():
    assert parse_date("20260805") == "2026-08-05"


def test_parse_date_returns_none_on_invalid_input():
    assert parse_date("not-a-date") is None
    assert parse_date(None) is None


def test_safe_int_converts_valid_value():
    assert safe_int("5") == 5
    assert safe_int(5) == 5


def test_safe_int_returns_default_on_empty_or_none():
    assert safe_int(None) == 0
    assert safe_int("") == 0
    assert safe_int("  ") == 0
    assert safe_int(None, default=-1) == -1


def test_safe_int_returns_default_on_invalid_string():
    assert safe_int("abc") == 0


def test_safe_float_converts_valid_value():
    assert safe_float("1.5") == 1.5
    assert safe_float(1.5) == 1.5


def test_safe_float_returns_default_on_empty_or_none():
    assert safe_float(None) == 0.0
    assert safe_float("") == 0.0
    assert safe_float("  ") == 0.0


def test_safe_float_returns_none_on_invalid_string():
    """safe_float의 예외 경로는 safe_int/safe_str과 달리 default를 무시하고
    항상 None을 반환한다 — 기존 동작을 고정하는 회귀 방지 테스트."""
    assert safe_float("abc") is None
    assert safe_float("abc", default=1.0) is None


def test_safe_int_falsy_valid_value_collapses_to_default():
    """0처럼 falsy하지만 유효한 값도 `if value` 가드 때문에 default로
    처리된다 — 기존 동작을 고정하는 회귀 방지 테스트."""
    assert safe_int(0, default=-1) == -1


def test_safe_str_falsy_valid_value_collapses_to_default():
    assert safe_str(0, default="?") == "?"


def test_safe_str_strips_whitespace():
    assert safe_str("  홍길동  ") == "홍길동"


def test_safe_str_returns_default_on_empty_or_none():
    assert safe_str(None) == ""
    assert safe_str("") == ""
    assert safe_str(None, default="?") == "?"
