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


def test_safe_str_strips_whitespace():
    assert safe_str("  홍길동  ") == "홍길동"


def test_safe_str_returns_default_on_empty_or_none():
    assert safe_str(None) == ""
    assert safe_str("") == ""
    assert safe_str(None, default="?") == "?"
