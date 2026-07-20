from types import SimpleNamespace

from main import _format_actual_result, _resolve_dates, _resolve_training_range


def test_format_actual_result_orders_by_final_rank():
    rows = [
        {"entry_number": 4, "final_rank": 3},
        {"entry_number": 1, "final_rank": 1},
        {"entry_number": 11, "final_rank": 4},
        {"entry_number": 3, "final_rank": 2},
    ]
    assert _format_actual_result(rows) == "1-3-4-11"


def test_format_actual_result_empty_when_no_rows():
    assert _format_actual_result([]) == ""


def test_format_actual_result_skips_missing_entry_number():
    rows = [{"entry_number": None, "final_rank": 1}, {"entry_number": 2, "final_rank": 2}]
    assert _format_actual_result(rows) == "2"


def test_resolve_dates_single_date_when_no_range_given():
    args = SimpleNamespace(date="2026-07-11", start_date=None, end_date=None)
    assert _resolve_dates(args) == ["2026-07-11"]


def test_resolve_dates_range_when_start_and_end_given():
    args = SimpleNamespace(date="2026-07-11", start_date="2026-07-10", end_date="2026-07-12")
    assert _resolve_dates(args) == ["2026-07-10", "2026-07-11", "2026-07-12"]


def test_resolve_dates_range_with_only_start_uses_date_as_end():
    args = SimpleNamespace(date="2026-07-12", start_date="2026-07-10", end_date=None)
    assert _resolve_dates(args) == ["2026-07-10", "2026-07-11", "2026-07-12"]


def test_resolve_dates_range_with_only_end_uses_date_as_start():
    args = SimpleNamespace(date="2026-07-10", start_date=None, end_date="2026-07-12")
    assert _resolve_dates(args) == ["2026-07-10", "2026-07-11", "2026-07-12"]


def test_resolve_training_range_defaults_start_date():
    args = SimpleNamespace(start_date=None, end_date=None)
    start, _ = _resolve_training_range(args)
    assert start == "2024-01-01"


def test_resolve_training_range_explicit_values_kept():
    args = SimpleNamespace(start_date="2025-01-01", end_date="2025-06-01")
    assert _resolve_training_range(args) == ("2025-01-01", "2025-06-01")
