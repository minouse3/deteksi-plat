from datetime import date

from src.date_validation import check_validity


def test_future_validity_is_valid():
    assert check_validity("05", "2027", current_date=date(2026, 6, 3)) == "valid"


def test_past_validity_is_expired():
    assert check_validity("03", "2025", current_date=date(2026, 6, 3)) == "expired"


def test_missing_validity_is_unknown():
    assert check_validity(None, None, current_date=date(2026, 6, 3)) == "unknown"


def test_invalid_month_is_unknown():
    assert check_validity("13", "2027", current_date=date(2026, 6, 3)) == "unknown"
