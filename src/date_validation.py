from __future__ import annotations

import calendar
from datetime import date


def check_validity(month: str | int | None, year: str | int | None, current_date: date | None = None) -> str:
    if month is None or year is None:
        return "unknown"

    try:
        month_int = int(month)
        year_int = int(year)
    except (TypeError, ValueError):
        return "unknown"

    if month_int < 1 or month_int > 12 or year_int < 1900:
        return "unknown"

    current = current_date or date.today()
    last_day = calendar.monthrange(year_int, month_int)[1]
    valid_until = date(year_int, month_int, last_day)
    return "valid" if current <= valid_until else "expired"
