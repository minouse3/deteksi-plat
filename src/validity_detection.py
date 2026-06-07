from __future__ import annotations

import re

from .config import VALIDITY_YEAR_PREFIX


def normalize_validity_text(raw_text: str | None) -> str:
    if not raw_text:
        return ""
    cleaned = raw_text.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    return re.sub(r"[^0-9]", "", cleaned)


def parse_validity_text(text: str | None) -> dict[str, str | None]:
    digits = normalize_validity_text(text)
    if len(digits) == 3:
        month = f"0{digits[0]}"
        year = f"{VALIDITY_YEAR_PREFIX}{digits[1:]}"
    elif len(digits) == 4:
        month = digits[:2]
        year = f"{VALIDITY_YEAR_PREFIX}{digits[2:]}"
    elif len(digits) == 6:
        month = digits[:2]
        year = digits[2:]
    else:
        return {"month": None, "year": None, "normalized": None}

    try:
        month_int = int(month)
        year_int = int(year)
    except ValueError:
        return {"month": None, "year": None, "normalized": None}

    if month_int < 1 or month_int > 12 or year_int < 2000 or year_int > 2099:
        return {"month": None, "year": None, "normalized": None}

    month = f"{month_int:02d}"
    year = str(year_int)
    return {"month": month, "year": year, "normalized": f"{month}-{year}"}
