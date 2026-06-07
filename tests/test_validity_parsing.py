from src.validity_detection import parse_validity_text


def test_parse_compact_validity():
    result = parse_validity_text("0527")
    assert result["month"] == "05"
    assert result["year"] == "2027"


def test_parse_validity_with_space():
    result = parse_validity_text("05 27")
    assert result["month"] == "05"
    assert result["year"] == "2027"


def test_parse_validity_missing_leading_zero_month():
    result = parse_validity_text("923")
    assert result["month"] == "09"
    assert result["year"] == "2023"


def test_parse_validity_with_four_digit_year():
    result = parse_validity_text("05-2027")
    assert result["month"] == "05"
    assert result["year"] == "2027"


def test_parse_validity_with_dot():
    result = parse_validity_text("05.27")
    assert result["month"] == "05"
    assert result["year"] == "2027"


def test_parse_validity_with_slash():
    result = parse_validity_text("05/27")
    assert result["month"] == "05"
    assert result["year"] == "2027"


def test_parse_validity_with_bullet():
    result = parse_validity_text("12-24")
    assert result["month"] == "12"
    assert result["year"] == "2024"


def test_parse_invalid_month_high():
    result = parse_validity_text("13-2027")
    assert result["month"] is None
    assert result["year"] is None
    assert result["normalized"] is None


def test_parse_invalid_month_zero():
    result = parse_validity_text("00-2027")
    assert result["month"] is None
    assert result["year"] is None
    assert result["normalized"] is None
