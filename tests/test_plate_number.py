from src.plate_number import normalize_plate_number


def test_normalize_plate_number_drops_noise_and_maps_context():
    assert normalize_plate_number("BD0T5410WHI") == "BD2541WH"


def test_normalize_plate_number_keeps_clean_plate():
    assert normalize_plate_number("F6797BB") == "F6797BB"


def test_normalize_plate_number_keeps_three_letter_suffix():
    assert normalize_plate_number("B6703WJF") == "B6703WJF"
