from __future__ import annotations

import pandas as pd

from src.evaluation import (
    build_paper_style_report,
    character_accuracy,
    plate_from_ocr_filename,
    validity_from_ocr_filename,
)


def test_plate_and_validity_from_ocr_filename():
    path = "dataset/103-E-6387-PL-04-20_jpeg.rf.abc.jpg"

    assert plate_from_ocr_filename(path) == "E6387PL"
    assert validity_from_ocr_filename(path) == "04-2020"


def test_character_accuracy_counts_numbers_and_letters():
    rows = character_accuracy([("B1234ABC", "B1284ADC")])
    result = {row.category: row for row in rows}

    assert result["Numbers"].total == 4
    assert result["Numbers"].errors == 1
    assert result["Numbers"].true == 3
    assert result["Letters"].total == 4
    assert result["Letters"].errors == 1
    assert result["Letters"].true == 3


def test_build_paper_style_report_from_filename_labels():
    predictions = pd.DataFrame(
        [
            {
                "image_path": "dataset/103-E-6387-PL-04-20_jpeg.rf.abc.jpg",
                "plate_number": "E6387PL",
                "validity_text": "04-2020",
                "detection_confidence": 0.9,
            },
            {
                "image_path": "dataset/117-E-2251-TI-01-22_jpg.rf.def.jpg",
                "plate_number": "E2251T1",
                "validity_text": "01-2023",
                "detection_confidence": 0.8,
            },
        ]
    )

    details, summary = build_paper_style_report(predictions)

    assert int(details["has_plate_ground_truth"].sum()) == 2
    assert int(details["plate_exact"].sum()) == 1
    plate_row = summary[summary["Characters"] == "Plate"].iloc[0]
    assert plate_row["Data Total"] == 2
    assert plate_row["Error"] == 1
    assert plate_row["Number of True"] == 1
    assert plate_row["Accuracy Rate"] == 50.0
