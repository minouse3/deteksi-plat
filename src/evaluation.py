from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_STEM_SPLIT_RE = re.compile(r"_(?:jpg|jpeg|png|bmp|webp|tif|tiff)\.rf\.", re.IGNORECASE)


@dataclass(frozen=True)
class CharacterAccuracy:
    category: str
    total: int
    errors: int

    @property
    def true(self) -> int:
        return self.total - self.errors

    @property
    def accuracy(self) -> float:
        return (self.true / self.total * 100.0) if self.total else 0.0


def normalize_plate_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def normalize_validity_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def plate_from_ocr_filename(image_path: str) -> str | None:
    """Infer plate ground truth from common Kaggle filename patterns."""
    name = Path(str(image_path)).name
    stem = IMAGE_STEM_SPLIT_RE.split(name)[0]
    stem = Path(stem).stem

    parts = stem.split("-")
    if len(parts) >= 6 and parts[2].isdigit():
        return normalize_plate_text("".join(parts[1:4]))

    if re.fullmatch(r"[A-Z]{1,2}[0-9][A-Z0-9]{4,8}", stem.upper()):
        return normalize_plate_text(stem)

    return None


def validity_from_ocr_filename(image_path: str) -> str | None:
    name = Path(str(image_path)).name
    stem = IMAGE_STEM_SPLIT_RE.split(name)[0]
    stem = Path(stem).stem
    parts = stem.split("-")
    if len(parts) >= 6 and parts[4].isdigit() and parts[5].isdigit():
        month = parts[4].zfill(2)
        year = parts[5]
        if len(year) == 2:
            year = f"20{year.zfill(2)}"
        if len(year) == 4:
            return f"{month}-{year}"
    return None


def levenshtein_alignment(expected: str, predicted: str) -> list[tuple[str | None, str | None]]:
    """Return aligned character pairs. None represents insertion/deletion."""
    expected = normalize_plate_text(expected)
    predicted = normalize_plate_text(predicted)
    m, n = len(expected), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution = 0 if expected[i - 1] == predicted[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution,
            )

    aligned: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution = 0 if expected[i - 1] == predicted[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + substitution:
                aligned.append((expected[i - 1], predicted[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((expected[i - 1], None))
            i -= 1
            continue
        aligned.append((None, predicted[j - 1]))
        j -= 1

    aligned.reverse()
    return aligned


def character_accuracy(records: Iterable[tuple[str, str]]) -> list[CharacterAccuracy]:
    totals = {"Numbers": 0, "Letters": 0}
    errors = {"Numbers": 0, "Letters": 0}

    for expected, predicted in records:
        for exp_char, pred_char in levenshtein_alignment(expected, predicted):
            if exp_char is None:
                continue
            category = "Numbers" if exp_char.isdigit() else "Letters"
            totals[category] += 1
            if exp_char != pred_char:
                errors[category] += 1

    return [
        CharacterAccuracy("Numbers", totals["Numbers"], errors["Numbers"]),
        CharacterAccuracy("Letters", totals["Letters"], errors["Letters"]),
    ]


def build_paper_style_report(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
    infer_from_filename: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = predictions.copy()
    df["image"] = df["image_path"].map(lambda value: Path(str(value)).name)

    if ground_truth is not None:
        gt = ground_truth.copy()
        gt["image"] = gt["image"].map(lambda value: Path(str(value)).name)
        df = gt.merge(df, on="image", how="left", suffixes=("_expected", "_predicted"))
        expected_col = "plate_number_expected"
        predicted_col = "plate_number_predicted"
        validity_expected_col = "validity_text_expected"
        validity_predicted_col = "validity_text_predicted"
    else:
        expected_col = "plate_number_expected"
        predicted_col = "plate_number"
        validity_expected_col = "validity_text_expected"
        validity_predicted_col = "validity_text"
        if infer_from_filename:
            df[expected_col] = df["image_path"].map(plate_from_ocr_filename)
            df[validity_expected_col] = df["image_path"].map(validity_from_ocr_filename)
        else:
            df[expected_col] = None
            df[validity_expected_col] = None

    df["plate_expected_norm"] = df[expected_col].map(normalize_plate_text)
    df["plate_predicted_norm"] = df[predicted_col].map(normalize_plate_text)
    df["validity_expected_norm"] = df[validity_expected_col].map(normalize_validity_text)
    df["validity_predicted_norm"] = df[validity_predicted_col].map(normalize_validity_text)
    df["has_plate_ground_truth"] = df["plate_expected_norm"] != ""
    df["has_validity_ground_truth"] = df["validity_expected_norm"] != ""
    df["detected"] = df.get("detection_confidence", pd.Series(index=df.index)).notna()
    df["plate_exact"] = df["has_plate_ground_truth"] & (
        df["plate_expected_norm"] == df["plate_predicted_norm"]
    )
    df["validity_exact"] = df["has_validity_ground_truth"] & (
        df["validity_expected_norm"] == df["validity_predicted_norm"]
    )

    plate_eval = df[df["has_plate_ground_truth"]].copy()
    char_records = zip(plate_eval["plate_expected_norm"], plate_eval["plate_predicted_norm"])
    char_rows = character_accuracy(char_records)

    plate_total = len(plate_eval)
    plate_true = int(plate_eval["plate_exact"].sum())
    summary_rows = [
        {
            "Characters": row.category,
            "Data Total": row.total,
            "Error": row.errors,
            "Number of True": row.true,
            "Accuracy Rate": row.accuracy,
        }
        for row in char_rows
    ]
    summary_rows.append(
        {
            "Characters": "Plate",
            "Data Total": plate_total,
            "Error": plate_total - plate_true,
            "Number of True": plate_true,
            "Accuracy Rate": (plate_true / plate_total * 100.0) if plate_total else 0.0,
        }
    )
    summary = pd.DataFrame(summary_rows)
    return df, summary
