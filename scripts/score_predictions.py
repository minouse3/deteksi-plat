from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score batch inference predictions against ground-truth CSV")
    parser.add_argument("--predictions", required=True, help="Batch inference results.csv")
    parser.add_argument("--ground-truth", default="tests/evaluation/ground_truth.csv")
    parser.add_argument("--output", default="outputs/evaluation/score_report.csv")
    return parser.parse_args()


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace(" ", "")


def image_key(path: str) -> str:
    return Path(str(path)).name


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    ground_truth = pd.read_csv(args.ground_truth)

    predictions["image"] = predictions["image_path"].map(image_key)
    merged = ground_truth.merge(predictions, on="image", how="left", suffixes=("_expected", "_predicted"))

    merged["plate_expected_norm"] = merged["plate_number_expected"].map(normalize_text)
    merged["plate_predicted_norm"] = merged["plate_number_predicted"].map(normalize_text)
    merged["validity_expected_norm"] = merged["validity_text_expected"].map(normalize_text)
    merged["validity_predicted_norm"] = merged["validity_text_predicted"].map(normalize_text)
    merged["status_expected_norm"] = merged["validity_status_expected"].map(normalize_text)
    merged["status_predicted_norm"] = merged["validity_status_predicted"].map(normalize_text)

    merged["detected"] = merged["error"].isna() & merged["image_path"].notna()
    merged["plate_exact"] = merged["plate_expected_norm"] == merged["plate_predicted_norm"]
    merged["validity_exact"] = merged["validity_expected_norm"] == merged["validity_predicted_norm"]
    merged["status_exact"] = merged["status_expected_norm"] == merged["status_predicted_norm"]
    merged["all_exact"] = merged["plate_exact"] & merged["validity_exact"] & merged["status_exact"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    total = len(merged)
    print(f"Images: {total}")
    for column, label in [
        ("detected", "Detection"),
        ("plate_exact", "Plate exact"),
        ("validity_exact", "Validity exact"),
        ("status_exact", "Status exact"),
        ("all_exact", "All exact"),
    ]:
        count = int(merged[column].sum())
        percent = (count / total * 100) if total else 0.0
        print(f"{label}: {count}/{total} ({percent:.1f}%)")

    misses = merged[~merged["all_exact"]]
    if not misses.empty:
        print("\nMismatches:")
        columns = [
            "image",
            "plate_number_expected",
            "plate_number_predicted",
            "validity_text_expected",
            "validity_text_predicted",
            "validity_status_expected",
            "validity_status_predicted",
            "error",
        ]
        print(misses[columns].to_string(index=False))

    print(f"\nWrote score report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
