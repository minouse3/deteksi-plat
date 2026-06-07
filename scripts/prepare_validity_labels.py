from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["image", "plate_number", "validity_month", "validity_year", "validity_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and clean custom validity annotation CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", default="datasets/validity_custom/images")
    parser.add_argument("--output", default="datasets/validity_custom/labels.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    df = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    valid_rows = []
    for idx, row in df.iterrows():
        errors = []
        image_path = image_root / str(row["image"])
        if not image_path.exists():
            errors.append("image file missing")
        try:
            month = int(row["validity_month"])
            year = int(row["validity_year"])
            if month < 1 or month > 12:
                errors.append("invalid month")
            if year < 2000 or year > 2099:
                errors.append("invalid year")
        except ValueError:
            errors.append("month/year not numeric")

        if errors:
            print(f"Row {idx}: {', '.join(errors)}")
        else:
            cleaned = row.copy()
            cleaned["validity_month"] = f"{int(row['validity_month']):02d}"
            cleaned["validity_year"] = str(int(row["validity_year"]))
            valid_rows.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(valid_rows, columns=REQUIRED_COLUMNS).to_csv(output_path, index=False)
    print(f"Wrote {len(valid_rows)} cleaned rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
