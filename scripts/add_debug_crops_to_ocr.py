from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add segmented debug crops into OCR character folders")
    parser.add_argument("--crops-dir", required=True, help="Directory containing char_01.jpg, char_02.jpg, ...")
    parser.add_argument("--labels", required=True, help="Character labels in left-to-right crop order")
    parser.add_argument("--output-dir", default="datasets/ocr_chars")
    parser.add_argument("--prefix", default="manual")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.output_dir)
    labels = [char.upper() for char in args.labels if char.strip()]
    crops = sorted(crops_dir.glob("char_*.jpg"))

    if len(crops) != len(labels):
        raise ValueError(f"Crop count ({len(crops)}) does not match label count ({len(labels)}): {args.labels}")

    for index, (crop, label) in enumerate(zip(crops, labels), start=1):
        target_dir = output_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{args.prefix}_{crops_dir.parent.name}_{crops_dir.name}_{index:02d}{crop.suffix.lower()}"
        shutil.copy2(crop, target)
        print(f"{crop} -> {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
