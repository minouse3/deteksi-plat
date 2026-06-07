from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle dataset for plate detection and OCR")
    parser.add_argument("--raw-root", default="datasets/kaggle_raw")
    parser.add_argument("--detection-output", default="datasets/plate_detection")
    parser.add_argument("--ocr-output", default="datasets/plate_ocr")
    return parser.parse_args()


def find_coco_json(root: Path) -> Path | None:
    for path in root.rglob("*.json"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError:
            continue
        if '"images"' in text and '"annotations"' in text:
            return path
    return None


def likely_image_root(coco_json: Path, raw_root: Path) -> Path:
    sibling_images = coco_json.parent.parent / "images"
    if sibling_images.exists():
        return sibling_images

    try:
        data = json.loads(coco_json.read_text(encoding="utf-8"))
        first = data.get("images", [{}])[0].get("file_name", "")
    except Exception:
        first = ""
    for directory in [coco_json.parent, raw_root, *raw_root.rglob("*")]:
        if directory.is_dir() and (directory / first).exists():
            return directory
    return raw_root


def create_empty_detection_yaml(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    payload = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "license_plate"}}
    (output / "plate_dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare_detection(raw_root: Path, output: Path) -> None:
    coco = find_coco_json(raw_root)
    if coco is None:
        print("No COCO JSON found. Created empty YOLO detection layout.")
        create_empty_detection_yaml(output)
        return
    image_root = likely_image_root(coco, raw_root)
    script = Path(__file__).with_name("convert_coco_to_yolo.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--coco-json",
            str(coco),
            "--image-dir",
            str(image_root),
            "--output-dir",
            str(output),
            "--merge-per-image",
        ],
        check=False,
    )


def prepare_ocr(raw_root: Path, output: Path) -> None:
    (output / "images").mkdir(parents=True, exist_ok=True)
    csv_files = list(raw_root.rglob("*.csv"))
    copied_csv = False
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        lower_columns = {column.lower(): column for column in df.columns}
        if any(name in lower_columns for name in ["plate_number", "text", "label", "license_plate"]):
            df.to_csv(output / "labels.csv", index=False)
            copied_csv = True
            print(f"Copied likely OCR labels from {csv_path}")
            break
    if not copied_csv:
        pd.DataFrame(columns=["image", "plate_number"]).to_csv(output / "labels.csv", index=False)
        print("No OCR CSV found. Created empty plate_ocr/labels.csv.")

    copied = 0
    for image in raw_root.rglob("*"):
        if image.suffix.lower() in IMAGE_EXTS and copied < 200:
            shutil.copy2(image, output / "images" / image.name)
            copied += 1
    print(f"Copied {copied} sample OCR images into {output / 'images'}")


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        print(f"Raw dataset root not found: {raw_root}")
        return 0
    prepare_detection(raw_root, Path(args.detection_output))
    prepare_ocr(raw_root, Path(args.ocr_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
