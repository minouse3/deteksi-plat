from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MIN_COMPONENT_AREA_UPPER
from src.preprocessing import otsu_threshold, remove_noise, resize_plate, to_grayscale
from src.segmentation import segment_characters, split_upper_lower_regions
from src.yolo_detector import YOLOPlateDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create KNN OCR character folders from labeled plate images")
    parser.add_argument("--labels", default="datasets/plate_ocr/labels.csv")
    parser.add_argument("--image-root", default="datasets/kaggle_raw/plate_text_dataset/plate_text_dataset/dataset")
    parser.add_argument("--output-dir", default="datasets/ocr_chars")
    parser.add_argument("--yolo-model", default="models/yolov8m_plate_detector.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all rows")
    return parser.parse_args()


def normalize_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def find_image(image_root: Path, filename: str) -> Path | None:
    direct = image_root / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    candidates = list(image_root.glob(f"{stem}.*"))
    if candidates:
        return candidates[0]

    # Roboflow exports often prefix labels in generated filenames.
    fuzzy = list(image_root.glob(f"*{stem}*"))
    return fuzzy[0] if fuzzy else None


def save_character(output_dir: Path, label: str, image, source_stem: str, index: int) -> None:
    target_dir = output_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source_stem}_{index:02d}.png"
    cv2.imwrite(str(target), image)


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLOPlateDetector(args.yolo_model, conf_threshold=args.conf_threshold) if Path(args.yolo_model).exists() else None

    df = pd.read_csv(labels_path)
    filename_col = "filename" if "filename" in df.columns else "image"
    label_col = "label" if "label" in df.columns else "plate_number"
    if filename_col not in df.columns or label_col not in df.columns:
        raise ValueError("Labels CSV must contain filename/image and label/plate_number columns")

    if args.max_images > 0:
        df = df.head(args.max_images)

    written = 0
    skipped_missing = 0
    skipped_mismatch = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing OCR chars"):
        label = normalize_label(row[label_col])
        if not label:
            continue

        image_path = find_image(image_root, str(row[filename_col]))
        if image_path is None:
            skipped_missing += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped_missing += 1
            continue

        plate = image
        if detector is not None:
            detection = detector.detect_plate(image)
            if detection is None:
                skipped_missing += 1
                continue
            plate = detector.crop_plate(image, detection["bbox"], padding=10)

        plate = resize_plate(plate, width=500)
        binary = otsu_threshold(to_grayscale(plate))
        cleaned = remove_noise(binary, min_area=MIN_COMPONENT_AREA_UPPER)
        upper, _ = split_upper_lower_regions(cleaned)
        chars = segment_characters(upper, min_area=MIN_COMPONENT_AREA_UPPER)

        if len(chars) != len(label):
            skipped_mismatch += 1
            continue

        for index, (char_label, char_image) in enumerate(zip(label, chars), start=1):
            save_character(output_dir, char_label, char_image, image_path.stem, index)
            written += 1

    print(f"Wrote {written} character images to {output_dir}")
    print(f"Skipped missing/unreadable images: {skipped_missing}")
    print(f"Skipped segmentation-label mismatches: {skipped_mismatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
