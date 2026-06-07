from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import KNN_K, YOLO_CONF_THRESHOLD
from src.knn_ocr import KNNOCR
from src.pipeline import process_vehicle_image
from src.yolo_detector import YOLOPlateDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch license plate validity inference")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--yolo-model", default="models/yolov8m_plate_detector.pt")
    parser.add_argument("--training", default="datasets/ocr_chars")
    parser.add_argument("--output-dir", default="outputs/batch")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-threshold", type=float, default=YOLO_CONF_THRESHOLD)
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    parser.add_argument("--digit-only-validity", action="store_true")
    return parser.parse_args()


def collect_images(image_dir: Path, limit: int, seed: int) -> list[Path]:
    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS]
    images = sorted(images)
    if limit > 0 and len(images) > limit:
        rng = random.Random(seed)
        images = rng.sample(images, limit)
        images.sort()
    return images


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOPlateDetector(args.yolo_model, conf_threshold=args.conf_threshold)
    plate_ocr = KNNOCR(k=args.knn_k, digit_only=False).train(args.training)
    validity_ocr = KNNOCR(k=args.knn_k, digit_only=True).train(args.training) if args.digit_only_validity else None

    rows = []
    images = collect_images(Path(args.image_dir), args.limit, args.seed)
    jsonl_path = output_dir / "results.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for index, image_path in enumerate(tqdm(images, desc="Batch inference"), start=1):
            debug_dir = output_dir / "debug" / f"{index:04d}_{image_path.stem}"
            result = process_vehicle_image(
                str(image_path),
                detector,
                plate_ocr,
                validity_ocr=validity_ocr,
                debug_dir=str(debug_dir),
            )
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            rows.append(
                {
                    "image_path": result.get("image_path"),
                    "plate_number": result.get("plate_number"),
                    "raw_plate_number": result.get("raw_plate_number"),
                    "validity_text": result.get("validity_text"),
                    "raw_validity_text": result.get("raw_validity_text"),
                    "validity_status": result.get("validity_status"),
                    "detection_confidence": result.get("detection_confidence"),
                    "num_plate_chars": (result.get("debug") or {}).get("num_plate_chars"),
                    "num_validity_chars": (result.get("debug") or {}).get("num_validity_chars"),
                    "error": result.get("error"),
                    "debug_dir": str(debug_dir),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    print(f"Wrote {len(rows)} results to {output_dir / 'results.csv'}")
    print(f"Wrote JSONL to {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
