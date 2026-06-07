from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import KNN_K
from src.knn_ocr import KNNOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save KNN OCR models as .joblib files")
    parser.add_argument("--training", default="datasets/ocr_chars", help="OCR character training directory")
    parser.add_argument("--plate-output", default="models/knn_plate_ocr.joblib")
    parser.add_argument("--validity-output", default="models/knn_validity_ocr.joblib")
    parser.add_argument("--knn-k", type=int, default=KNN_K)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plate_ocr = KNNOCR(k=args.knn_k, digit_only=False).train(args.training)
    plate_ocr.save(args.plate_output)
    print(f"Saved plate OCR KNN model to {args.plate_output}")

    validity_ocr = KNNOCR(k=args.knn_k, digit_only=True).train(args.training)
    validity_ocr.save(args.validity_output)
    print(f"Saved validity OCR KNN model to {args.validity_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
