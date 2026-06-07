from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEBUG_OUTPUT_DIR, KNN_K, YOLO_CONF_THRESHOLD
from src.knn_ocr import KNNOCR
from src.pipeline import process_vehicle_image
from src.utils import print_json
from src.yolo_detector import YOLOPlateDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indonesian license plate validity recognition")
    parser.add_argument("--image", required=True, help="Path to full vehicle image")
    parser.add_argument("--yolo-model", required=True, help="Path to trained YOLO plate detector .pt model")
    parser.add_argument("--training", help="OCR character training directory")
    parser.add_argument("--plate-ocr-model", help="Saved KNN OCR .joblib model for plate number OCR")
    parser.add_argument("--validity-ocr-model", help="Saved KNN OCR .joblib model for validity digit OCR")
    parser.add_argument("--debug-dir", default=DEBUG_OUTPUT_DIR, help="Directory for debug images")
    parser.add_argument("--conf-threshold", type=float, default=YOLO_CONF_THRESHOLD)
    parser.add_argument("--knn-k", type=int, default=KNN_K, help="Number of neighbors for KNN OCR")
    parser.add_argument("--digit-only-validity", action="store_true", help="Train separate digit-only KNN for validity OCR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector = YOLOPlateDetector(args.yolo_model, conf_threshold=args.conf_threshold)

    if args.plate_ocr_model:
        plate_ocr = KNNOCR.load(args.plate_ocr_model)
    elif args.training:
        plate_ocr = KNNOCR(k=args.knn_k, digit_only=False).train(args.training)
    else:
        raise ValueError("Provide either --plate-ocr-model or --training")

    if args.validity_ocr_model:
        validity_ocr = KNNOCR.load(args.validity_ocr_model)
    elif args.digit_only_validity and args.training:
        validity_ocr = KNNOCR(k=args.knn_k, digit_only=True).train(args.training)
    else:
        validity_ocr = None

    result = process_vehicle_image(args.image, detector, plate_ocr, validity_ocr=validity_ocr, debug_dir=args.debug_dir)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
