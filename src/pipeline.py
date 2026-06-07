from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import DEBUG_OUTPUT_DIR, MIN_COMPONENT_AREA_LOWER, MIN_COMPONENT_AREA_UPPER, PLATE_CROP_PADDING
from .date_validation import check_validity
from .preprocessing import deskew_or_align_plate, load_image, otsu_threshold, remove_noise, resize_plate, to_grayscale
from .plate_number import normalize_plate_number
from .segmentation import segment_characters, segment_validity_characters, split_upper_lower_regions
from .utils import ensure_dir, save_image
from .validity_detection import parse_validity_text


def _save_segments(directory: Path, segments: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, segment in enumerate(segments, start=1):
        cv2.imwrite(str(directory / f"char_{index:02d}.jpg"), _debug_segment_preview(segment))


def _debug_segment_preview(segment) -> object:
    """Save OCR crops as readable black-on-white previews."""
    if segment is None or segment.size == 0:
        return segment

    preview = segment.copy()
    if len(preview.shape) == 3:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    unique_values = set(int(value) for value in cv2.minMaxLoc(preview)[:2])
    if len(unique_values) > 1:
        _, preview = cv2.threshold(preview, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border_pixels = np.concatenate([preview[0, :], preview[-1, :], preview[:, 0], preview[:, -1]])
    border_white_ratio = (border_pixels == 255).mean() if border_pixels.size else 0.0

    # Debug images are easier to inspect as black glyphs on white paper.
    if border_white_ratio < 0.5:
        preview = cv2.bitwise_not(preview)

    pad_y = max(4, preview.shape[0] // 12)
    pad_x = max(4, preview.shape[1] // 8)
    return cv2.copyMakeBorder(preview, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)


def process_vehicle_image(image_path: str, detector, plate_ocr, validity_ocr=None, debug_dir: str = DEBUG_OUTPUT_DIR) -> dict:
    debug_root = ensure_dir(debug_dir)
    image = load_image(image_path)
    save_image(debug_root / "01_original.jpg", image)

    detection = detector.detect_plate(image)
    if detection is None:
        return {
            "image_path": image_path,
            "plate_number": None,
            "validity_text": None,
            "validity_month": None,
            "validity_year": None,
            "validity_status": "unknown",
            "plate_bbox": None,
            "detection_confidence": None,
            "error": "No plate detected",
            "debug": {"plate_detected": False},
        }

    detected = detector.draw_detection(image, detection["bbox"], detection["confidence"])
    save_image(debug_root / "02_detected_plate_bbox.jpg", detected)

    if hasattr(detector, "crop_plate_with_context"):
        crop = detector.crop_plate_with_context(image, detection["bbox"], padding=PLATE_CROP_PADDING)
    else:
        crop = detector.crop_plate(image, detection["bbox"], padding=PLATE_CROP_PADDING)
    if crop.size == 0:
        return {
            "image_path": image_path,
            "plate_number": None,
            "validity_status": "unknown",
            "error": "Invalid plate crop",
            "plate_bbox": detection["bbox"],
            "detection_confidence": detection["confidence"],
        }

    crop = deskew_or_align_plate(crop)
    crop = resize_plate(crop)
    save_image(debug_root / "03_cropped_plate.jpg", crop)
    gray = to_grayscale(crop)
    save_image(debug_root / "04_grayscale_plate.jpg", gray)
    binary = otsu_threshold(gray)
    save_image(debug_root / "05_binary_plate.jpg", binary)
    cleaned = remove_noise(binary, min_area=MIN_COMPONENT_AREA_LOWER)
    save_image(debug_root / "06_cleaned_binary_plate.jpg", cleaned)

    upper, lower = split_upper_lower_regions(cleaned)
    if upper is not None:
        save_image(debug_root / "07_upper_region.jpg", upper)
    if lower is not None:
        save_image(debug_root / "08_lower_validity_region.jpg", lower)

    upper_chars = segment_characters(upper, min_area=MIN_COMPONENT_AREA_UPPER)
    lower_chars = segment_validity_characters(lower, min_area=MIN_COMPONENT_AREA_LOWER)
    _save_segments(debug_root / "09_upper_segmented_chars", upper_chars)
    _save_segments(debug_root / "10_lower_segmented_digits", lower_chars)

    try:
        raw_plate_number = plate_ocr.predict_sequence(upper_chars) if upper_chars else None
        plate_number = normalize_plate_number(raw_plate_number)
        validity_reader = validity_ocr or plate_ocr
        raw_validity = validity_reader.predict_sequence(lower_chars) if lower_chars else ""
    except Exception:
        return {
            "image_path": image_path,
            "plate_number": None,
            "validity_text": None,
            "validity_month": None,
            "validity_year": None,
            "validity_status": "unknown",
            "plate_bbox": detection["bbox"],
            "detection_confidence": detection["confidence"],
            "error": "OCR failed",
            "debug": {
                "plate_detected": True,
                "upper_region_found": upper is not None and upper.size > 0,
                "lower_region_found": lower is not None and lower.size > 0,
                "num_plate_chars": len(upper_chars),
                "num_validity_chars": len(lower_chars),
            },
        }

    parsed = parse_validity_text(raw_validity)
    status = check_validity(parsed["month"], parsed["year"])

    return {
        "image_path": image_path,
        "plate_number": plate_number,
        "raw_plate_number": raw_plate_number,
        "validity_text": parsed["normalized"],
        "raw_validity_text": raw_validity,
        "validity_month": parsed["month"],
        "validity_year": parsed["year"],
        "validity_status": status,
        "plate_bbox": detection["bbox"],
        "detection_confidence": detection["confidence"],
        "error": None,
        "debug": {
            "plate_detected": True,
            "upper_region_found": upper is not None and upper.size > 0,
            "lower_region_found": lower is not None and lower.size > 0,
            "num_plate_chars": len(upper_chars),
            "num_validity_chars": len(lower_chars),
        },
    }
