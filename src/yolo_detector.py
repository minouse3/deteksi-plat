from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PLATE_CROP_PADDING, YOLO_CONF_THRESHOLD


@dataclass
class Detection:
    bbox: list[int]
    confidence: float


class YOLOPlateDetector:
    def __init__(self, model_path: str, conf_threshold: float = YOLO_CONF_THRESHOLD):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def detect_plate(self, image: np.ndarray) -> dict | None:
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        best: Detection | None = None

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.conf_threshold:
                    continue
                xyxy = box.xyxy[0].detach().cpu().numpy()
                bbox = [int(round(v)) for v in xyxy.tolist()]
                if best is None or confidence > best.confidence:
                    best = Detection(bbox=bbox, confidence=confidence)

        if best is None:
            return None
        return {"bbox": best.bbox, "confidence": float(best.confidence)}

    @staticmethod
    def crop_plate(
        image: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
        padding: int = PLATE_CROP_PADDING,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            return image[0:0, 0:0].copy()
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def crop_plate_with_context(
        image: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
        padding: int = PLATE_CROP_PADDING,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        x_pad = max(padding, int(box_width * 0.08))
        top_pad = max(padding, int(box_height * 0.08))
        bottom_pad = max(padding, int(box_height * 0.45))
        x1 = max(0, x1 - x_pad)
        y1 = max(0, y1 - top_pad)
        x2 = min(width, x2 + x_pad)
        y2 = min(height, y2 + bottom_pad)
        if x2 <= x1 or y2 <= y1:
            return image[0:0, 0:0].copy()
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def draw_detection(
        image: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
        confidence: float,
    ) -> np.ndarray:
        output = image.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"plate {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return output
