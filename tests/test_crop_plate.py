import numpy as np

from src.yolo_detector import YOLOPlateDetector


def test_crop_plate_clamps_bbox_to_boundaries():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YOLOPlateDetector.crop_plate(image, [-10, -10, 30, 40], padding=10)
    assert crop.shape == (50, 40, 3)


def test_crop_plate_returns_non_empty_for_valid_bbox():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YOLOPlateDetector.crop_plate(image, [20, 20, 50, 60], padding=0)
    assert crop.size > 0
    assert crop.shape == (40, 30, 3)


def test_crop_plate_handles_padding():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YOLOPlateDetector.crop_plate(image, [20, 20, 50, 60], padding=10)
    assert crop.shape == (60, 50, 3)
