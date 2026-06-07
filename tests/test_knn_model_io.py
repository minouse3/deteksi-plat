from __future__ import annotations

import cv2
import numpy as np

from src.knn_ocr import KNNOCR


def write_char(path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def test_knn_ocr_save_and_load(tmp_path):
    training_dir = tmp_path / "training"
    zero = np.full((40, 20), 255, dtype=np.uint8)
    one = np.full((40, 20), 255, dtype=np.uint8)
    cv2.circle(zero, (10, 20), 7, 0, 2)
    cv2.line(one, (10, 8), (10, 32), 0, 2)

    write_char(training_dir / "0" / "zero.png", zero)
    write_char(training_dir / "1" / "one.png", one)

    model_path = tmp_path / "knn.joblib"
    trained = KNNOCR(k=1, digit_only=True).train(str(training_dir))
    trained.save(str(model_path))

    loaded = KNNOCR.load(str(model_path))

    assert loaded.is_trained
    assert loaded.digit_only
    assert loaded.predict_character(zero) == "0"
    assert loaded.predict_character(one) == "1"
