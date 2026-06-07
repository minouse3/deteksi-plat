from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .config import CHAR_IMAGE_SIZE, KNN_K
from .preprocessing import maybe_invert_binary, otsu_threshold, to_grayscale


class KNNOCR:
    def __init__(self, k: int = KNN_K, char_size: tuple[int, int] = CHAR_IMAGE_SIZE, digit_only: bool = False):
        self.k = k
        self.char_size = char_size
        self.digit_only = digit_only
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.is_trained = False

    def prepare_character_image(self, char_image: np.ndarray) -> np.ndarray:
        gray = to_grayscale(char_image) if len(char_image.shape) == 3 else char_image.copy()
        if len(np.unique(gray)) > 2:
            binary = otsu_threshold(gray)
        else:
            binary = maybe_invert_binary(gray)
        resized = cv2.resize(binary, self.char_size, interpolation=cv2.INTER_AREA)
        return (resized.flatten() / 255.0).astype(np.float32)

    def load_training_data(self, training_dir: str) -> tuple[np.ndarray, np.ndarray]:
        root = Path(training_dir)
        if not root.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        features: list[np.ndarray] = []
        labels: list[str] = []
        allowed = set("0123456789") if self.digit_only else set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            label = class_dir.name.upper()
            if label not in allowed:
                continue
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                features.append(self.prepare_character_image(image))
                labels.append(label)

        if not features:
            raise ValueError(f"No OCR training images found in {training_dir}")
        return np.vstack(features), np.array(labels)

    def train(self, training_dir: str) -> "KNNOCR":
        features, labels = self.load_training_data(training_dir)
        n_neighbors = min(self.k, len(labels))
        if n_neighbors != self.classifier.n_neighbors:
            self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classifier.fit(features, labels)
        self.is_trained = True
        return self

    def save(self, model_path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained KNN OCR model")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "k": self.k,
                "char_size": self.char_size,
                "digit_only": self.digit_only,
            },
            path,
        )

    @classmethod
    def load(cls, model_path: str) -> "KNNOCR":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"KNN OCR model not found: {model_path}")
        payload = joblib.load(path)
        ocr = cls(
            k=int(payload["k"]),
            char_size=tuple(payload["char_size"]),
            digit_only=bool(payload["digit_only"]),
        )
        ocr.classifier = payload["classifier"]
        ocr.is_trained = True
        return ocr

    def predict_character(self, char_image: np.ndarray) -> str:
        if not self.is_trained:
            raise RuntimeError("KNN OCR model is not trained")
        features = self.prepare_character_image(char_image).reshape(1, -1)
        prediction = str(self.classifier.predict(features)[0])
        if self.digit_only and not prediction.isdigit():
            return ""
        return prediction

    def predict_sequence(self, char_images: list[np.ndarray]) -> str:
        return "".join(self.predict_character(char_image) for char_image in char_images)
