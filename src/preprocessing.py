from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def resize_plate(image: np.ndarray, width: int = 1000) -> np.ndarray:
    if image.size == 0:
        return image
    height, current_width = image.shape[:2]
    if current_width == 0 or current_width == width:
        return image.copy()
    scale = width / current_width
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_CUBIC)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return maybe_invert_binary(binary)


def maybe_invert_binary(binary: np.ndarray) -> np.ndarray:
    white_ratio = float(np.count_nonzero(binary == 255)) / binary.size if binary.size else 0.0
    if binary.size:
        border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        border_white_ratio = float(np.count_nonzero(border == 255)) / border.size
    else:
        border_white_ratio = 0.0
    if white_ratio > 0.65 or border_white_ratio > 0.55:
        return cv2.bitwise_not(binary)
    return binary


def remove_noise(binary: np.ndarray, min_area: int = 30) -> np.ndarray:
    binary = maybe_invert_binary(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)


def deskew_or_align_plate(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image

    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy()

    image_area = image.shape[0] * image.shape[1]
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image_area * 0.05:
        return image.copy()

    rect = cv2.minAreaRect(contour)
    (_, _), (rect_width, rect_height), angle = rect
    if rect_width <= 0 or rect_height <= 0:
        return image.copy()

    if rect_width < rect_height:
        angle += 90
    if abs(angle) < 1.0 or abs(angle) > 20.0:
        return image.copy()

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
