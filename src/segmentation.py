from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import maybe_invert_binary, otsu_threshold, to_grayscale


def trim_plate_border(binary_image: np.ndarray) -> np.ndarray:
    if binary_image is None or binary_image.size == 0:
        return binary_image

    binary = maybe_invert_binary(binary_image)
    height, width = binary.shape[:2]
    if height < 20 or width < 20:
        return binary

    row_density = np.count_nonzero(binary == 255, axis=1) / float(width)
    col_density = np.count_nonzero(binary == 255, axis=0) / float(height)

    top = 0
    while top < height * 0.20 and row_density[top] > 0.45:
        top += 1

    bottom = height - 1
    while bottom > height * 0.80 and row_density[bottom] > 0.45:
        bottom -= 1

    left = 0
    while left < width * 0.10 and col_density[left] > 0.35:
        left += 1

    right = width - 1
    while right > width * 0.90 and col_density[right] > 0.35:
        right -= 1

    if bottom <= top or right <= left:
        return binary
    return binary[top : bottom + 1, left : right + 1]


def horizontal_projection(binary_image: np.ndarray) -> np.ndarray:
    binary = maybe_invert_binary(binary_image)
    return np.count_nonzero(binary == 255, axis=1)


def _row_groups(mask: np.ndarray, min_gap: int = 3) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for idx, active in enumerate(mask):
        if active:
            if start is None:
                start = idx
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                groups.append((start, idx - gap + 1))
                start = None
                gap = 0
    if start is not None:
        groups.append((start, len(mask)))
    return groups


def find_text_bands(binary_image: np.ndarray) -> list[tuple[int, int]]:
    projection = horizontal_projection(binary_image)
    if projection.size == 0 or projection.max() == 0:
        return []
    threshold = max(2, int(projection.max() * 0.12))
    mask = projection >= threshold
    groups = _row_groups(mask)
    min_height = max(3, binary_image.shape[0] // 30)
    bands = [(start, end) for start, end in groups if end - start >= min_height]
    return sorted(bands, key=lambda band: band[0])


def split_upper_lower_regions(plate_image_or_binary: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(plate_image_or_binary.shape) == 3:
        binary = otsu_threshold(to_grayscale(plate_image_or_binary))
    else:
        binary = maybe_invert_binary(plate_image_or_binary)
    binary = trim_plate_border(binary)

    height, width = binary.shape[:2]
    if height > 0 and width > 0:
        # Indonesian plates have a stable two-row layout. Crops from full-plate YOLO
        # boxes are more reliable with physical zones than with raw projections,
        # because borders, screws, glare, and plate frames often dominate projection.
        upper = binary[int(height * 0.08) : int(height * 0.58), :]
        lower = binary[int(height * 0.48) : int(height * 0.88), :]
        if np.count_nonzero(upper == 255) > 0 and np.count_nonzero(lower == 255) > 0:
            return upper, lower

    bands = find_text_bands(binary)
    if not bands:
        height = binary.shape[0]
        return binary[: int(height * 0.65), :], binary[int(height * 0.65) :, :]

    if len(bands) == 1:
        start, end = bands[0]
        midpoint = min(binary.shape[0], end + max(1, (binary.shape[0] - end) // 2))
        return binary[start:midpoint, :], binary[midpoint:, :]

    scored = []
    for start, end in bands:
        area = int(np.count_nonzero(binary[start:end, :] == 255))
        width_coverage = float(np.count_nonzero(np.count_nonzero(binary[start:end, :] == 255, axis=0))) / binary.shape[1]
        band_height = end - start
        # Plate borders often form dense full-width bands. They are not text.
        if width_coverage > 0.85 and band_height < binary.shape[0] * 0.25:
            continue
        scored.append((area, start, end))

    if not scored:
        height = binary.shape[0]
        return binary[: int(height * 0.65), :], binary[int(height * 0.65) :, :]

    upper_candidates = [item for item in scored if item[1] < binary.shape[0] * 0.75]
    upper_area, upper_start, upper_end = max(upper_candidates or scored, key=lambda item: item[0])
    lower_bands = [(start, end) for _, start, end in scored if start > upper_start]

    if lower_bands:
        lower_start, lower_end = max(lower_bands, key=lambda band: band[1] - band[0])
    else:
        lower_start = upper_end
        lower_end = binary.shape[0]

    pad = max(2, binary.shape[0] // 40)
    upper = binary[max(0, upper_start - pad) : min(binary.shape[0], upper_end + pad), :]
    lower = binary[max(0, lower_start - pad) : min(binary.shape[0], lower_end + pad), :]
    return upper, lower


def segment_characters(binary_region: np.ndarray, min_area: int = 30) -> list[np.ndarray]:
    if binary_region is None or binary_region.size == 0:
        return []

    if len(np.unique(binary_region)) > 2:
        _, base = cv2.threshold(binary_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        base = binary_region.copy()

    height, width = base.shape[:2]
    candidates = [maybe_invert_binary(base), cv2.bitwise_not(maybe_invert_binary(base))]

    def find_boxes(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            extent = area / float(w * h) if w * h else 0.0
            if h < max(4, height * 0.18):
                continue
            if w < 2:
                continue
            if w > width * 0.35:
                continue
            if h > height * 0.95:
                continue
            if w > width * 0.20 and h < height * 0.35:
                continue
            if extent < 0.08:
                continue
            found.append((x, y, w, h))
        return found

    def projection_boxes(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
        work = candidate.copy()
        top = int(height * 0.12)
        bottom = int(height * 0.90)
        left = int(width * 0.02)
        right = int(width * 0.98)
        work[:top, :] = 0
        work[bottom:, :] = 0
        work[:, :left] = 0
        work[:, right:] = 0

        projection = np.count_nonzero(work == 255, axis=0)
        threshold = max(5, int(height * 0.06))
        mask = projection >= threshold
        groups: list[tuple[int, int]] = []
        start: int | None = None
        gap = 0
        max_gap = max(4, int(width * 0.025))
        for idx, active in enumerate(mask):
            if active:
                if start is None:
                    start = idx
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= max_gap:
                    groups.append((start, idx - gap + 1))
                    start = None
                    gap = 0
        if start is not None:
            groups.append((start, len(mask)))

        found = []
        for x1, x2 in groups:
            if x2 - x1 < 5:
                continue
            if (x1 < width * 0.03 or x2 > width * 0.97) and (x2 - x1) < width * 0.04:
                continue
            column_slice = work[:, x1:x2]
            ys, xs = np.where(column_slice == 255)
            if len(xs) == 0:
                continue
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1
            box_height = y2 - y1
            box_width = x2 - x1
            if box_height < max(4, height * 0.18):
                continue
            found.append((x1, y1, box_width, box_height))
        return found

    scored = []
    for candidate in candidates:
        boxes = find_boxes(candidate)
        projected = projection_boxes(candidate)
        if len(projected) > len(boxes):
            boxes = projected

        # Prefer boxes that span a meaningful portion of the text band. On white
        # plates, the inverted image can expose only the inner holes of glyphs
        # (for example B/8/0) and those holes can outnumber the real characters.
        tall_count = sum(1 for _, _, _, h in boxes if h >= height * 0.38)
        plausible_count = 1 if 4 <= len(boxes) <= 9 else 0
        total_width = sum(w for _, _, w, _ in boxes)
        scored.append((tall_count, plausible_count, len(boxes), -total_width, candidate, boxes))

    _, _, _, _, binary, boxes = max(scored, key=lambda item: (item[0], item[1], item[2], item[3]))

    boxes.sort(key=lambda item: item[0])
    chars: list[np.ndarray] = []
    for x, y, w, h in boxes:
        chars.append(binary[max(0, y - 1) : min(height, y + h + 1), max(0, x - 1) : min(width, x + w + 1)])
    return chars


def segment_validity_characters(binary_region: np.ndarray, min_area: int = 10) -> list[np.ndarray]:
    if binary_region is None or binary_region.size == 0:
        return []

    if len(np.unique(binary_region)) > 2:
        _, base = cv2.threshold(binary_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        base = binary_region.copy()

    height, width = base.shape[:2]

    # White inset panels are common for the lower validity period on newer plates.
    # Crop the panel first, then invert so dark digits become foreground.
    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if y > height * 0.20 and w > width * 0.20 and h > height * 0.25 and area / float(w * h) > 0.45:
            panel_candidates.append((x, y, w, h))

    if panel_candidates:
        x, y, w, h = max(panel_candidates, key=lambda box: box[2] * box[3])
        pad = max(2, int(min(w, h) * 0.04))
        panel = base[max(0, y - pad) : min(height, y + h + pad), max(0, x - pad) : min(width, x + w + pad)]
        return segment_characters(cv2.bitwise_not(panel), min_area=min_area)

    candidate = maybe_invert_binary(base)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if h < height * 0.18 or h > height * 0.65:
            continue
        if w < 3 or w > width * 0.12:
            continue
        if y < height * 0.12:
            continue
        extent = area / float(w * h) if w * h else 0.0
        if extent < 0.12:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda item: item[0])
    if len(boxes) > 4:
        heights = sorted(h for _, _, _, h in boxes)
        median_height = heights[len(heights) // 2]
        boxes = [box for box in boxes if box[3] >= median_height * 0.55]
    chars = []
    for x, y, w, h in boxes:
        chars.append(candidate[max(0, y - 1) : min(height, y + h + 1), max(0, x - 1) : min(width, x + w + 1)])
    return chars
