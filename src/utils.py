from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_image(path: str | Path, image) -> None:
    if image is None or getattr(image, "size", 0) == 0:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))
