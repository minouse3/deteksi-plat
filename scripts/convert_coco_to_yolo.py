from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO plate annotations to YOLO format")
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="datasets/plate_detection")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-per-image", action="store_true", help="Merge all COCO boxes in each image into one plate box")
    parser.add_argument("--expand-x", type=float, default=0.08)
    parser.add_argument("--expand-top", type=float, default=0.08)
    parser.add_argument("--expand-bottom", type=float, default=0.45)
    return parser.parse_args()


def find_image(image_dir: Path, file_name: str) -> Path | None:
    direct = image_dir / file_name
    if direct.exists():
        return direct
    matches = list(image_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _merge_annotations(
    annotations: list[dict],
    image_width: float,
    image_height: float,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
) -> list[list[float]]:
    boxes = [ann.get("bbox", [0, 0, 0, 0]) for ann in annotations]
    boxes = [box for box in boxes if len(box) == 4 and box[2] > 0 and box[3] > 0]
    if not boxes:
        return []

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    width = x2 - x1
    height = y2 - y1

    x1 = _clip(x1 - width * expand_x, 0, image_width)
    x2 = _clip(x2 + width * expand_x, 0, image_width)
    y1 = _clip(y1 - height * expand_top, 0, image_height)
    y2 = _clip(y2 + height * expand_bottom, 0, image_height)
    return [[x1, y1, x2 - x1, y2 - y1]]


def convert(
    coco_json: Path,
    image_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    merge_per_image: bool = False,
    expand_x: float = 0.08,
    expand_top: float = 0.08,
    expand_bottom: float = 0.45,
) -> None:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(ann)

    items = list(images.values())
    random.Random(seed).shuffle(items)
    split_index = int(len(items) * (1 - val_ratio))
    splits = {"train": items[:split_index], "val": items[split_index:]}

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, split_images in splits.items():
        for image_info in tqdm(split_images, desc=f"Writing {split}"):
            source = find_image(image_dir, image_info["file_name"])
            if source is None:
                continue
            target_image = output_dir / "images" / split / source.name
            shutil.copy2(source, target_image)

            width = float(image_info.get("width") or 1)
            height = float(image_info.get("height") or 1)
            lines = []
            annotations = annotations_by_image.get(image_info["id"], [])
            boxes = (
                _merge_annotations(annotations, width, height, expand_x, expand_top, expand_bottom)
                if merge_per_image
                else [ann.get("bbox", [0, 0, 0, 0]) for ann in annotations]
            )
            for x, y, w, h in boxes:
                if w <= 0 or h <= 0:
                    continue
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {w / width:.6f} {h / height:.6f}")
            (output_dir / "labels" / split / f"{source.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "license_plate"},
    }
    (output_dir / "plate_dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    convert(
        Path(args.coco_json),
        Path(args.image_dir),
        Path(args.output_dir),
        args.val_ratio,
        args.seed,
        merge_per_image=args.merge_per_image,
        expand_x=args.expand_x,
        expand_top=args.expand_top,
        expand_bottom=args.expand_bottom,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
