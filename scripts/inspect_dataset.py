from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect downloaded Kaggle dataset layout")
    parser.add_argument("--root", default="datasets/kaggle_raw")
    parser.add_argument("--max-depth", type=int, default=4)
    return parser.parse_args()


def tree_summary(root: Path, max_depth: int) -> None:
    print(f"Dataset root: {root.resolve()}")
    if not root.exists():
        print("Root does not exist.")
        return
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        if path.is_dir():
            image_count = sum(1 for child in path.iterdir() if child.suffix.lower() in IMAGE_EXTS) if path.exists() else 0
            print(f"{indent}[D] {path.name}/ images={image_count}")
        else:
            print(f"{indent}[F] {path.name}")


def find_candidates(root: Path) -> None:
    image_dirs = []
    coco_files = []
    label_files = []
    for path in root.rglob("*"):
        if path.is_dir() and any(child.suffix.lower() in IMAGE_EXTS for child in path.iterdir()):
            image_dirs.append(path)
        elif path.is_file():
            name = path.name.lower()
            if path.suffix.lower() == ".json":
                text = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                if '"images"' in text and '"annotations"' in text:
                    coco_files.append(path)
            if path.suffix.lower() in {".csv", ".txt"}:
                label_files.append(path)

    print("\nLikely image folders:")
    for path in image_dirs:
        print(f"- {path}")
    print("\nLikely COCO JSON files:")
    for path in coco_files:
        print(f"- {path}")
    print("\nLikely CSV/TXT label files:")
    for path in label_files:
        print(f"- {path}")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    tree_summary(root, args.max_depth)
    if root.exists():
        find_candidates(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
