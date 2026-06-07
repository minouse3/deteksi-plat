# Project Guidance for Codex

- Use Python 3.11.
- Keep code modular and focused.
- Use YOLO for full-image license plate detection.
- Use Otsu thresholding, segmentation, and KNN as the article-inspired OCR baseline.
- Detect upper and lower plate regions.
- Treat the upper region as the main license plate number.
- Treat the lower region as the validity period.
- Use digit-only OCR for the lower validity-period region.
- Save debug images for every major stage when running the pipeline.
- Run `pytest` after code changes.
- Avoid hardcoded dataset paths. Prefer CLI arguments and configurable defaults.
