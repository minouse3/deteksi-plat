# Evaluation

Use `ground_truth.csv` to score batch inference output.

Run inference over the current test images:

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/batch_inference.py \
  --image-dir tests \
  --yolo-model models/yolov8m_plate_detector.pt \
  --training datasets/ocr_chars \
  --output-dir outputs/evaluation/test_images \
  --limit 0 \
  --digit-only-validity \
  --knn-k 1
```

Score predictions:

```bash
python scripts/score_predictions.py \
  --predictions outputs/evaluation/test_images/results.csv \
  --ground-truth tests/evaluation/ground_truth.csv \
  --output outputs/evaluation/score_report.csv
```

Metrics:

- `detected`: YOLO returned a plate.
- `plate_exact`: normalized plate number exactly matches ground truth.
- `validity_exact`: parsed validity text exactly matches ground truth.
- `status_exact`: `valid`, `expired`, or `unknown` exactly matches ground truth.
- `all_exact`: plate, validity, and status are all correct.
