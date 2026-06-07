# Indonesian License Plate Validity Recognition

A computer-vision project for Indonesian vehicle license plate recognition and validity-period checking. The system uses YOLOv8 for detection and KNN for OCR to report whether a plate is valid, expired, or unknown.

---

## Quick Start (Using Docker)

If you just want to run the web app and test images, use Docker. It handles all system dependencies for you.

1.  **Download Pre-trained Models:**
    Download the models from [this Google Drive folder](https://drive.google.com/drive/folders/1o9HRHYtgqB1BKdfHF6DVTZ6mpKSojhxF?usp=drive_link) and place them in the models/ directory:
    - yolov8m_plate_detector.pt
    - knn_plate_ocr.joblib
    - knn_validity_ocr.joblib

2.  **Start the Application:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the Web App:**
    Open your browser to http://localhost:7860. You can drag and drop images and see the full recognition pipeline results.

---

## Full Development Guide (From Scratch)

Follow these steps if you want to feel the full project: download datasets, prepare data, train models, and run evaluations.

### 1. Environment Setup
Requires Python 3.11.

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install all dependencies (including training/dev tools)
pip install -r requirements.txt
```

### 2. Dataset Setup (Kaggle)
We use the Indonesian Plate Number dataset from Kaggle.

1.  **Download:**
    ```bash
    kaggle datasets download -d linkgish/indonesian-plate-number-from-multi-sources -p datasets/kaggle_raw --unzip
    ```
2.  **Prepare:**
    This script organizes the raw data for YOLO and OCR training.
    ```bash
    python scripts/prepare_kaggle_dataset.py --raw-root datasets/kaggle_raw
    ```

### 3. Model Training

#### A. Train YOLO Plate Detector
You can choose different YOLOv8 model sizes based on your speed/accuracy needs:
- **Nano (n):** yolov8n.pt (Fastest, lightest)
- **Small (s):** yolov8s.pt
- **Medium (m):** yolov8m.pt (Recommended balance, used for yolov8m_plate_detector.pt)
- **Large (l):** yolov8l.pt
- **XLarge (x):** yolov8x.pt (Most accurate, slowest)

```bash
yolo detect train \
  model=yolov8m.pt \
  data=datasets/plate_detection/plate_dataset.yaml \
  epochs=50 \
  imgsz=640
# After training, rename your best weights to: models/yolov8m_plate_detector.pt
```

#### B. Train KNN OCR Models
Extract character crops and train the classification models:
```bash
# 1. Extract characters
python scripts/prepare_ocr_chars.py --yolo-model models/yolov8m_plate_detector.pt

# 2. Train models
python scripts/train_knn_models.py --training datasets/ocr_chars
```

### 4. Running & Evaluation

- **Run Single Image (CLI):**
  ```bash
  python main.py --image tests/fixtures/test8.jpg
  ```
- **Run Batch Inference:**
  ```bash
  python scripts/batch_inference.py --image-dir tests/fixtures --output-dir outputs/evaluation
  ```
- **Run Unit Tests:**
  ```bash
  pytest tests/ -v
  ```

---

## Project Structure

```text
.
├── datasets/           # Data preparation folders (ignored by git)
├── docs/               # Technical documentation & project logs
├── models/             # Trained weights (.pt and .joblib)
├── notebooks/          # Evaluation & visualization notebooks
├── outputs/            # Inference results and debug images
├── scripts/            # Utility scripts for data prep & training
├── src/                # Core system logic
│   ├── config.py       # Global parameters & paths
│   ├── pipeline.py     # Main processing sequence
│   ├── yolo_detector.py # YOLO model wrapper
│   ├── knn_ocr.py      # Character recognition logic
│   └── segmentation.py # Plate region & character extraction
├── tests/
│   ├── fixtures/       # Sample test images
│   └── evaluation/     # Ground truth & scoring tools
├── app.py              # Interactive web interface
├── main.py             # CLI entry point
├── Dockerfile          # Container configuration
└── docker-compose.yml  # Multi-container orchestration
```

## Limitations
- Recognition depends on detection quality.
- KNN OCR is a baseline; performance may vary with lighting, tilt, or non-standard fonts.
- Validity detection requires clear visibility of the lower month/year text.
