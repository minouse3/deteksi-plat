# Urutan Penelitian dan Pengerjaan Proyek

## Gambaran Umum

Dokumen ini menjelaskan urutan penelitian dan pengerjaan proyek `license_plate_validity_recognition` secara lengkap, mulai dari persiapan dataset sampai evaluasi hasil. Proyek ini bertujuan membangun sistem pengenalan pelat nomor kendaraan Indonesia yang dapat menerima citra kendaraan penuh, mendeteksi area pelat, mengenali nomor utama pelat, membaca masa berlaku, dan menentukan status validitas pelat.

Metode dasar OCR pada proyek ini mengacu pada `document-1.pdf`, yaitu penggunaan Otsu thresholding dan K-Nearest Neighbour (KNN). Pengembangan tambahan pada proyek ini adalah penggunaan YOLO untuk deteksi pelat otomatis serta penambahan pembacaan masa berlaku pelat.

## 1. Studi Literatur

Tahap pertama adalah mempelajari metode yang digunakan pada artikel rujukan:

```text
Recognition Number of The Vehicle Plate Using Otsu Method and K-Nearest Neighbour Classification
```

Artikel tersebut menjelaskan pipeline pengenalan pelat nomor dengan tahapan:

1. Input citra pelat.
2. Konversi citra ke grayscale.
3. Thresholding menggunakan metode Otsu.
4. Pembersihan noise.
5. Segmentasi karakter.
6. Klasifikasi karakter menggunakan KNN.
7. Evaluasi akurasi karakter dan akurasi pelat.

Dari studi literatur tersebut, proyek ini menggunakan Otsu dan KNN sebagai baseline OCR. Namun, proyek dikembangkan lebih lanjut agar dapat menerima gambar kendaraan penuh dan membaca informasi masa berlaku pelat.

## 2. Perumusan Masalah

Setelah memahami baseline dari artikel, permasalahan proyek dirumuskan sebagai berikut:

1. Sistem harus dapat menerima citra kendaraan penuh.
2. Sistem harus dapat mendeteksi area pelat secara otomatis.
3. Sistem harus dapat melakukan crop pelat berdasarkan hasil deteksi.
4. Sistem harus dapat mengenali nomor utama pelat.
5. Sistem harus dapat mendeteksi dan membaca masa berlaku pelat.
6. Sistem harus dapat menentukan status masa berlaku.
7. Sistem harus menyediakan debug output untuk analisis setiap tahap.
8. Sistem harus menyediakan evaluasi hasil secara batch.

## 3. Penentuan Dataset

Dataset utama yang digunakan berasal dari Kaggle:

```text
Indonesian Plate Number from Multi Sources
https://www.kaggle.com/datasets/linkgish/indonesian-plate-number-from-multi-sources
```

Dataset tersebut digunakan karena berisi data pelat nomor kendaraan Indonesia dari berbagai sumber. Dataset dimanfaatkan untuk dua kebutuhan utama:

1. Dataset deteksi pelat untuk melatih YOLO.
2. Dataset OCR/text untuk pelat nomor dan karakter.

Selain dataset Kaggle, proyek juga mendukung dataset custom untuk label masa berlaku karena dataset publik belum tentu menyediakan anotasi masa berlaku yang lengkap.

## 4. Persiapan Environment

Proyek menggunakan Python 3.11 dan virtual environment, tanpa Conda. Environment dibuat agar dependency proyek terisolasi.

Langkah umum pada Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Dependency utama:

```text
ultralytics
opencv-python
numpy
scikit-image
scikit-learn
pandas
matplotlib
pytest
tqdm
pyyaml
```

File dependency:

```text
requirements.txt
```

## 5. Download Dataset

Dataset diunduh menggunakan Kaggle API atau secara manual dari halaman Kaggle.

Contoh perintah Kaggle CLI:

```bash
kaggle datasets download \
  -d linkgish/indonesian-plate-number-from-multi-sources \
  -p datasets/kaggle_raw \
  --unzip
```

Folder tujuan:

```text
datasets/kaggle_raw/
```

Dataset Kaggle dapat memiliki struktur folder yang berbeda setelah diunduh. Oleh karena itu, proyek tidak mengasumsikan satu struktur tetap.

## 6. Inspeksi Dataset

Setelah dataset tersedia, struktur dataset diperiksa dengan script:

```bash
python scripts/inspect_dataset.py --root datasets/kaggle_raw
```

Script ini digunakan untuk:

1. Menampilkan ringkasan folder.
2. Mencari folder yang berisi gambar.
3. Mencari file anotasi.
4. Mencari file COCO JSON.
5. Mencari file CSV atau TXT label.

Tujuan tahap ini adalah mengetahui bagian dataset mana yang dapat digunakan untuk deteksi pelat dan bagian mana yang dapat digunakan untuk OCR.

## 7. Persiapan Dataset Deteksi YOLO

Dataset deteksi disiapkan ke format YOLO:

```text
datasets/plate_detection/
  images/
    train/
    val/
  labels/
    train/
    val/
  plate_dataset.yaml
```

Jika dataset memiliki anotasi COCO, script konversi digunakan:

```bash
python scripts/convert_coco_to_yolo.py \
  --coco-json path/to/annotations.json \
  --image-dir path/to/images \
  --output-dir datasets/plate_detection
```

Pada proyek ini, anotasi COCO dapat berisi box kecil atau box per karakter. Karena yang dibutuhkan YOLO adalah deteksi satu area pelat utuh, script mendukung penggabungan box per gambar menjadi satu bounding box pelat.

Script persiapan otomatis:

```bash
python scripts/prepare_kaggle_dataset.py --raw-root datasets/kaggle_raw
```

Output penting:

```text
datasets/plate_detection/plate_dataset.yaml
```

File YAML ini digunakan oleh Ultralytics YOLO saat training.

## 8. Training YOLO Plate Detector

YOLO digunakan untuk mendeteksi lokasi pelat pada citra kendaraan penuh.

Contoh training:

```bash
yolo detect train \
  model=yolov8m.pt \
  data=datasets/plate_detection/plate_dataset.yaml \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  device=0
```

Catatan:

1. `yolov8m.pt` digunakan sebagai starting model.
2. Rancangan proyek mendukung penggunaan YOLOv11 dari Ultralytics apabila model tersebut digunakan saat training.
3. Model umum YOLO perlu dilatih ulang pada data pelat agar dapat mendeteksi pelat Indonesia dengan baik.
4. GPU seperti RTX 3070 8 GB dapat digunakan untuk mempercepat training.

Setelah training, model terbaik disalin ke:

```text
models/yolov8m_plate_detector.pt
```

Model inilah yang digunakan pada proses inference.

## 9. Persiapan Dataset OCR Karakter

OCR KNN membutuhkan dataset karakter yang sudah dipisahkan per kelas. Struktur dataset OCR:

```text
datasets/ocr_chars/
  A/
  B/
  C/
  ...
  Z/
  0/
  1/
  ...
  9/
```

Setiap folder berisi crop karakter untuk label tersebut. Contoh:

```text
datasets/ocr_chars/B/...
datasets/ocr_chars/5/...
datasets/ocr_chars/O/...
```

Dataset OCR dapat diperoleh dari:

1. Subset OCR/text pada Kaggle.
2. Crop hasil segmentasi debug.
3. Sample manual yang sudah divalidasi labelnya.

Persiapan sample OCR dapat dibantu dengan script:

```bash
python scripts/prepare_ocr_chars.py
```

Untuk menambahkan crop debug ke dataset OCR:

```bash
python scripts/add_debug_crops_to_ocr.py \
  --crops-dir outputs/debug/example/09_upper_segmented_chars \
  --labels B1234ABC \
  --output-dir datasets/ocr_chars \
  --prefix manual
```

Tahap ini harus dilakukan hati-hati. Crop yang bukan karakter, seperti blob, border, titik, baut, atau noise, tidak boleh dimasukkan sebagai sample training.

## 10. Persiapan Label Masa Berlaku

Dataset Kaggle belum tentu menyediakan label masa berlaku pelat. Oleh karena itu, proyek mendukung CSV custom dengan format:

```csv
image,plate_number,validity_month,validity_year,validity_text
img001.jpg,B3974FXL,12,2024,12-2024
img002.jpg,B1234ABC,05,2027,05-2027
```

Validasi label masa berlaku dilakukan dengan:

```bash
python scripts/prepare_validity_labels.py \
  --csv path/to/custom_validity.csv \
  --image-root datasets/validity_custom/images \
  --output datasets/validity_custom/labels.csv
```

Script ini memeriksa:

1. Kelengkapan kolom.
2. Keberadaan file gambar.
3. Validitas bulan.
4. Validitas tahun.
5. Output CSV yang sudah dibersihkan.

## 11. Implementasi Modul Deteksi Pelat

Deteksi pelat diimplementasikan pada:

```text
src/yolo_detector.py
```

Fungsi utama:

1. Memuat model YOLO.
2. Mendeteksi pelat dari citra penuh.
3. Memilih bounding box dengan confidence tertinggi.
4. Melakukan crop pelat dengan padding.
5. Menggambar bounding box untuk debug.

Output deteksi:

```json
{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.91
}
```

Jika tidak ada pelat terdeteksi, pipeline mengembalikan error:

```json
{
  "plate_number": null,
  "validity_status": "unknown",
  "error": "No plate detected"
}
```

## 12. Implementasi Preprocessing

Preprocessing diimplementasikan pada:

```text
src/preprocessing.py
```

Tahap preprocessing:

1. Load image.
2. Resize crop pelat.
3. Konversi grayscale.
4. Thresholding Otsu.
5. Inversi biner jika diperlukan.
6. Noise removal menggunakan connected components.
7. Deskew/alignment sederhana.

Preprocessing ini bertujuan menghasilkan citra biner yang lebih mudah disegmentasi.

## 13. Implementasi Segmentasi

Segmentasi diimplementasikan pada:

```text
src/segmentation.py
```

Segmentasi terdiri dari:

1. Pemisahan upper region dan lower region.
2. Segmentasi karakter nomor utama.
3. Segmentasi digit masa berlaku.

Upper region berisi nomor utama pelat, misalnya:

```text
B5838TOO
```

Lower region berisi masa berlaku, misalnya:

```text
07-28
```

Metode yang digunakan:

1. Horizontal projection.
2. Connected components.
3. Contour detection.
4. Filtering berdasarkan ukuran komponen.
5. Sorting karakter dari kiri ke kanan.

## 14. Implementasi OCR KNN

OCR diimplementasikan pada:

```text
src/knn_ocr.py
```

Alur OCR:

1. Load dataset karakter dari `datasets/ocr_chars`.
2. Konversi karakter ke grayscale.
3. Thresholding Otsu.
4. Resize ke ukuran tetap.
5. Flatten menjadi vektor fitur.
6. Training KNN.
7. Prediksi karakter.
8. Gabungkan prediksi menjadi string.

Mode OCR:

1. OCR normal untuk nomor utama: huruf dan angka.
2. OCR digit-only untuk masa berlaku: angka saja.

Parameter utama:

```text
KNN_K = 3
CHAR_IMAGE_SIZE = (20, 40)
```

Pada eksperimen dengan sample OCR kecil dan manual, `--knn-k 1` sering digunakan agar sample koreksi langsung berpengaruh pada prediksi.

## 15. Implementasi Parsing Masa Berlaku

Parsing masa berlaku diimplementasikan pada:

```text
src/validity_detection.py
```

Input OCR masa berlaku dapat memiliki berbagai bentuk:

```text
0527
05 27
05.27
05-27
05/27
05 2027
05-2027
12-24
```

Output parsing:

```json
{
  "month": "05",
  "year": "2027",
  "normalized": "05-2027"
}
```

Jika teks tidak dapat diparse:

```json
{
  "month": null,
  "year": null,
  "normalized": null
}
```

## 16. Implementasi Validasi Tanggal

Validasi status masa berlaku diimplementasikan pada:

```text
src/date_validation.py
```

Aturan:

1. Masa berlaku aktif sampai akhir bulan yang terbaca.
2. Jika tanggal saat ini masih berada sebelum atau dalam bulan tersebut, status `valid`.
3. Jika tanggal saat ini melewati bulan tersebut, status `expired`.
4. Jika bulan atau tahun tidak valid, status `unknown`.

Contoh:

```text
07-2028 -> valid
03-2025 -> expired
invalid -> unknown
```

## 17. Integrasi Pipeline

Pipeline utama diimplementasikan pada:

```text
src/pipeline.py
```

Urutan pipeline:

1. Load citra input.
2. Simpan debug original image.
3. Deteksi pelat dengan YOLO.
4. Simpan debug bounding box.
5. Crop pelat.
6. Resize dan alignment crop.
7. Grayscale.
8. Otsu thresholding.
9. Noise removal.
10. Split upper/lower region.
11. Segmentasi karakter upper.
12. Segmentasi digit lower.
13. OCR nomor utama.
14. OCR masa berlaku.
15. Parsing masa berlaku.
16. Cek status masa berlaku.
17. Return output JSON.

Debug output:

```text
outputs/debug/
  01_original.jpg
  02_detected_plate_bbox.jpg
  03_cropped_plate.jpg
  04_grayscale_plate.jpg
  05_binary_plate.jpg
  06_cleaned_binary_plate.jpg
  07_upper_region.jpg
  08_lower_validity_region.jpg
  09_upper_segmented_chars/
  10_lower_segmented_digits/
```

## 18. Implementasi CLI

CLI utama terdapat pada:

```text
main.py
```

Contoh penggunaan:

```bash
MPLCONFIGDIR=/tmp/matplotlib python main.py \
  --image tests/fixtures/test8.jpg \
  --yolo-model models/yolov8m_plate_detector.pt \
  --training datasets/ocr_chars \
  --debug-dir outputs/debug/test8 \
  --digit-only-validity \
  --knn-k 1
```

CLI melakukan:

1. Load YOLO detector.
2. Training/load KNN OCR dari folder karakter.
3. Menjalankan pipeline.
4. Mencetak output JSON.
5. Menyimpan debug image.

## 19. Pengujian Gambar Tunggal

Pengujian awal dilakukan pada gambar `tests/fixtures/test1.jpg` sampai `tests/fixtures/test10.jpg`. Tujuannya adalah mengecek:

1. Apakah YOLO menemukan pelat.
2. Apakah crop pelat tepat.
3. Apakah upper/lower region terpisah.
4. Apakah jumlah karakter hasil segmentasi sesuai.
5. Apakah OCR membaca nomor utama dengan benar.
6. Apakah masa berlaku dapat diparse.

Contoh hasil benar pada `tests/fixtures/test8.jpg`:

```json
{
  "image_path": "tests/fixtures/test8.jpg",
  "plate_number": "B5838TOO",
  "raw_plate_number": "B5838TOO",
  "validity_text": "07-2028",
  "raw_validity_text": "0728",
  "validity_month": "07",
  "validity_year": "2028",
  "validity_status": "valid",
  "plate_bbox": [82, 0, 1280, 440],
  "detection_confidence": 0.6709648966789246
}
```

Contoh hasil dengan kesalahan OCR pada `tests/fixtures/test10.jpg`:

```json
{
  "image_path": "tests/fixtures/test10.jpg",
  "plate_number": "IJ6533AUB",
  "raw_plate_number": "1J6533AUB",
  "validity_text": null,
  "raw_validity_text": "110201",
  "validity_status": "unknown",
  "detection_confidence": 0.7577901482582092
}
```

Pengujian gambar tunggal digunakan untuk debugging manual sebelum evaluasi batch.

## 20. Batch Inference

Batch inference dilakukan dengan script:

```text
scripts/batch_inference.py
```

Contoh perintah:

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/batch_inference.py \
  --image-dir datasets/kaggle_raw/plate_text_dataset/plate_text_dataset/dataset \
  --yolo-model models/yolov8m_plate_detector.pt \
  --training datasets/ocr_chars \
  --output-dir outputs/evaluation/kaggle_ocr_500 \
  --limit 500 \
  --seed 123 \
  --digit-only-validity \
  --knn-k 1
```

Output:

```text
outputs/evaluation/kaggle_ocr_500/results.csv
outputs/evaluation/kaggle_ocr_500/results.jsonl
outputs/evaluation/kaggle_ocr_500/debug/
```

Batch inference digunakan untuk melihat performa sistem pada banyak gambar sekaligus.

## 21. Evaluasi Hasil

Evaluasi dilakukan dengan script:

```text
scripts/paper_style_evaluation.py
```

Contoh perintah:

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/paper_style_evaluation.py \
  --predictions outputs/evaluation/kaggle_ocr_500/results.csv \
  --output-dir outputs/evaluation/kaggle_ocr_500 \
  --notebook notebooks/evaluation_results.ipynb \
  --title "Indonesian License Plate Recognition Evaluation"
```

Output evaluasi:

```text
outputs/evaluation/kaggle_ocr_500/paper_style_summary.csv
outputs/evaluation/kaggle_ocr_500/paper_style_details.csv
outputs/evaluation/kaggle_ocr_500/results_and_discussion.md
notebooks/evaluation_results.ipynb
```

Evaluasi menghitung:

1. Total gambar.
2. Jumlah pelat terdeteksi.
3. Detection rate.
4. Akurasi karakter angka.
5. Akurasi karakter huruf.
6. Akurasi nomor pelat utuh.
7. Akurasi masa berlaku.
8. Distribusi status validitas.

## 22. Hasil Evaluasi Utama

Hasil batch 500 gambar:

| Metrik | Hasil |
| --- | ---: |
| Total gambar | 500 |
| Pelat terdeteksi | 366 |
| Detection rate | 73.20% |
| Ground truth nomor pelat tersedia | 280 |
| Nomor pelat benar | 32 |
| Akurasi nomor pelat utuh | 11.43% |
| Ground truth masa berlaku tersedia | 94 |
| Masa berlaku benar | 5 |
| Akurasi masa berlaku | 5.32% |

Tabel akurasi karakter:

| Characters | Data Total | Error | Number of True | Accuracy Rate |
| --- | ---: | ---: | ---: | ---: |
| Numbers | 1122 | 424 | 698 | 62.21% |
| Letters | 897 | 432 | 465 | 51.84% |
| Plate | 280 | 248 | 32 | 11.43% |

## 23. Pembuatan Notebook Evaluasi

Notebook evaluasi dibuat agar hasil dapat dianalisis dengan tabel dan grafik.

Notebook:

```text
notebooks/evaluation_results.ipynb
```

Isi notebook:

1. Formula akurasi.
2. Tabel hasil evaluasi.
3. Grafik accuracy by category.
4. Grafik distribusi status masa berlaku.
5. Histogram confidence deteksi YOLO.
6. Contoh mismatch.
7. Catatan pembahasan.

Chart yang dihasilkan:

```text
outputs/evaluation/kaggle_ocr_500/accuracy_by_category.png
outputs/evaluation/kaggle_ocr_500/validity_status_distribution.png
outputs/evaluation/kaggle_ocr_500/detection_confidence_histogram.png
```

## 24. Unit Testing

Unit test dibuat untuk memastikan fungsi penting berjalan sesuai ekspektasi.

Folder test:

```text
tests/
```

Test yang tersedia:

1. `test_date_validation.py`
2. `test_validity_parsing.py`
3. `test_crop_plate.py`
4. `test_plate_number.py`
5. `test_evaluation.py`

Perintah menjalankan test:

```bash
.venv/bin/pytest
```

Hasil test terakhir:

```text
22 passed
```

Unit test memeriksa:

1. Parsing masa berlaku.
2. Validasi tanggal.
3. Crop bounding box pelat.
4. Normalisasi format nomor pelat.
5. Perhitungan evaluasi.

## 25. Penyusunan Dokumentasi

Dokumentasi proyek dibuat dalam beberapa file:

```text
README.md
docs/latar_belakang_metode.md
docs/hasil_pembahasan_kesimpulan.md
docs/urutan_penelitian_proyek.md
```

Fungsi masing-masing dokumen:

| Dokumen | Fungsi |
| --- | --- |
| `README.md` | Panduan setup, training, inference, dan evaluasi |
| `docs/latar_belakang_metode.md` | Latar belakang, permasalahan, teori, dan metode |
| `docs/hasil_pembahasan_kesimpulan.md` | Hasil evaluasi, pembahasan, output, dan kesimpulan |
| `docs/urutan_penelitian_proyek.md` | Urutan penelitian dan pengerjaan proyek |

## 26. Alur Penelitian Secara Ringkas

Urutan penelitian secara ringkas:

1. Studi literatur dari `document-1.pdf`.
2. Menentukan pengembangan dari baseline artikel.
3. Menyiapkan environment Python 3.11.
4. Mengunduh dataset Kaggle.
5. Menginspeksi struktur dataset.
6. Mengonversi dataset deteksi ke format YOLO.
7. Melatih YOLO plate detector.
8. Menyimpan model YOLO ke `models/yolov8m_plate_detector.pt`.
9. Menyiapkan dataset OCR karakter.
10. Menyiapkan label masa berlaku bila tersedia.
11. Mengimplementasikan deteksi pelat.
12. Mengimplementasikan preprocessing Otsu.
13. Mengimplementasikan segmentasi upper/lower.
14. Mengimplementasikan OCR KNN.
15. Mengimplementasikan parsing masa berlaku.
16. Mengimplementasikan validasi status masa berlaku.
17. Mengintegrasikan seluruh proses dalam pipeline.
18. Membuat CLI inference.
19. Menguji gambar tunggal.
20. Memperbaiki segmentasi dan OCR berdasarkan debug output.
21. Menjalankan batch inference.
22. Menghitung evaluasi.
23. Membuat notebook dan chart.
24. Menjalankan unit test.
25. Menyusun dokumentasi hasil.

## 27. Diagram Alur Sistem

Diagram alur sistem secara tekstual:

```text
Dataset Kaggle
    |
    v
Inspeksi Dataset
    |
    v
Persiapan Dataset Deteksi YOLO ---------
    |                                   |
    v                                   |
Training YOLO                           |
    |                                   |
    v                                   |
models/yolov8m_plate_detector.pt                |
                                        |
Dataset OCR Karakter                    |
    |                                   |
    v                                   |
Training KNN OCR                        |
    |                                   |
    v                                   v
Input Citra Kendaraan ------------> YOLO Plate Detection
                                        |
                                        v
                                  Crop Area Pelat
                                        |
                                        v
                                  Preprocessing Otsu
                                        |
                                        v
                              Split Upper dan Lower Region
                                  |                 |
                                  v                 v
                          Segmentasi Nomor     Segmentasi Masa Berlaku
                                  |                 |
                                  v                 v
                              KNN OCR          KNN Digit OCR
                                  |                 |
                                  v                 v
                          Nomor Pelat       Parsing Bulan-Tahun
                                  |                 |
                                  v                 v
                                Output JSON + Debug Image
                                        |
                                        v
                                  Evaluasi Batch
                                        |
                                        v
                               Notebook dan Laporan
```

## 28. Catatan Akhir

Urutan pengerjaan proyek ini dimulai dari dataset, dilanjutkan dengan training deteksi, persiapan OCR, implementasi pipeline, pengujian gambar tunggal, evaluasi batch, dan dokumentasi. Pendekatan ini membuat setiap tahap dapat diuji secara terpisah. Jika terjadi kesalahan, debug output dapat digunakan untuk mengetahui apakah masalah berasal dari deteksi pelat, preprocessing, segmentasi, OCR, parsing masa berlaku, atau evaluasi.
