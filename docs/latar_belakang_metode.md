# Latar Belakang, Permasalahan, Landasan Teori, dan Metode

## Identitas Proyek

Nama proyek: `license_plate_validity_recognition`

Proyek ini merupakan sistem pengenalan pelat nomor kendaraan Indonesia berbasis computer vision. Sistem menerima citra kendaraan secara utuh, mendeteksi area pelat nomor secara otomatis, mengenali nomor utama pelat, membaca masa berlaku yang tercetak pada bagian bawah pelat, lalu menentukan status masa berlaku sebagai `valid`, `expired`, atau `unknown`.

Proyek ini dikembangkan berdasarkan artikel pada `document-1.pdf`, yaitu **"Recognition Number of The Vehicle Plate Using Otsu Method and K-Nearest Neighbour Classification"** oleh Maulidia R. Hidayah, Isa Akhlis, dan Endang Sugiharti. Artikel tersebut menggunakan metode Otsu untuk mengubah citra menjadi citra biner dan K-Nearest Neighbour (KNN) untuk klasifikasi karakter pelat nomor.

Kontribusi tambahan pada proyek ini adalah:

1. Input berupa citra kendaraan penuh, bukan hanya citra pelat yang sudah dipotong manual.
2. Penambahan deteksi otomatis area pelat menggunakan YOLO melalui Ultralytics, dengan rancangan yang mendukung model YOLOv11 atau model YOLO lain yang telah dilatih pada data pelat.
3. Penambahan deteksi teks masa berlaku pelat pada bagian bawah nomor utama.
4. Parsing masa berlaku menjadi bulan dan tahun.
5. Penentuan status masa berlaku pelat.
6. Penyimpanan debug image pada setiap tahap utama.
7. Evaluasi batch dan notebook analisis hasil.

## Latar Belakang

Jumlah kendaraan di Indonesia terus bertambah, sehingga kebutuhan sistem identifikasi kendaraan otomatis juga meningkat. License Plate Recognition (LPR) dapat digunakan pada sistem parkir, gerbang tol, pengawasan lalu lintas, keamanan area, dan manajemen kendaraan. Pada sistem tersebut, pelat nomor berfungsi sebagai identitas kendaraan yang dapat diproses secara otomatis dari citra kamera.

Artikel pada `document-1.pdf` menunjukkan bahwa pengolahan citra dan klasifikasi sederhana dapat digunakan untuk mengenali karakter pelat nomor. Metode yang digunakan dalam artikel adalah konversi grayscale, thresholding Otsu, segmentasi karakter, dan klasifikasi KNN. Pendekatan ini cukup relevan sebagai baseline karena sederhana, mudah dijelaskan, dan dapat dievaluasi secara terpisah pada tiap tahap.

Namun, pendekatan pada artikel memiliki batasan penting: citra input diasumsikan sudah berupa pelat yang dipotong. Pada kondisi nyata, input biasanya berupa foto kendaraan penuh, sehingga sistem perlu menemukan lokasi pelat terlebih dahulu. Selain itu, pelat kendaraan Indonesia juga memiliki informasi masa berlaku pada bagian bawah pelat. Informasi ini tidak hanya berguna untuk mengenali identitas kendaraan, tetapi juga untuk memeriksa apakah masa berlaku pelat masih aktif.

## Permasalahan

Permasalahan utama yang diselesaikan proyek ini adalah bagaimana mengenali nomor pelat kendaraan Indonesia dan masa berlakunya dari citra kendaraan secara otomatis.

Permasalahan tersebut dapat diuraikan menjadi beberapa submasalah:

1. Bagaimana mendeteksi lokasi pelat dari citra kendaraan penuh.
2. Bagaimana memotong area pelat secara otomatis dari hasil deteksi.
3. Bagaimana mengubah citra pelat menjadi citra biner agar karakter lebih mudah disegmentasi.
4. Bagaimana memisahkan area nomor utama dan area masa berlaku.
5. Bagaimana melakukan segmentasi karakter pada nomor utama.
6. Bagaimana melakukan segmentasi digit pada masa berlaku.
7. Bagaimana mengenali karakter menggunakan KNN.
8. Bagaimana mengubah hasil OCR masa berlaku menjadi format bulan dan tahun.
9. Bagaimana menentukan status masa berlaku berdasarkan tanggal saat ini.

Tantangan citra yang sering muncul adalah variasi warna pelat, kemiringan pelat, pencahayaan tidak merata, glare, baut, pelindung pelat, font yang berbeda, karakter yang aus, motion blur, dan batas pelat yang ikut tersegmentasi sebagai objek.

## Landasan Teori

### License Plate Recognition

License Plate Recognition (LPR) adalah proses mengenali nomor pelat kendaraan dari citra. Secara umum, LPR terdiri dari beberapa tahap: akuisisi citra, deteksi pelat, preprocessing, segmentasi karakter, pengenalan karakter, dan post-processing. Pada proyek ini, tahap deteksi pelat dilakukan menggunakan YOLO, sedangkan OCR karakter menggunakan pendekatan baseline Otsu dan KNN.

### Grayscale

Citra RGB diubah menjadi grayscale untuk menyederhanakan informasi warna menjadi intensitas piksel. Tahap ini mengurangi kompleksitas pemrosesan karena metode thresholding Otsu bekerja pada distribusi intensitas citra.

### Thresholding Otsu

Metode Otsu digunakan untuk menentukan nilai ambang secara otomatis berdasarkan histogram intensitas. Tujuannya adalah memisahkan piksel foreground dan background. Pada proyek ini, Otsu digunakan untuk membentuk citra biner dari area pelat dan crop karakter.

Dalam konteks pelat nomor, foreground biasanya adalah karakter, sedangkan background adalah area dasar pelat. Karena pelat lama dan pelat baru dapat memiliki kombinasi warna yang berbeda, hasil biner dapat diinversi agar karakter menjadi foreground yang konsisten.

### Segmentasi Citra

Segmentasi citra adalah proses memisahkan objek penting dari citra. Pada proyek ini, segmentasi dilakukan pada dua level:

1. Segmentasi area teks: memisahkan region nomor utama dan region masa berlaku.
2. Segmentasi karakter: memisahkan karakter atau digit satu per satu.

Segmentasi dilakukan dengan memanfaatkan citra biner, connected components, kontur, dan proyeksi horizontal/vertikal.

### K-Nearest Neighbour

K-Nearest Neighbour (KNN) adalah metode klasifikasi berbasis kedekatan data. Pada proyek ini, setiap crop karakter diubah menjadi ukuran tetap, diratakan menjadi vektor fitur, lalu diklasifikasikan berdasarkan jarak terhadap data training karakter.

KNN digunakan sebagai baseline karena sesuai dengan metode pada `document-1.pdf`. Untuk nomor pelat utama, kelas yang dikenali adalah huruf `A-Z` dan angka `0-9`. Untuk masa berlaku, OCR dapat menggunakan mode digit-only karena bagian masa berlaku hanya membutuhkan angka.

### YOLO untuk Deteksi Pelat

YOLO digunakan untuk mendeteksi bounding box pelat pada citra kendaraan penuh. Penambahan YOLO merupakan perluasan dari metode pada artikel, karena artikel menggunakan citra pelat yang sudah dipotong. Dengan YOLO, sistem dapat menerima citra kendaraan penuh dan menemukan area pelat secara otomatis.

Implementasi proyek menggunakan library Ultralytics. Rancangan deteksi mendukung penggunaan model YOLOv11 atau model YOLO lain yang sudah dilatih untuk objek pelat kendaraan. Model hasil training disimpan sebagai:

```text
models/yolov8m_plate_detector.pt
```

### Validasi Masa Berlaku

Masa berlaku pelat dibaca dari area bawah pelat. Hasil OCR digit diparsing ke format:

```text
MM-YYYY
```

Contoh:

```text
0728 -> 07-2028
10.29 -> 10-2029
05-2027 -> 05-2027
```

Status masa berlaku ditentukan dengan aturan: pelat dianggap masih valid sampai akhir bulan pada tahun yang terbaca.

## Metode yang Digunakan

Alur metode pada proyek ini adalah sebagai berikut:

1. Input citra kendaraan.
2. Deteksi pelat menggunakan YOLO.
3. Crop area pelat berdasarkan bounding box.
4. Resize crop pelat agar ukuran pemrosesan lebih stabil.
5. Konversi ke grayscale.
6. Thresholding Otsu.
7. Inversi biner bila diperlukan.
8. Noise removal menggunakan connected components.
9. Pemisahan region atas dan bawah.
10. Segmentasi karakter nomor utama.
11. Segmentasi digit masa berlaku.
12. OCR nomor utama menggunakan KNN.
13. OCR masa berlaku menggunakan KNN digit-only.
14. Parsing teks masa berlaku menjadi bulan dan tahun.
15. Validasi status masa berlaku.
16. Penyimpanan hasil JSON dan debug image.

Output sistem berbentuk data JSON, misalnya:

```json
{
  "image_path": "tests/fixtures/test10.jpg",
  "plate_number": "G6533AUB",
  "validity_text": "10-2029",
  "validity_month": "10",
  "validity_year": "2029",
  "validity_status": "valid",
  "plate_bbox": [0, 0, 610, 237],
  "detection_confidence": 0.75
}
```

## Keterkaitan dengan Operasi Pengolahan Citra

Berikut adalah bagian proyek yang berkaitan dengan operasi pengolahan citra.

### 1. Perbaikan Kualitas Citra (Image Enhancement)

Bagian ini ada pada proyek, tetapi masih bersifat sederhana. Contohnya:

1. Resize pelat agar ukuran pemrosesan lebih konsisten.
2. Konversi grayscale.
3. Gaussian blur ringan sebelum thresholding.
4. Inversi biner agar foreground karakter lebih konsisten.
5. Deskew/alignment sederhana untuk mengurangi kemiringan ringan.

File terkait:

```text
src/preprocessing.py
```

### 2. Pemugaran Citra (Image Restoration)

Bagian ini belum menjadi fokus utama proyek. Proyek belum melakukan pemugaran citra secara khusus, misalnya deblurring, glare removal, super-resolution, atau rekonstruksi piksel yang rusak.

Namun, terdapat proses sederhana yang mendekati pemugaran ringan, yaitu noise removal berbasis connected components untuk menghapus komponen kecil yang tidak dianggap karakter.

### 3. Pemampatan Citra (Image Compression)

Bagian ini tidak menjadi bagian metode utama. Proyek tidak melakukan kompresi citra sebagai proses analisis. File debug disimpan sebagai `.jpg`, tetapi itu hanya format penyimpanan output, bukan metode pemampatan yang dianalisis.

### 4. Segmentasi Citra (Image Segmentation)

Bagian ini merupakan komponen penting dalam proyek. Segmentasi digunakan untuk:

1. Memisahkan area pelat dari citra kendaraan menggunakan YOLO.
2. Memisahkan region nomor utama dan region masa berlaku.
3. Memisahkan karakter nomor utama.
4. Memisahkan digit masa berlaku.

File terkait:

```text
src/yolo_detector.py
src/segmentation.py
```

### 5. Pengorakan Citra (Image Analysis)

Bagian ini ada pada proyek dalam bentuk analisis struktur citra biner, antara lain:

1. Analisis connected components.
2. Analisis kontur.
3. Analisis proyeksi horizontal untuk menemukan band teks.
4. Analisis ukuran bounding box karakter.
5. Analisis jumlah karakter dan digit yang ditemukan.

Hasil analisis ini digunakan untuk menentukan apakah upper region dan lower region ditemukan, serta berapa jumlah karakter yang berhasil disegmentasi.

### 6. Rekonstruksi Citra (Image Reconstruction)

Bagian ini tidak digunakan sebagai metode utama. Proyek tidak membangun ulang citra dari proyeksi atau data lain. Output debug hanya menyimpan hasil antara, seperti crop pelat, citra grayscale, citra biner, dan crop karakter.

## Dataset

Dataset yang digunakan berasal dari Kaggle:

```text
Indonesian Plate Number from Multi Sources
https://www.kaggle.com/datasets/linkgish/indonesian-plate-number-from-multi-sources
```

Dataset digunakan untuk dua kebutuhan:

1. Deteksi pelat menggunakan subset detection.
2. OCR nomor pelat menggunakan subset OCR/text.

Karena dataset publik belum tentu menyediakan label masa berlaku bagian bawah pelat, proyek juga mendukung label masa berlaku custom dalam format CSV.

## Tools dan Library

Environment pengembangan yang terbaca pada mesin saat dokumen ini dibuat:

| Komponen | Versi/Keterangan |
| --- | --- |
| Sistem operasi | Linux 7.0.10-1-cachyos x86_64 |
| Python | 3.11.14 |
| OpenCV | 4.13.0 |
| NumPy | 2.4.6 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| matplotlib | 3.10.9 |
| pytest | 9.0.3 |
| tqdm | 4.67.3 |
| PyYAML | 6.0.3 |
| Ultralytics | 8.4.60 |
| PyTorch | 2.12.0+cu130 |
| CUDA pada PyTorch saat dicek | Tidak aktif pada sesi pengecekan |

Dependensi utama proyek terdapat pada:

```text
requirements.txt
```

## Hardware

Hardware yang terbaca pada mesin saat dokumen ini dibuat:

| Komponen | Spesifikasi |
| --- | --- |
| CPU | Intel Core i3-10100F @ 3.60 GHz |
| Core/Thread | 4 core / 8 thread |
| RAM | 15 GiB |
| Swap | 15 GiB |
| GPU yang digunakan pada eksperimen training sebelumnya | NVIDIA GeForce RTX 3070 8 GB |

Catatan: pada sesi pengecekan dokumen ini, `nvidia-smi` tidak dapat berkomunikasi dengan driver NVIDIA dan `torch.cuda.is_available()` bernilai `False`. Pada eksperimen training sebelumnya, log Ultralytics sempat mendeteksi CUDA device `NVIDIA GeForce RTX 3070, 7834 MiB`. Karena itu, penggunaan GPU bergantung pada kondisi driver NVIDIA, CUDA, dan instalasi PyTorch di environment yang sedang aktif.

## Struktur Modul Utama

| Modul | Fungsi |
| --- | --- |
| `src/yolo_detector.py` | Deteksi dan crop pelat menggunakan YOLO |
| `src/preprocessing.py` | Grayscale, Otsu thresholding, noise removal, alignment sederhana |
| `src/segmentation.py` | Split upper/lower region dan segmentasi karakter |
| `src/knn_ocr.py` | Training dan prediksi OCR menggunakan KNN |
| `src/validity_detection.py` | Normalisasi dan parsing teks masa berlaku |
| `src/date_validation.py` | Penentuan status valid/expired/unknown |
| `src/pipeline.py` | Integrasi seluruh proses dari input citra sampai output JSON |
| `scripts/batch_inference.py` | Inference banyak gambar |
| `scripts/paper_style_evaluation.py` | Evaluasi seperti tabel akurasi pada artikel |

## Evaluasi

Proyek menyediakan evaluasi batch dan notebook analisis. Evaluasi dapat menghitung:

1. Detection rate.
2. Akurasi nomor pelat secara exact match.
3. Akurasi karakter angka.
4. Akurasi karakter huruf.
5. Akurasi masa berlaku.
6. Distribusi status masa berlaku.
7. Histogram confidence deteksi YOLO.

File evaluasi yang sudah dibuat:

```text
notebooks/evaluation_results.ipynb
outputs/evaluation/kaggle_ocr_500/paper_style_summary.csv
outputs/evaluation/kaggle_ocr_500/paper_style_details.csv
outputs/evaluation/kaggle_ocr_500/results_and_discussion.md
```

## Kesimpulan

Project ini mengimplementasikan metode dasar dari `document-1.pdf`, yaitu Otsu thresholding dan KNN classification untuk pengenalan karakter pelat nomor. Pengembangan tambahan dilakukan dengan menambahkan deteksi pelat otomatis menggunakan YOLO/YOLOv11 melalui Ultralytics serta deteksi masa berlaku pelat pada bagian bawah nomor utama.

Dengan kombinasi tersebut, sistem tidak hanya mengenali nomor pelat, tetapi juga menghasilkan informasi masa berlaku dan status validitas pelat. Proyek ini juga menyediakan debug image dan evaluasi berbasis tabel/grafik agar setiap tahap pemrosesan dapat dianalisis secara terpisah.
