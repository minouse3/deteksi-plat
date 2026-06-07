# Hasil, Pembahasan, Output, dan Kesimpulan

## Ringkasan Pengujian

Pengujian dilakukan untuk melihat kemampuan sistem dalam mendeteksi pelat nomor, mengenali nomor utama pelat, membaca masa berlaku, dan menentukan status validitas pelat. Sistem yang diuji menggunakan kombinasi:

1. YOLO untuk deteksi area pelat.
2. Grayscale dan thresholding Otsu untuk pembentukan citra biner.
3. Noise removal dan segmentasi karakter.
4. K-Nearest Neighbour (KNN) untuk OCR karakter.
5. Parsing masa berlaku menjadi bulan dan tahun.
6. Validasi tanggal masa berlaku.

Pengujian utama dilakukan pada 500 gambar dari dataset Kaggle OCR/text:

```text
datasets/kaggle_raw/plate_text_dataset/plate_text_dataset/dataset
```

Output batch inference disimpan pada:

```text
outputs/evaluation/kaggle_ocr_500/results.csv
outputs/evaluation/kaggle_ocr_500/results.jsonl
```

Evaluasi paper-style disimpan pada:

```text
outputs/evaluation/kaggle_ocr_500/paper_style_summary.csv
outputs/evaluation/kaggle_ocr_500/paper_style_details.csv
outputs/evaluation/kaggle_ocr_500/results_and_discussion.md
notebooks/evaluation_results.ipynb
```

## Formula Akurasi

Perhitungan akurasi mengikuti bentuk perhitungan pada artikel rujukan `document-1.pdf`:

```text
accuracy rate (%) = right data total / data total * 100%
```

Pada pengujian ini, akurasi dihitung untuk beberapa tingkat:

1. Akurasi deteksi pelat.
2. Akurasi pengenalan karakter angka.
3. Akurasi pengenalan karakter huruf.
4. Akurasi pengenalan nomor pelat secara utuh.
5. Akurasi pembacaan masa berlaku.

Untuk pengenalan karakter, kesalahan dihitung menggunakan alignment berbasis edit distance. Jika karakter ground truth tidak sama dengan karakter prediksi pada posisi hasil alignment, maka karakter tersebut dihitung sebagai error. Jika karakter ground truth hilang atau terganti, karakter tersebut juga dihitung sebagai error.

## Hasil Evaluasi Batch 500 Gambar

Hasil evaluasi batch 500 gambar adalah sebagai berikut:

| Metrik | Hasil |
| --- | ---: |
| Total gambar diuji | 500 |
| Pelat terdeteksi | 366 |
| Detection rate | 73.20% |
| Rata-rata confidence deteksi | 0.7606 |
| Confidence minimum | 0.2591 |
| Confidence median | 0.8428 |
| Confidence maksimum | 0.9020 |
| Gambar dengan ground truth nomor pelat dari filename | 280 |
| Nomor pelat benar secara utuh | 32 |
| Akurasi nomor pelat utuh | 11.43% |
| Gambar dengan ground truth masa berlaku dari filename | 94 |
| Masa berlaku benar | 5 |
| Akurasi masa berlaku | 5.32% |

Distribusi hasil status masa berlaku pada 500 gambar:

| Status | Jumlah |
| --- | ---: |
| `unknown` | 406 |
| `expired` | 63 |
| `valid` | 31 |

Jumlah hasil masa berlaku yang berhasil diparse:

```text
94/500 = 18.80%
```

## Tabel Akurasi Bergaya Artikel

Tabel berikut mengikuti bentuk evaluasi pada artikel rujukan, yaitu membandingkan jumlah data, jumlah error, jumlah benar, dan accuracy rate.

| Characters | Data Total | Error | Number of True | Accuracy Rate |
| --- | ---: | ---: | ---: | ---: |
| Numbers | 1122 | 424 | 698 | 62.21% |
| Letters | 897 | 432 | 465 | 51.84% |
| Plate | 280 | 248 | 32 | 11.43% |

Berdasarkan tabel tersebut, sistem dapat mengenali karakter angka sebesar 62.21% dan karakter huruf sebesar 51.84% pada subset data yang memiliki ground truth dari filename. Untuk pengenalan nomor pelat secara utuh, sistem memperoleh 32 hasil benar dari 280 data, sehingga akurasi nomor pelat utuh adalah:

```text
accuracy rate = 32 / 280 * 100%
              = 11.43%
```

Nilai akurasi karakter lebih tinggi daripada akurasi pelat utuh karena satu kesalahan karakter saja sudah membuat satu nomor pelat dianggap salah secara exact match. Dengan demikian, walaupun sebagian karakter dapat dikenali, hasil akhir nomor pelat tetap dihitung salah apabila terdapat satu karakter yang hilang, tertukar, atau berlebih.

## Output Hasil Sistem

Output sistem berbentuk JSON. Format output utama terdiri dari:

| Field | Keterangan |
| --- | --- |
| `image_path` | Path gambar input |
| `plate_number` | Hasil akhir nomor pelat setelah post-processing |
| `raw_plate_number` | Hasil OCR mentah sebelum post-processing |
| `validity_text` | Hasil parsing masa berlaku dalam format `MM-YYYY` |
| `raw_validity_text` | Hasil OCR mentah bagian masa berlaku |
| `validity_month` | Bulan masa berlaku |
| `validity_year` | Tahun masa berlaku |
| `validity_status` | Status `valid`, `expired`, atau `unknown` |
| `plate_bbox` | Bounding box pelat hasil YOLO |
| `detection_confidence` | Confidence deteksi pelat |
| `debug` | Informasi tahap internal, seperti jumlah karakter tersegmentasi |

### Contoh Output Berhasil

Contoh pada `tests/fixtures/test8.jpg`, yaitu pelat putih dengan font hitam:

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
  "detection_confidence": 0.6709648966789246,
  "debug": {
    "plate_detected": true,
    "upper_region_found": true,
    "lower_region_found": true,
    "num_plate_chars": 8,
    "num_validity_chars": 4
  }
}
```

Pada contoh ini, sistem berhasil:

1. Mendeteksi area pelat.
2. Memisahkan nomor utama dan masa berlaku.
3. Melakukan segmentasi 8 karakter nomor utama.
4. Membaca nomor utama sebagai `B5838TOO`.
5. Membaca masa berlaku `0728`.
6. Mengubah masa berlaku menjadi `07-2028`.
7. Menentukan status sebagai `valid`.

### Contoh Output dengan Kesalahan OCR

Contoh pada `tests/fixtures/test10.jpg`, yaitu pelat putih dengan font hitam:

```json
{
  "image_path": "tests/fixtures/test10.jpg",
  "plate_number": "IJ6533AUB",
  "raw_plate_number": "1J6533AUB",
  "validity_text": null,
  "raw_validity_text": "110201",
  "validity_month": null,
  "validity_year": null,
  "validity_status": "unknown",
  "plate_bbox": [0, 0, 610, 237],
  "detection_confidence": 0.7577901482582092,
  "debug": {
    "plate_detected": true,
    "upper_region_found": true,
    "lower_region_found": true,
    "num_plate_chars": 9,
    "num_validity_chars": 6
  }
}
```

Berdasarkan inspeksi visual, pelat pada gambar tersebut terlihat sebagai:

```text
G 6533 AUB
10-29
```

Hasil yang diharapkan adalah:

```text
plate_number: G6533AUB
validity_text: 10-2029
validity_status: valid
```

Namun sistem menghasilkan `IJ6533AUB` dan masa berlaku `unknown`. Kesalahan ini terjadi karena:

1. Karakter `G` di sisi kiri tersegmentasi menjadi lebih dari satu komponen.
2. Hasil OCR mentah membaca bagian awal sebagai `1J`.
3. Region masa berlaku menangkap komponen tambahan berupa garis atau border.
4. Digit masa berlaku tersegmentasi menjadi 6 komponen, padahal seharusnya 4 digit.
5. Raw validity `110201` tidak sesuai pola bulan-tahun yang valid.

Contoh ini menunjukkan bahwa keberhasilan deteksi pelat belum selalu menjamin keberhasilan OCR. Tahap segmentasi karakter dan pembersihan komponen masih sangat menentukan hasil akhir.

## Pembahasan Deteksi Pelat

Deteksi pelat menggunakan YOLO menghasilkan 366 deteksi dari 500 gambar, atau 73.20%. Nilai ini menunjukkan bahwa model deteksi sudah dapat menemukan area pelat pada sebagian besar data uji. Confidence median sebesar 0.8428 menunjukkan bahwa banyak deteksi memiliki tingkat keyakinan relatif tinggi.

Namun masih terdapat 134 gambar yang tidak menghasilkan deteksi pelat. Beberapa faktor yang dapat memengaruhi kegagalan deteksi adalah:

1. Pelat terlalu kecil pada gambar.
2. Sudut pengambilan gambar terlalu miring.
3. Pencahayaan rendah atau glare terlalu kuat.
4. Pelat tertutup objek lain.
5. Bentuk pelat atau warna pelat berbeda dari mayoritas data training.
6. Dataset memiliki variasi sumber dan kualitas gambar.

Deteksi pelat merupakan tahap pertama yang sangat berpengaruh. Jika pelat tidak terdeteksi, maka proses crop, OCR, dan validasi masa berlaku tidak dapat dilakukan.

## Pembahasan Preprocessing Otsu

Metode Otsu digunakan untuk mengubah citra grayscale menjadi citra biner. Pada banyak kasus, Otsu membantu memperjelas karakter karena karakter dan background pelat memiliki perbedaan intensitas yang cukup jelas.

Tahapan preprocessing yang digunakan adalah:

1. Resize crop pelat.
2. Konversi grayscale.
3. Gaussian blur ringan.
4. Thresholding Otsu.
5. Inversi biner bila diperlukan.
6. Noise removal.

Pada pelat hitam dengan karakter putih dan pelat putih dengan karakter hitam, sistem perlu memastikan bahwa karakter tetap menjadi foreground yang konsisten. Oleh karena itu, terdapat mekanisme `maybe_invert_binary()` untuk membalik citra biner apabila foreground/background tidak sesuai.

Walaupun demikian, Otsu memiliki kelemahan ketika pencahayaan tidak merata. Jika terdapat glare, bayangan, baut, pelindung pelat, atau tekstur background yang kuat, histogram intensitas dapat berubah sehingga threshold yang dipilih tidak selalu memisahkan karakter dengan sempurna.

## Pembahasan Segmentasi Karakter

Segmentasi karakter dilakukan dengan connected components, kontur, dan analisis proyeksi. Sistem memisahkan pelat menjadi dua region:

1. Upper region untuk nomor utama.
2. Lower region untuk masa berlaku.

Pada beberapa gambar, segmentasi berjalan baik dan menghasilkan jumlah karakter sesuai ekspektasi. Contoh `test8.jpg` menghasilkan:

```text
num_plate_chars = 8
num_validity_chars = 4
```

Jumlah tersebut sesuai dengan nomor `B5838TOO` dan masa berlaku `0728`.

Pada contoh lain seperti `test10.jpg`, segmentasi menghasilkan:

```text
num_plate_chars = 9
num_validity_chars = 6
```

Jumlah tersebut tidak sesuai dengan struktur visual pelat. Hal ini menyebabkan OCR membaca karakter tambahan dan gagal melakukan parsing masa berlaku. Kesalahan segmentasi seperti ini dapat terjadi karena komponen non-karakter, border, atau bagian karakter yang terpisah dianggap sebagai karakter tersendiri.

## Pembahasan OCR KNN

OCR pada proyek ini menggunakan KNN sesuai baseline pada `document-1.pdf`. Setiap crop karakter diubah menjadi citra ukuran tetap, lalu diklasifikasikan berdasarkan kedekatan dengan data training.

Kelebihan KNN pada proyek ini:

1. Mudah diimplementasikan.
2. Sesuai dengan metode artikel rujukan.
3. Tidak membutuhkan proses training kompleks.
4. Hasilnya dapat dianalisis langsung dari sample karakter.

Keterbatasan yang ditemukan:

1. Sangat bergantung pada kualitas crop segmentasi.
2. Sensitif terhadap font yang berbeda.
3. Sensitif terhadap ketebalan karakter.
4. Sensitif terhadap noise dan potongan karakter.
5. Membutuhkan data sample karakter yang representatif.
6. Satu karakter yang salah dapat membuat nomor pelat utuh salah.

Hal ini terlihat pada akurasi huruf dan angka. Angka memperoleh akurasi 62.21%, sedangkan huruf memperoleh 51.84%. Huruf cenderung lebih sulit karena bentuk beberapa huruf mirip dengan angka atau huruf lain, seperti:

```text
B dan 8
O dan 0
I dan 1
G dan C
D dan O
S dan 5
Z dan 2
```

## Pembahasan Masa Berlaku

Fitur masa berlaku merupakan tambahan dari proyek ini. Sistem membaca teks pada bagian bawah pelat dan mencoba mengubah hasil OCR menjadi format `MM-YYYY`.

Contoh format yang didukung:

```text
0728 -> 07-2028
05 27 -> 05-2027
05.27 -> 05-2027
05-2027 -> 05-2027
12-24 -> 12-2024
```

Pada batch 500 gambar, masa berlaku berhasil diparse pada 94 gambar atau 18.80%. Dari 94 gambar yang memiliki ground truth masa berlaku dari filename, hanya 5 hasil yang benar secara exact match.

Rendahnya hasil ini dipengaruhi oleh beberapa faktor:

1. Tidak semua dataset menampilkan masa berlaku dengan jelas.
2. Region bawah sering lebih kecil daripada nomor utama.
3. Digit masa berlaku lebih mudah tertutup glare atau border.
4. Separator seperti titik, strip, atau dot dapat terbaca sebagai komponen.
5. Segmentasi bawah kadang menangkap garis tepi pelat.
6. Dataset publik tidak selalu memiliki label masa berlaku yang lengkap.

Walaupun hasil batch masih rendah, fitur ini sudah membentuk pipeline lengkap: deteksi region bawah, OCR digit-only, parsing bulan-tahun, dan validasi status.

## Pembahasan Warna Pelat

Dataset dan contoh pengujian menunjukkan adanya dua jenis tampilan pelat:

1. Pelat lama: background hitam dengan font putih.
2. Pelat baru: background putih dengan font hitam.

Pada proyek ini, warna pelat belum diklasifikasikan secara eksplisit sebagai label tersendiri. Penyesuaian warna dilakukan melalui preprocessing citra biner. Fungsi inversi biner mencoba menjaga agar karakter menjadi foreground yang konsisten sebelum segmentasi dan OCR.

Contoh `test8.jpg` menunjukkan bahwa pelat putih dengan font hitam dapat diproses dengan hasil benar setelah segmentasi karakter menghasilkan crop yang tepat dan OCR memiliki sample yang sesuai. Contoh `test10.jpg` menunjukkan bahwa pelat putih tetap dapat menimbulkan kesalahan apabila karakter sisi kiri atau masa berlaku tersegmentasi tidak tepat.

## Debug Output

Sistem menyimpan debug image untuk membantu analisis. File debug utama meliputi:

```text
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

Debug output berguna untuk mengetahui tahap mana yang menyebabkan kesalahan. Contohnya:

1. Jika `02_detected_plate_bbox.jpg` salah, maka masalah ada pada YOLO detection.
2. Jika `05_binary_plate.jpg` buruk, maka masalah ada pada thresholding atau pencahayaan.
3. Jika `09_upper_segmented_chars/` berisi komponen yang bukan karakter, maka masalah ada pada segmentasi.
4. Jika segmentasi sudah benar tetapi prediksi salah, maka masalah ada pada KNN OCR atau data training.
5. Jika raw validity terbaca tetapi tidak dapat diparse, maka masalah ada pada format hasil OCR masa berlaku.

## Perbandingan dengan Artikel Rujukan

Artikel pada `document-1.pdf` menggunakan input berupa citra pelat yang sudah tersedia dan menerapkan Otsu serta KNN untuk mengenali karakter. Proyek ini mempertahankan bagian utama tersebut, yaitu:

1. Grayscale.
2. Thresholding Otsu.
3. Noise removal.
4. Segmentasi karakter.
5. KNN classification.

Perbedaan utama proyek ini adalah:

1. Input berupa citra kendaraan penuh.
2. Penambahan YOLO untuk deteksi otomatis pelat.
3. Penambahan pembacaan masa berlaku.
4. Penambahan parsing bulan dan tahun.
5. Penambahan validasi status masa berlaku.
6. Penambahan batch inference.
7. Penambahan notebook evaluasi dan chart.

Dengan demikian, proyek ini memperluas baseline artikel dari pengenalan nomor pelat pada citra crop menjadi pipeline yang lebih lengkap untuk citra kendaraan dan informasi masa berlaku pelat.

## Kendala yang Ditemukan

Selama pengujian, beberapa kendala utama yang ditemukan adalah:

1. Karakter dapat terpotong karena crop pelat kurang tepat.
2. Border pelat dapat ikut terbaca sebagai karakter.
3. Baut dan pelindung pelat dapat mengubah bentuk karakter.
4. Karakter pada pelat putih dan pelat hitam memiliki kontras yang berbeda.
5. Font pelat tidak selalu seragam.
6. Karakter yang mirip dapat tertukar.
7. Masa berlaku sering berukuran kecil dan lebih sulit disegmentasi.
8. Separator pada masa berlaku dapat ikut menjadi komponen.
9. Pencahayaan dan glare dapat mengganggu Otsu thresholding.
10. Posisi pelat miring dapat menyebabkan segmentasi karakter tidak stabil.

Kendala-kendala tersebut sejalan dengan pembahasan pada artikel rujukan, terutama mengenai pengaruh posisi pelat, font, noise, baut, pencahayaan, dan variasi fisik pelat terhadap hasil segmentasi dan pengenalan.

## Kesimpulan

Berdasarkan implementasi dan pengujian yang dilakukan, dapat disimpulkan bahwa:

1. Proyek berhasil mengimplementasikan pipeline pengenalan pelat nomor kendaraan Indonesia berbasis YOLO, Otsu thresholding, segmentasi citra, dan KNN OCR.
2. Metode Otsu dan KNN dari `document-1.pdf` berhasil diadaptasi sebagai baseline OCR pada proyek ini.
3. YOLO berhasil menambahkan kemampuan deteksi pelat otomatis dari citra kendaraan penuh.
4. Sistem berhasil menambahkan fitur pembacaan masa berlaku pelat yang tidak terdapat pada baseline artikel.
5. Sistem dapat menghasilkan output JSON berisi nomor pelat, masa berlaku, status validitas, bounding box pelat, confidence deteksi, dan informasi debug.
6. Pada evaluasi 500 gambar, deteksi pelat mencapai 73.20%.
7. Pada subset yang memiliki ground truth nomor pelat, akurasi nomor pelat utuh adalah 11.43%.
8. Akurasi karakter angka adalah 62.21%, sedangkan akurasi karakter huruf adalah 51.84%.
9. Masa berlaku berhasil diparse pada 18.80% dari 500 gambar, dengan akurasi exact match 5.32% pada subset yang memiliki ground truth masa berlaku.
10. Kesalahan terbesar berasal dari segmentasi karakter, variasi font, noise fisik pelat, glare, border, dan keterbatasan data training OCR.

Secara umum, proyek ini sudah membentuk sistem end-to-end yang dapat menerima citra kendaraan, mendeteksi pelat, membaca nomor utama, membaca masa berlaku, dan menghasilkan status validitas. Hasil evaluasi juga menunjukkan bahwa setiap tahap dapat dianalisis secara terpisah melalui debug image, tabel akurasi, dan notebook evaluasi.

## Saran Pengembangan

Beberapa pengembangan yang dapat dilakukan berikutnya adalah:

1. Menambah data training OCR untuk variasi pelat hitam dan pelat putih.
2. Menambahkan klasifikasi tipe pelat berdasarkan warna background dan warna karakter.
3. Memperbaiki segmentasi masa berlaku agar border dan separator tidak ikut terbaca sebagai digit.
4. Menambahkan rectification perspektif untuk pelat miring.
5. Menambahkan preprocessing untuk glare dan pencahayaan tidak merata.
6. Membandingkan KNN dengan model OCR lain sebagai eksperimen tambahan.
7. Membuat ground truth masa berlaku yang lebih lengkap.
8. Menambahkan evaluasi terpisah untuk pelat hitam dan pelat putih.
9. Menambahkan confidence per karakter OCR.
10. Menambahkan laporan evaluasi otomatis untuk setiap folder dataset.
