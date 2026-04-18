# 🚗 ALPR Lampung
### Automatic License Plate Recognition untuk Identifikasi Wilayah Kendaraan Provinsi Lampung

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Baidu-0052CC?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Self--collected-blueviolet?style=flat-square)

---

Sistem ALPR dua-pipeline berbasis deep learning yang mendeteksi plat nomor kendaraan dan mengidentifikasi asal wilayah kabupaten/kota di Provinsi Lampung berdasarkan kode huruf belakang plat `BE`. Dibangun sebagai project akademik mata kuliah **Deep Learning** — Teknik Informatika, Universitas Lampung (2026).

---

## Daftar Isi

- [Gambaran Sistem](#gambaran-sistem)
- [Perbandingan Dua Pipeline](#perbandingan-dua-pipeline)
- [Dataset](#dataset)
- [Struktur Folder](#struktur-folder)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Evaluasi & Metrik](#evaluasi--metrik)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Keterbatasan](#keterbatasan)
- [Rencana Pengembangan](#rencana-pengembangan)
- [Referensi](#referensi)
- [Author](#author)

---

## Gambaran Sistem

Sistem ini menyelesaikan satu tujuan utama:

> **Diberikan sebuah foto atau frame video kendaraan bermotor, identifikasi secara otomatis kabupaten/kota asal kendaraan tersebut berdasarkan kode plat nomor Lampung (BE).**

Contoh output:

```
Input  : [foto kendaraan dengan plat BE 1234 AR]
Output : Plat    → BE 1234 AR
         Wilayah → Kota Bandar Lampung
         Pipeline → A (YOLOv8 + PaddleOCR)
         Confidence → high
```

Pipeline berakhir pada **identifikasi wilayah**, bukan sekadar pembacaan teks — sehingga sistem ini dapat menjadi komponen pada aplikasi manajemen parkir, portal kawasan, atau survei komposisi kendaraan.

---

## Perbandingan Dua Pipeline

Proyek ini mengimplementasikan dan membandingkan dua pendekatan arsitektur untuk tugas yang sama:

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│         PIPELINE A              │   │         PIPELINE B              │
│   Detection-based (2 tahap)     │   │   OCR End-to-end (1 tahap)      │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│                                 │   │                                 │
│  Gambar → YOLOv8                │   │  Gambar                         │
│           ↓                     │   │     ↓                           │
│         Crop plat               │   │  PaddleOCR (full image)         │
│           ↓                     │   │     ↓                           │
│         PaddleOCR               │   │  Ekstrak kode wilayah           │
│           ↓                     │   │     ↓                           │
│  Ekstrak kode wilayah           │   │  Identifikasi wilayah           │
│           ↓                     │   │                                 │
│  Identifikasi wilayah           │   │  ✓ Lebih cepat                  │
│                                 │   │  ✗ Lebih rentan noise           │
│  ✓ Lebih akurat                 │   │                                 │
│  ✗ Lebih lambat                 │   │                                 │
└─────────────────────────────────┘   └─────────────────────────────────┘
                       ↓                             ↓
               ┌───────────────────────────────────────┐
               │     region_mapper.py                  │
               │  Kode huruf → Nama kabupaten/kota     │
               └───────────────────────────────────────┘
```

Perbandingan dilakukan pada metrik: **Region Classification Accuracy**, **CER**, **Character Accuracy**, dan **Inference Time**.

---

## Dataset

Dataset dikumpulkan secara mandiri — tidak menggunakan dataset publik.

| Atribut | Detail |
|---|---|
| Lokasi pengambilan | Jl. AH Nasution 3, Taman Kota Metro, Kota Metro, Lampung |
| Perangkat kamera | iPhone 6s Plus (1080p @ 30fps) |
| Kondisi pencahayaan | Siang (10.00–13.00 WIB) dan sore (14.00–17.00 WIB) |
| Jenis kendaraan | Sepeda motor dan mobil |
| Format plat | Plat hitam (karakter putih) dan plat putih (karakter hitam) |
| Target jumlah gambar | 300–450 gambar teranotasi |
| Format anotasi | YOLO (bounding box `license_plate`) |
| Tools anotasi | Roboflow |
| Pembagian dataset | 70% train / 20% valid / 10% test (source-based split) |

> **Catatan privasi:** Dataset dikumpulkan di ruang publik untuk keperluan akademik. Visualisasi plat pada publikasi disamarkan sebagian sesuai pertimbangan privasi.

### Kode Wilayah Plat Lampung (BE)

Identifikasi wilayah didasarkan pada **huruf pertama** dari pasangan huruf belakang plat:

| Kode | Wilayah | Kode | Wilayah |
|---|---|---|---|
| A, B, C, Y | Kota Bandar Lampung | N, P | Kab. Lampung Timur |
| F | Kota Metro | J, K | Kab. Lampung Utara |
| D, E, O | Kab. Lampung Selatan | V, Z | Kab. Tanggamus |
| M | Kab. Lampung Barat | W | Kab. Way Kanan |
| G, H, I | Kab. Lampung Tengah | S, T | Kab. Tulang Bawang |
| R | Kab. Pesawaran | Q | Kab. Tulang Bawang Barat |
| U | Kab. Pringsewu | L | Kab. Mesuji |
| X | Kab. Pesisir Barat | | |

---

## Struktur Folder

```
alpr-lampung/
├── data/
│   ├── raw_frames/                  # Frame pilihan dari video, belum dianotasi
│   │   ├── sesi_01_siang/
│   │   ├── sesi_02_siang/
│   │   ├── sesi_03_sore/
│   │   └── session_log.csv          # Metadata setiap sesi pengambilan data
│   └── dataset_yolo/                # Export dari Roboflow (format YOLOv8)
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml
├── models/
│   ├── yolov8n/weights/             # best.pt dan last.pt YOLOv8 Nano
│   └── yolov8s/weights/             # best.pt dan last.pt YOLOv8 Small
├── src/
│   ├── 01_extract_frames.py         # Ekstraksi frame dari video
│   ├── 02_pipeline_a_yolo_ocr.py    # Pipeline A: YOLOv8 + PaddleOCR
│   ├── 03_pipeline_b_paddleocr.py   # Pipeline B: PaddleOCR end-to-end
│   ├── region_mapper.py             # Lookup kode → wilayah Lampung
│   └── evaluate.py                  # Hitung semua metrik evaluasi
├── notebooks/
│   ├── 01_training_yolov8.ipynb     # Training YOLOv8 di Google Colab
│   ├── 02_eval_detection.ipynb      # Evaluasi deteksi plat
│   └── 03_comparison_pipeline.ipynb # Perbandingan Pipeline A vs B
├── results/
│   ├── pipeline_a/
│   │   ├── detection_metrics.json   # mAP, precision, recall
│   │   └── ocr_results.csv          # Teks terbaca + wilayah + ground truth
│   └── pipeline_b/
│       └── ocr_results.csv
└── README.md
```

> `models/` dan `data/` tidak di-push ke GitHub. Lihat bagian [Instalasi](#instalasi) untuk cara mendapatkannya.

---

## Instalasi

### Prasyarat

- Python 3.10 atau lebih baru
- pip
- (Opsional) GPU dengan CUDA untuk training lebih cepat

### Clone repository

```bash
git clone https://github.com/<username>/alpr-lampung.git
cd alpr-lampung
```

### Install dependensi

```bash
pip install ultralytics paddlepaddle paddleocr opencv-python tqdm
```

Untuk training di Google Colab (disarankan), seluruh dependensi sudah tersedia atau dapat diinstall langsung di notebook.

---

## Cara Penggunaan

### 1. Ekstraksi frame dari video

```bash
python src/01_extract_frames.py \
  --video rekaman_sesi_01.mp4 \
  --output data/raw_frames/sesi_01_siang \
  --interval 10 \
  --blur-threshold 80 \
  --similarity-threshold 0.95
```

| Argumen | Default | Keterangan |
|---|---|---|
| `--interval` | `10` | Ambil 1 frame setiap N frame |
| `--blur-threshold` | `80.0` | Laplacian variance minimum (frame lebih tajam = nilai lebih tinggi) |
| `--similarity-threshold` | `0.95` | Histogram similarity maksimum (1.0 = identik, akan dibuang) |

### 2. Anotasi dataset

Upload gambar dari `data/raw_frames/` ke [Roboflow](https://roboflow.com), anotasi bounding box plat dengan label `license_plate`, lalu export ke `data/dataset_yolo/` dalam format **YOLOv8**.

### 3. Training YOLOv8

Buka dan jalankan `notebooks/01_training_yolov8.ipynb` di Google Colab (GPU T4).

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data/dataset_yolo/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    save_period=10,
)
```

### 4. Jalankan Pipeline A (YOLOv8 + PaddleOCR)

```bash
python src/02_pipeline_a_yolo_ocr.py \
  --input data/dataset_yolo/test/images \
  --model models/yolov8n/weights/best.pt \
  --output results/pipeline_a/ocr_results.csv
```

### 5. Jalankan Pipeline B (PaddleOCR end-to-end)

```bash
python src/03_pipeline_b_paddleocr.py \
  --input data/dataset_yolo/test/images \
  --output results/pipeline_b/ocr_results.csv
```

### 6. Evaluasi dan perbandingan

```bash
python src/evaluate.py \
  --pipeline-a results/pipeline_a/ocr_results.csv \
  --pipeline-b results/pipeline_b/ocr_results.csv
```

### Uji `region_mapper` secara mandiri

```bash
python src/region_mapper.py
```

Atau sebagai modul:

```python
from src.region_mapper import identify_region

result = identify_region("BE 1234 AR")
print(result["region_name"])    # → Kota Bandar Lampung
print(result["confidence"])     # → high
```

---

## Evaluasi & Metrik

Evaluasi dilakukan pada tiga level:

### Level 1 — Deteksi plat (Pipeline A saja)

| Metrik | Keterangan |
|---|---|
| Precision | Proporsi deteksi yang benar terhadap total prediksi |
| Recall | Proporsi plat yang berhasil terdeteksi terhadap total plat |
| mAP@0.5 | Mean Average Precision pada IoU threshold 0.5 |
| mAP@0.5:0.95 | Mean Average Precision pada rentang IoU 0.5–0.95 |

### Level 2 — Pembacaan karakter OCR (kedua pipeline)

| Metrik | Keterangan |
|---|---|
| Character Accuracy | % karakter terbaca benar dibanding ground truth |
| CER | Character Error Rate — edit distance / panjang ground truth |

### Level 3 — Identifikasi wilayah, metrik utama (kedua pipeline)

| Metrik | Keterangan |
|---|---|
| **Region Classification Accuracy** | % kendaraan yang wilayahnya berhasil diidentifikasi benar |
| Inference Time (ms/gambar) | Kecepatan rata-rata per gambar |

---

## Hasil Eksperimen

> 🔄 **Bagian ini akan diperbarui setelah eksperimen selesai.**

### Deteksi Plat — Pipeline A

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n | — | — | — | — |
| YOLOv8s | — | — | — | — |

### Perbandingan Pipeline A vs B

| Metrik | Pipeline A | Pipeline B |
|---|---|---|
| Region Classification Accuracy | — | — |
| Character Accuracy | — | — |
| CER | — | — |
| Inference Time (ms/gambar) | — | — |

---

## Keterbatasan

- Dataset terbatas pada satu lokasi (Jl. AH Nasution 3, Metro) dan dua kondisi cahaya (siang dan sore)
- Tidak mencakup kondisi malam hari, hujan, atau blur ekstrem
- Kode wilayah yang jarang muncul (misal `X` untuk Pesisir Barat, `L` untuk Mesuji) kemungkinan kurang terwakili dalam dataset
- Identifikasi wilayah menggunakan rule-based mapping, bukan model klasifikasi terlatih
- Tidak ada integrasi basis data kendaraan atau pelacakan multi-objek

---

## Rencana Pengembangan

- [ ] Perluasan dataset ke kondisi malam hari dan hujan ringan
- [ ] Fine-tuning PaddleOCR pada karakter plat Indonesia
- [ ] Inferensi real-time pada video streaming (OpenCV + ONNX/TensorRT)
- [ ] Koreksi perspektif (perspective correction) sebelum OCR
- [ ] Integrasi multi-object tracking (ByteTrack / DeepSORT)
- [ ] Antarmuka web sederhana sebagai proof-of-concept sistem parkir

---

## Referensi

```
[1] Jocher et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
[2] Du et al. (2020). PP-OCR: A Practical Ultra Lightweight OCR System. arXiv:2009.09941
[3] Terven & Cordova-Esparza (2023). A Comprehensive Review of YOLO Architectures.
    Machine Learning and Knowledge Extraction, 5(4), 1680–1716.
[4] Laroca et al. (2021). An Efficient and Layout-Independent ALPR System.
    IET Intelligent Transport Systems, 15(4), 483–503.
[5] Bjorklund et al. (2019). Robust License Plate Recognition Using Neural Networks.
    Pattern Recognition, 93, 134–146.
```

---

## Author

**M. Hibban Ramadhan**
Program Studi Teknik Informatika — Universitas Lampung
Mata Kuliah Deep Learning, 2026
Dosen: Rio Ariesta Pradipta, S.Kom., M.TI.

---

<div align="center">
  <sub>Dibuat sebagai project akademik · Dataset hanya untuk keperluan penelitian</sub>
</div>
