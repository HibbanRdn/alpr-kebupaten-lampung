"""
01_extract_frames.py
====================
Ekstraksi frame dari video mentah untuk dataset ALPR Lampung.

Fitur:
- Ambil frame setiap N frame
- Skip frame yang terlalu mirip dengan frame sebelumnya (similarity check via histogram)
- Skip frame yang terlalu buram (Laplacian blur detection)
- Progress bar di terminal
- Simpan session_log.csv secara otomatis

Penggunaan:
    python 01_extract_frames.py --video VIDEO.mp4 --output data/raw_frames/sesi_01_siang
    python 01_extract_frames.py --video VIDEO.mp4 --output data/raw_frames/sesi_01_siang --interval 10 --blur-threshold 80 --similarity-threshold 0.95

Argumen:
    --video             Path ke file video (.mp4, .mov, dll)
    --output            Folder output untuk menyimpan frame
    --interval          Ambil 1 frame setiap N frame (default: 10)
    --blur-threshold    Nilai minimum Laplacian variance agar frame tidak dianggap buram (default: 80.0)
    --similarity-threshold  Nilai maksimum histogram similarity agar frame tidak dianggap duplikat (default: 0.95)
    --quality           Kualitas JPEG output, 1-100 (default: 92)

Dependensi:
    pip install opencv-python tqdm
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm



# Fungsi utilitas


def compute_blur_score(frame: np.ndarray) -> float:
    """
    Hitung skor ketajaman frame menggunakan variance of Laplacian.
    Skor lebih tinggi = lebih tajam.
    Skor di bawah blur_threshold = dianggap buram.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_histogram(frame: np.ndarray) -> np.ndarray:
    """
    Hitung histogram HSV dari frame untuk similarity check.
    Menggunakan HSV agar lebih robust terhadap perubahan pencahayaan ringan.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compute_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Bandingkan dua histogram menggunakan metode Bhattacharyya (diinversi).
    Return: nilai 0.0 (berbeda total) hingga 1.0 (identik).
    """
    bhatt_dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return 1.0 - bhatt_dist


def format_duration(seconds: float) -> str:
    """Format durasi dalam detik ke string MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"



# Fungsi utama ekstraksi


def extract_frames(
    video_path: str,
    output_dir: str,
    interval: int = 10,
    blur_threshold: float = 80.0,
    similarity_threshold: float = 0.95,
    quality: int = 92,
) -> dict:
    """
    Ekstrak frame dari video dan simpan ke output_dir.

    Returns:
        dict berisi statistik proses ekstraksi.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Validasi input
    if not video_path.exists():
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Buka video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s  = total_frames / fps if fps > 0 else 0

    print(f"\n{'─'*52}")
    print(f"  Video     : {video_path.name}")
    print(f"  Resolusi  : {width}x{height}")
    print(f"  FPS       : {fps:.1f}")
    print(f"  Durasi    : {format_duration(duration_s)} ({total_frames:,} frame)")
    print(f"  Interval  : setiap {interval} frame")
    print(f"  Output    : {output_dir}")
    print(f"{'─'*52}\n")

    # Estimasi jumlah frame kandidat (sebelum filter)
    candidates = total_frames // interval

    # State
    stats = {
        "video_name"       : video_path.name,
        "sesi_id"          : output_dir.name,
        "tanggal"          : datetime.now().strftime("%Y-%m-%d"),
        "waktu_mulai"      : datetime.now().strftime("%H:%M"),
        "fps"              : round(fps, 2),
        "durasi_detik"     : round(duration_s, 1),
        "total_frame_video": total_frames,
        "frame_kandidat"   : candidates,
        "disimpan"         : 0,
        "dibuang_buram"    : 0,
        "dibuang_mirip"    : 0,
    }

    frame_idx       = 0
    saved_count     = 0
    skipped_blur    = 0
    skipped_similar = 0
    prev_hist       = None

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    with tqdm(
        total=candidates,
        desc="Mengekstrak",
        unit="frame",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour="cyan",
    ) as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:

                # ── Filter 1: Blur ───────────────────────────
                blur_score = compute_blur_score(frame)
                if blur_score < blur_threshold:
                    skipped_blur += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "saved": saved_count,
                        "blur" : skipped_blur,
                        "sim"  : skipped_similar,
                    }, refresh=False)
                    frame_idx += 1
                    continue

                # ── Filter 2: Similarity ─────────────────────
                curr_hist = compute_histogram(frame)
                if prev_hist is not None:
                    sim = compute_similarity(prev_hist, curr_hist)
                    if sim >= similarity_threshold:
                        skipped_similar += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            "saved": saved_count,
                            "blur" : skipped_blur,
                            "sim"  : skipped_similar,
                        }, refresh=False)
                        frame_idx += 1
                        continue

                # ── Simpan frame ─────────────────────────────
                filename = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(filename), frame, encode_params)
                saved_count += 1
                prev_hist    = curr_hist

                pbar.update(1)
                pbar.set_postfix({
                    "saved": saved_count,
                    "blur" : skipped_blur,
                    "sim"  : skipped_similar,
                }, refresh=False)

            frame_idx += 1

    cap.release()

    stats["waktu_selesai"]   = datetime.now().strftime("%H:%M")
    stats["disimpan"]        = saved_count
    stats["dibuang_buram"]   = skipped_blur
    stats["dibuang_mirip"]   = skipped_similar

    # ── Ringkasan ────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  Selesai!")
    print(f"  Frame kandidat   : {candidates:>6,}")
    print(f"  Dibuang (buram)  : {skipped_blur:>6,}  (blur score < {blur_threshold})")
    print(f"  Dibuang (mirip)  : {skipped_similar:>6,}  (similarity >= {similarity_threshold})")
    print(f"  Disimpan         : {saved_count:>6,}  → {output_dir}")
    print(f"{'─'*52}\n")

    return stats



# Update session_log.csv


def update_session_log(stats: dict, log_path: str = "data/raw_frames/session_log.csv") -> None:
    """
    Tambahkan baris baru ke session_log.csv.
    Buat file + header jika belum ada.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sesi_id", "tanggal", "waktu_mulai", "waktu_selesai",
        "video_name", "fps", "durasi_detik", "total_frame_video",
        "frame_kandidat", "disimpan", "dibuang_buram", "dibuang_mirip",
    ]

    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: stats[k] for k in fieldnames})

    print(f"  Session log diperbarui → {log_path}\n")



# Entry point


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ekstraksi frame dari video untuk dataset ALPR Lampung.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path ke file video input (.mp4, .mov, dll)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Folder output untuk menyimpan frame (misal: data/raw_frames/sesi_01_siang)"
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Ambil 1 frame setiap N frame"
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=80.0,
        help="Laplacian variance minimum agar frame tidak dianggap buram"
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.95,
        help="Histogram similarity maksimum agar frame tidak dianggap duplikat (0.0–1.0)"
    )
    parser.add_argument(
        "--quality", type=int, default=92,
        help="Kualitas JPEG output (1–100)"
    )
    parser.add_argument(
        "--log", default="data/raw_frames/session_log.csv",
        help="Path ke file session_log.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start = time.time()
    stats = extract_frames(
        video_path           = args.video,
        output_dir           = args.output,
        interval             = args.interval,
        blur_threshold       = args.blur_threshold,
        similarity_threshold = args.similarity_threshold,
        quality              = args.quality,
    )
    elapsed = time.time() - start
    print(f"  Total waktu proses: {elapsed:.1f} detik")

    update_session_log(stats, log_path=args.log)