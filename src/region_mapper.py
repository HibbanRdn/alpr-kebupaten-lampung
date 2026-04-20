"""
region_mapper.py
================
Lookup table kode huruf belakang plat nomor → nama wilayah Provinsi Lampung.

Semua kendaraan di Lampung menggunakan awalan 'BE', dengan pembeda wilayah
pada huruf PERTAMA dari pasangan huruf belakang setelah angka.

Contoh pembacaan:
    BE 1234 AR  → huruf pertama belakang = 'A' → Kota Bandar Lampung
    BE 5678 FG  → huruf pertama belakang = 'F' → Kota Metro
    BE 9012 GP  → huruf pertama belakang = 'G' → Kab. Lampung Tengah

Penggunaan sebagai modul:
    from region_mapper import identify_region, parse_plate

    result = identify_region("BE 1234 AR")
    print(result)
    # {
    #     "raw_text"      : "BE 1234 AR",
    #     "is_lampung"    : True,
    #     "region_code"   : "A",
    #     "region_name"   : "Kota Bandar Lampung",
    #     "confidence"    : "high",
    #     "notes"         : ""
    # }

Penggunaan langsung (demo):
    python region_mapper.py
"""

import re
from typing import Optional



# Lookup table: huruf pertama belakang → nama wilayah
# Sumber: Polda Lampung / SAMSAT Provinsi Lampung
# 15 kabupaten/kota sesuai pembagian administratif Lampung


REGION_MAP: dict[str, str] = {
    # Kota Bandar Lampung
    "A": "Kota Bandar Lampung",
    "B": "Kota Bandar Lampung",
    "C": "Kota Bandar Lampung",
    "Y": "Kota Bandar Lampung",

    # Kota Metro
    "F": "Kota Metro",

    # Kabupaten Lampung Selatan
    "D": "Kab. Lampung Selatan",
    "E": "Kab. Lampung Selatan",
    "O": "Kab. Lampung Selatan",

    # Kabupaten Lampung Barat
    "M": "Kab. Lampung Barat",

    # Kabupaten Lampung Tengah
    "G": "Kab. Lampung Tengah",
    "H": "Kab. Lampung Tengah",
    "I": "Kab. Lampung Tengah",

    # Kabupaten Lampung Timur
    "N": "Kab. Lampung Timur",
    "P": "Kab. Lampung Timur",

    # Kabupaten Lampung Utara
    "J": "Kab. Lampung Utara",
    "K": "Kab. Lampung Utara",

    # Kabupaten Tanggamus
    "V": "Kab. Tanggamus",
    "Z": "Kab. Tanggamus",

    # Kabupaten Way Kanan
    "W": "Kab. Way Kanan",

    # Kabupaten Tulang Bawang
    "S": "Kab. Tulang Bawang",
    "T": "Kab. Tulang Bawang",

    # Kabupaten Tulang Bawang Barat
    "Q": "Kab. Tulang Bawang Barat",

    # Kabupaten Pesawaran
    "R": "Kab. Pesawaran",

    # Kabupaten Pringsewu
    "U": "Kab. Pringsewu",

    # Kabupaten Mesuji
    "L": "Kab. Mesuji",

    # Kabupaten Pesisir Barat
    "X": "Kab. Pesisir Barat",
}

# Kode huruf yang sering salah baca oleh OCR (ambiguitas visual)
# Digunakan untuk memberi catatan pada hasil identifikasi
OCR_AMBIGUOUS_PAIRS: list[tuple[str, str]] = [
    ("I", "1"),   # huruf I vs angka 1
    ("O", "0"),   # huruf O vs angka 0
    ("I", "J"),   # I dan J mirip secara visual
    ("G", "C"),   # G dan C mirip pada font plat
    ("S", "5"),   # S vs angka 5
    ("Z", "2"),   # Z vs angka 2
    ("B", "8"),   # B vs angka 8
]



# Fungsi parsing plat nomor


def normalize_plate(raw_text: str) -> str:
    """
    Normalisasi teks hasil OCR sebelum diparse.
    - Uppercase semua karakter
    - Hapus karakter selain huruf, angka, dan spasi
    - Collapse spasi ganda
    """
    text = raw_text.upper()
    text = re.sub(r"[^A-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_plate(raw_text: str) -> Optional[dict]:
    """
    Parse teks plat nomor Indonesia format Lampung (BE XXXX XX).

    Format yang diterima (fleksibel terhadap spasi):
        "BE 1234 AR"  → standar
        "BE1234AR"    → tanpa spasi (kadang output OCR)
        "BE 1234AR"   → spasi tidak konsisten

    Returns:
        dict dengan key: prefix, number, suffix
        None jika format tidak dikenali
    """
    text = normalize_plate(raw_text)

    # Pola: 1-2 huruf awal | 1-4 angka | 1-3 huruf akhir
    # Spasi bersifat opsional untuk mengakomodasi variasi output OCR
    pattern = r"^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$"
    match = re.match(pattern, text)

    if not match:
        return None

    return {
        "prefix": match.group(1),   # misal: "BE"
        "number": match.group(2),   # misal: "1234"
        "suffix": match.group(3),   # misal: "AR"
    }



# Fungsi identifikasi wilayah


def identify_region(raw_text: str) -> dict:
    """
    Identifikasi wilayah asal kendaraan dari teks plat nomor.

    Args:
        raw_text: Teks plat nomor hasil OCR (misal "BE 1234 AR")

    Returns:
        dict dengan key:
            raw_text        : teks input asli
            normalized_text : teks setelah normalisasi
            is_lampung      : True jika prefix adalah 'BE'
            region_code     : huruf pertama suffix (kode wilayah)
            region_name     : nama wilayah Lampung, atau pesan error
            confidence      : 'high' | 'low' | 'unknown'
            notes           : catatan tambahan (misal potensi OCR error)
    """
    result = {
        "raw_text"        : raw_text,
        "normalized_text" : normalize_plate(raw_text),
        "is_lampung"      : False,
        "region_code"     : None,
        "region_name"     : "Tidak dikenali",
        "confidence"      : "unknown",
        "notes"           : "",
    }

    parsed = parse_plate(raw_text)

    # Gagal parse → format plat tidak valid
    if parsed is None:
        result["notes"] = "Format plat tidak dikenali. Periksa hasil OCR."
        return result

    # Cek apakah plat Lampung (prefix BE)
    if parsed["prefix"] != "BE":
        result["notes"] = f"Bukan plat Lampung. Prefix: '{parsed['prefix']}'"
        return result

    result["is_lampung"] = True

    # Ambil huruf pertama dari suffix sebagai kode wilayah
    region_code = parsed["suffix"][0]
    result["region_code"] = region_code

    # Lookup ke REGION_MAP
    if region_code in REGION_MAP:
        result["region_name"] = REGION_MAP[region_code]
        result["confidence"]  = "high"
    else:
        result["region_name"] = "Kode wilayah tidak dikenal"
        result["confidence"]  = "unknown"
        result["notes"]       = f"Huruf '{region_code}' tidak ada di REGION_MAP."
        return result

    # Cek potensi ambiguitas OCR pada kode wilayah
    ambiguous_notes = []
    for char_a, char_b in OCR_AMBIGUOUS_PAIRS:
        if region_code == char_a:
            # Cek apakah char_b juga valid sebagai kode wilayah
            if char_b in REGION_MAP:
                ambiguous_notes.append(
                    f"'{char_a}' mungkin terbaca sebagai '{char_b}' "
                    f"({REGION_MAP[char_b]}) oleh OCR."
                )
            else:
                ambiguous_notes.append(
                    f"'{char_a}' mungkin terbaca sebagai '{char_b}' (bukan kode wilayah valid) oleh OCR."
                )
        elif region_code == char_b:
            if char_a in REGION_MAP:
                ambiguous_notes.append(
                    f"'{char_b}' mungkin terbaca sebagai '{char_a}' "
                    f"({REGION_MAP[char_a]}) oleh OCR."
                )

    if ambiguous_notes:
        result["confidence"] = "low"
        result["notes"]      = " | ".join(ambiguous_notes)

    return result


def identify_region_batch(plate_list: list[str]) -> list[dict]:
    """
    Proses identifikasi wilayah untuk banyak plat sekaligus.

    Args:
        plate_list: List teks plat nomor

    Returns:
        List dict hasil identify_region untuk setiap plat
    """
    return [identify_region(plate) for plate in plate_list]



# Demo & self-test


def _run_demo():
    """Jalankan demo dan self-test sederhana."""

    test_cases = [
        # (input, expected_region)
        ("BE 1234 AR", "Kota Bandar Lampung"),    # A → Bandar Lampung
        ("BE 5678 FG", "Kota Metro"),              # F → Metro
        ("BE 9012 GP", "Kab. Lampung Tengah"),     # G → Lampung Tengah
        ("BE 3456 NT", "Kab. Lampung Timur"),      # N → Lampung Timur
        ("BE 7890 KI", "Kab. Lampung Utara"),      # K → Lampung Utara
        ("BE 1111 SR", "Kab. Tulang Bawang"),      # S → Tulang Bawang
        ("BE 2222 WP", "Kab. Way Kanan"),           # W → Way Kanan
        ("BE 3333 QN", "Kab. Tulang Bawang Barat"),# Q → Tulang Bawang Barat
        ("BE 4444 XR", "Kab. Pesisir Barat"),      # X → Pesisir Barat
        # Kasus OCR tidak ideal
        ("BE1234AR",   "Kota Bandar Lampung"),      # tanpa spasi
        ("be 1234 ar", "Kota Bandar Lampung"),      # lowercase
        # Kasus gagal
        ("B 1234 AR",  "Tidak dikenali"),           # bukan Lampung
        ("INVALID",    "Tidak dikenali"),            # format salah
        ("BE 1234",    "Tidak dikenali"),            # tidak ada suffix
    ]

    print("=" * 62)
    print("  DEMO region_mapper.py — Identifikasi Wilayah Plat Lampung")
    print("=" * 62)

    passed = 0
    failed = 0

    for raw_text, expected in test_cases:
        result  = identify_region(raw_text)
        actual  = result["region_name"]
        ok      = actual == expected
        status  = "✓" if ok else "✗"

        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n  [{status}] Input     : '{raw_text}'")
        print(f"       Normalized : '{result['normalized_text']}'")
        print(f"       Wilayah    : {actual}")
        print(f"       Confidence : {result['confidence']}")
        if result["notes"]:
            print(f"       Catatan    : {result['notes']}")
        if not ok:
            print(f"       !! Expected: {expected}")

    print(f"\n{'─' * 62}")
    print(f"  Hasil: {passed} lulus, {failed} gagal dari {len(test_cases)} test case")
    print(f"{'─' * 62}\n")

    # Tampilkan seluruh lookup table
    print("\n  LOOKUP TABLE — Kode Plat Lampung\n")
    print(f"  {'Kode':<6} {'Wilayah'}")
    print(f"  {'─'*4}   {'─'*30}")

    # Kelompokkan kode per wilayah
    from collections import defaultdict
    grouped: dict[str, list[str]] = defaultdict(list)
    for code, region in sorted(REGION_MAP.items()):
        grouped[region].append(code)

    for region, codes in sorted(grouped.items()):
        print(f"  {', '.join(codes):<6}  {region}")

    print()


if __name__ == "__main__":
    _run_demo()