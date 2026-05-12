"""
=============================================================
TESIS S2 - IMAGE CAPTIONING AKSARA LONTARA
File 1: Persiapan Dataset
=============================================================
Fungsi: Mempersiapkan dataset gambar + caption JSON
        menjadi format siap pakai untuk fine-tuning BLIP
"""

import os
import json
import shutil
import random
from PIL import Image
import torchvision.transforms as transforms

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
RAW_IMAGE_DIR   = "dataset/images"          # folder gambar mentah
CAPTION_FILE    = "dataset/captions.json"   # file caption JSON
OUTPUT_DIR      = "dataset/processed"       # output hasil proses
IMAGE_SIZE      = 384                       # ukuran gambar untuk BLIP
TRAIN_RATIO     = 0.8                       # 80% data training
VAL_RATIO       = 0.1                       # 10% data validasi
TEST_RATIO      = 0.1                       # 10% data testing
RANDOM_SEED     = 42

# 19 Karakter Aksara Lontara
KARAKTER_LONTARA = [
    "a", "ba", "ca", "da", "ga",
    "ha", "ja", "ka", "la", "ma",
    "na", "nga", "nya", "pa", "ra",
    "sa", "ta", "wa", "ya"
]

# 5 Jenis Variasi per Karakter
VARIASI = [
    "var1_dasar",
    "var2_titik_atas",
    "var3_titik_ganda",
    "var4_awalan",
    "var5_akhiran"
]


# ─────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────

def buat_struktur_folder():
    """Membuat struktur folder dataset yang terorganisir."""
    folders = [
        f"{OUTPUT_DIR}/train/images",
        f"{OUTPUT_DIR}/val/images",
        f"{OUTPUT_DIR}/test/images",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Struktur folder berhasil dibuat.")


def load_captions(caption_file):
    """Membaca file caption JSON."""
    with open(caption_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Caption dimuat: {len(data)} entri ditemukan.")
    return data


def preprocess_gambar(image_path, output_path, size=IMAGE_SIZE):
    """
    Resize dan normalisasi gambar untuk BLIP.
    BLIP menggunakan ukuran 384x384.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((size, size), Image.LANCZOS)
    img_resized.save(output_path)


def split_dataset(data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Membagi dataset menjadi train, val, test.
    Stratified by karakter agar distribusi merata.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = data[:n_train]
    val   = data[n_train:n_train + n_val]
    test  = data[n_train + n_val:]

    print(f"📊 Pembagian dataset:")
    print(f"   Train : {len(train)} pasang ({train_ratio*100:.0f}%)")
    print(f"   Val   : {len(val)} pasang ({val_ratio*100:.0f}%)")
    print(f"   Test  : {len(test)} pasang ({(1-train_ratio-val_ratio)*100:.0f}%)")

    return train, val, test


def simpan_split_json(train, val, test):
    """Menyimpan hasil split ke file JSON terpisah."""
    splits = {"train": train, "val": val, "test": test}
    for split_name, split_data in splits.items():
        output_path = f"{OUTPUT_DIR}/{split_name}/captions_{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Disimpan: {output_path} ({len(split_data)} entri)")


def proses_gambar_semua(data, split_name):
    """Memproses semua gambar untuk satu split."""
    print(f"\n🔄 Memproses gambar untuk split: {split_name}")
    for item in data:
        src  = os.path.join(RAW_IMAGE_DIR, item["image"])
        dst  = os.path.join(OUTPUT_DIR, split_name, "images", os.path.basename(item["image"]))
        if os.path.exists(src):
            preprocess_gambar(src, dst)
        else:
            print(f"⚠️  Gambar tidak ditemukan: {src}")
    print(f"✅ Selesai memproses gambar untuk {split_name}.")


def buat_contoh_caption_json():
    """
    Membuat contoh file captions.json berdasarkan
    19 karakter × 5 variasi × 5 caption = 475 pasang.
    """
    # Template caption per karakter per variasi
    template = {
        "a": {
            "bentuk": "dua lengkungan simetris menyerupai mahkota dengan ujung menghadap ke atas",
            "goresan": "dua goresan melengkung yang bertemu di bagian bawah"
        },
        "ba": {
            "bentuk": "goresan melengkung ke kiri menyerupai angka dua terbalik",
            "goresan": "satu goresan spiral yang melengkung ke arah kiri bawah"
        },
        "ca": {
            "bentuk": "lengkungan spiral ke dalam menghadap kiri bawah",
            "goresan": "goresan melingkar yang berputar ke dalam dari atas ke kiri"
        },
        "da": {
            "bentuk": "lengkungan seperti tapal kuda menghadap ke bawah",
            "goresan": "goresan melengkung tunggal yang membentuk setengah lingkaran ke bawah"
        },
        "ga": {
            "bentuk": "dua lengkungan kecil seperti alis yang simetris",
            "goresan": "dua goresan melengkung kecil yang berdampingan secara horizontal"
        },
        "ha": {
            "bentuk": "dua lingkaran penuh yang berdampingan secara horizontal",
            "goresan": "dua goresan melingkar tertutup yang sejajar"
        },
        "ja": {
            "bentuk": "goresan melengkung seperti ekor dengan ujung spiral",
            "goresan": "satu goresan panjang yang melengkung dan berputar di ujungnya"
        },
        "ka": {
            "bentuk": "dua lengkungan bertumpuk menyerupai huruf M",
            "goresan": "dua goresan melengkung yang saling berhubungan di bagian tengah"
        },
        "la": {
            "bentuk": "goresan lengkung tunggal menyerupai huruf V terbalik",
            "goresan": "satu goresan melengkung yang membentuk puncak di bagian atas"
        },
        "ma": {
            "bentuk": "dua lengkungan seperti huruf M dengan lekukan yang dalam",
            "goresan": "dua goresan melengkung besar yang bertemu di bagian tengah bawah"
        },
        "na": {
            "bentuk": "lengkungan tunggal sederhana menyerupai huruf V",
            "goresan": "satu goresan melengkung yang membentuk sudut di bagian bawah"
        },
        "nga": {
            "bentuk": "goresan diagonal menyerupai petir atau angka tujuh",
            "goresan": "satu goresan miring dengan sudut tajam di bagian tengah"
        },
        "nya": {
            "bentuk": "goresan spiral ganda yang saling melengkung",
            "goresan": "dua goresan spiral kecil yang terhubung secara vertikal"
        },
        "pa": {
            "bentuk": "dua lengkungan kecil yang menghadap ke bawah",
            "goresan": "dua goresan melengkung kecil yang berdampingan dengan ujung ke bawah"
        },
        "ra": {
            "bentuk": "lengkungan tunggal kecil menyerupai topi atau cangkir terbalik",
            "goresan": "satu goresan melengkung kecil yang membentuk busur di bagian atas"
        },
        "sa": {
            "bentuk": "lingkaran oval tunggal yang tertutup sempurna",
            "goresan": "satu goresan melingkar yang membentuk elips tertutup"
        },
        "ta": {
            "bentuk": "lengkungan sederhana menyerupai simbol caret menghadap ke atas",
            "goresan": "satu goresan melengkung ringan dengan puncak di bagian tengah atas"
        },
        "wa": {
            "bentuk": "goresan melengkung panjang yang memanjang ke kanan",
            "goresan": "satu goresan panjang yang melengkung dengan ekor ke arah kanan"
        },
        "ya": {
            "bentuk": "dua lengkungan besar simetris menyerupai mahkota besar",
            "goresan": "dua goresan melengkung besar yang berdampingan dan simetris"
        },
    }

    variasi_deskripsi = {
        "var1_dasar": {
            "label": "dasar",
            "tambahan": "tanpa tanda tambahan"
        },
        "var2_titik_atas": {
            "label": "dengan tanda titik tunggal di atas",
            "tambahan": "dilengkapi satu titik penanda bunyi di bagian atas"
        },
        "var3_titik_ganda": {
            "label": "dengan tanda dua titik di atas",
            "tambahan": "dilengkapi dua titik penanda bunyi di bagian atas"
        },
        "var4_awalan": {
            "label": "dengan tanda awalan di sisi kiri",
            "tambahan": "dilengkapi goresan pembuka di sisi kiri"
        },
        "var5_akhiran": {
            "label": "dengan tanda akhiran di kanan atas",
            "tambahan": "dilengkapi goresan penutup di bagian kanan atas"
        },
    }

    dataset = []

    for karakter in KARAKTER_LONTARA:
        info   = template[karakter]
        bentuk = info["bentuk"]
        goresan = info["goresan"]

        for var_key, var_info in variasi_deskripsi.items():
            label    = var_info["label"]
            tambahan = var_info["tambahan"]

            image_path = f"{karakter}/{karakter}_{var_key}.jpg"

            # 5 variasi caption per gambar
            captions = [
                f"Aksara {karakter} {label} berbentuk {bentuk}",
                f"Bentuk aksara {karakter} {label} terdiri dari {goresan} yang {tambahan}",
                f"Aksara {karakter} memiliki {goresan} dengan ciri {label}",
                f"Goresan aksara {karakter} {label} membentuk {bentuk} dan {tambahan}",
                f"Karakter aksara Lontara {karakter} {label} ditandai dengan {bentuk} serta {tambahan}",
            ]

            dataset.append({
                "image": image_path,
                "karakter": karakter,
                "variasi": var_key,
                "captions": captions
            })

    # Simpan ke file
    os.makedirs("dataset", exist_ok=True)
    with open(CAPTION_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ File captions.json berhasil dibuat!")
    print(f"   Total entri  : {len(dataset)}")
    print(f"   Total caption: {len(dataset) * 5}")
    print(f"   Disimpan di  : {CAPTION_FILE}")
    return dataset


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PERSIAPAN DATASET AKSARA LONTARA")
    print("  Image Captioning dengan BLIP Fine-Tuning")
    print("=" * 55)

    # Step 1: Buat contoh caption JSON
    print("\n📝 Step 1: Membuat file captions.json...")
    data = buat_contoh_caption_json()

    # Step 2: Buat struktur folder
    print("\n📁 Step 2: Membuat struktur folder...")
    buat_struktur_folder()

    # Step 3: Flatten captions (1 gambar = 1 caption terpilih untuk training)
    print("\n🔄 Step 3: Mempersiapkan data untuk split...")
    flat_data = []
    for item in data:
        for caption in item["captions"]:
            flat_data.append({
                "image": item["image"],
                "caption": caption,
                "karakter": item["karakter"],
                "variasi": item["variasi"]
            })
    print(f"   Total pasang (gambar × caption): {len(flat_data)}")

    # Step 4: Split dataset
    print("\n✂️  Step 4: Membagi dataset...")
    train, val, test = split_dataset(flat_data)

    # Step 5: Simpan JSON split
    print("\n💾 Step 5: Menyimpan file JSON per split...")
    simpan_split_json(train, val, test)

    print("\n" + "=" * 55)
    print("  ✅ PERSIAPAN DATASET SELESAI!")
    print(f"  Total data    : {len(flat_data)} pasang")
    print(f"  Train         : {len(train)} pasang")
    print(f"  Val           : {len(val)} pasang")
    print(f"  Test          : {len(test)} pasang")
    print("=" * 55)
