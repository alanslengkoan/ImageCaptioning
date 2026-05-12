"""
=============================================================
TESIS S2 - IMAGE CAPTIONING AKSARA LONTARA
File 4: Inference - Prediksi Caption Gambar Baru
=============================================================
Gunakan file ini untuk memprediksi caption gambar aksara
Lontara yang belum pernah dilihat model sebelumnya.
"""

import os
import torch
import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
MODEL_DIR  = "model/blip_lontara/best_model"
MAX_LENGTH = 64
NUM_BEAMS  = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
def load_model(model_dir=MODEL_DIR):
    """Memuat model BLIP yang sudah di-fine-tune."""
    print(f"📥 Memuat model dari: {model_dir}")
    processor = BlipProcessor.from_pretrained(model_dir)
    model     = BlipForConditionalGeneration.from_pretrained(model_dir)
    model     = model.to(DEVICE)
    model.eval()
    print(f"✅ Model siap digunakan di {DEVICE}")
    return processor, model


# ─────────────────────────────────────────
# PREDIKSI CAPTION
# ─────────────────────────────────────────
def prediksi_caption(image_path, processor, model,
                     max_length=MAX_LENGTH,
                     num_beams=NUM_BEAMS):
    """
    Memprediksi caption dari satu gambar aksara Lontara.

    Args:
        image_path  : path ke file gambar
        processor   : BLIP processor
        model       : BLIP model yang sudah fine-tune
        max_length  : panjang maksimal caption
        num_beams   : jumlah beam untuk beam search

    Returns:
        caption (str): deskripsi aksara dalam Bahasa Indonesia
    """
    # Load dan proses gambar
    image   = Image.open(image_path).convert("RGB")
    inputs  = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Beam search untuk caption lebih akurat
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,   # hindari pengulangan frasa
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def prediksi_batch(image_paths, processor, model):
    """
    Memprediksi caption untuk banyak gambar sekaligus.

    Args:
        image_paths: list path gambar
        processor  : BLIP processor
        model      : BLIP model

    Returns:
        list of (image_path, caption)
    """
    results = []
    for idx, path in enumerate(image_paths, 1):
        print(f"  [{idx}/{len(image_paths)}] Memproses: {os.path.basename(path)}")
        caption = prediksi_caption(path, processor, model)
        results.append({"gambar": path, "caption": caption})
        print(f"   → {caption}\n")
    return results


def prediksi_folder(folder_path, processor, model, ekstensi=(".jpg", ".jpeg", ".png")):
    """
    Memprediksi caption untuk semua gambar dalam satu folder.

    Args:
        folder_path : path ke folder berisi gambar aksara
        processor   : BLIP processor
        model       : BLIP model
        ekstensi    : tuple ekstensi gambar yang diproses

    Returns:
        list hasil prediksi
    """
    image_paths = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(ekstensi)
    ]

    print(f"📂 Ditemukan {len(image_paths)} gambar di: {folder_path}\n")
    return prediksi_batch(image_paths, processor, model)


# ─────────────────────────────────────────
# CONTOH PENGGUNAAN
# ─────────────────────────────────────────
def demo():
    """Demo prediksi caption untuk semua 19 karakter."""
    print("\n" + "=" * 55)
    print("  DEMO INFERENCE - AKSARA LONTARA")
    print("  Image Captioning dengan BLIP Fine-Tuning")
    print("=" * 55)

    # Load model
    processor, model = load_model()

    # Contoh: prediksi satu gambar
    print("\n📌 Contoh Prediksi Satu Gambar:")
    print("-" * 40)

    contoh_gambar = "dataset/processed/test/images/ka_var1.jpg"

    if os.path.exists(contoh_gambar):
        caption = prediksi_caption(contoh_gambar, processor, model)
        print(f"  Gambar  : {contoh_gambar}")
        print(f"  Caption : {caption}")
    else:
        print(f"  ⚠️  File tidak ditemukan: {contoh_gambar}")
        print("  💡 Pastikan path gambar sudah benar.")

    # Contoh: prediksi folder
    print("\n📌 Contoh Prediksi Seluruh Folder Test:")
    print("-" * 40)

    test_folder = "dataset/processed/test/images"
    if os.path.exists(test_folder):
        hasil = prediksi_folder(test_folder, processor, model)

        # Simpan hasil
        import json
        output_path = "hasil_evaluasi/hasil_inference.json"
        os.makedirs("hasil_evaluasi", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hasil, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Hasil inference disimpan: {output_path}")
    else:
        print(f"  ⚠️  Folder tidak ditemukan: {test_folder}")

    print("\n" + "=" * 55)
    print("  ✅ DEMO SELESAI!")
    print("=" * 55)


# ─────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediksi caption Aksara Lontara menggunakan BLIP"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path ke satu gambar aksara Lontara"
    )
    parser.add_argument(
        "--folder", type=str, default=None,
        help="Path ke folder berisi gambar aksara Lontara"
    )
    parser.add_argument(
        "--model_dir", type=str, default=MODEL_DIR,
        help="Path ke folder model fine-tuned"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Jalankan mode demo"
    )

    args = parser.parse_args()

    if args.demo or (args.image is None and args.folder is None):
        demo()
    else:
        processor, model = load_model(args.model_dir)

        if args.image:
            print(f"\n🖼️  Gambar: {args.image}")
            caption = prediksi_caption(args.image, processor, model)
            print(f"📝 Caption: {caption}")

        elif args.folder:
            hasil = prediksi_folder(args.folder, processor, model)
            import json
            output = "hasil_evaluasi/hasil_inference.json"
            os.makedirs("hasil_evaluasi", exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(hasil, f, ensure_ascii=False, indent=2)
            print(f"✅ Disimpan: {output}")
