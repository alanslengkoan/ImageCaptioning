"""
=============================================================
TESIS S2 - IMAGE CAPTIONING AKSARA LONTARA
File 3: Evaluasi Model
=============================================================
Metrik: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import nltk

# Download resource NLTK yang dibutuhkan
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
CONFIG = {
    "model_dir"   : "model/blip_lontara/best_model",
    "test_json"   : "dataset/processed/test/captions_test.json",
    "test_img_dir": "dataset/processed/test/images",
    "output_dir"  : "hasil_evaluasi",
    "max_length"  : 64,
    "num_beams"   : 4,              # beam search untuk caption lebih baik
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
}


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
def load_model(model_dir, device):
    print(f"📥 Memuat model dari: {model_dir}")
    processor = BlipProcessor.from_pretrained(model_dir)
    model     = BlipForConditionalGeneration.from_pretrained(model_dir)
    model     = model.to(device)
    model.eval()
    print("✅ Model berhasil dimuat.")
    return processor, model


# ─────────────────────────────────────────
# GENERATE CAPTION
# ─────────────────────────────────────────
def generate_caption(image_path, processor, model, device, max_length=64, num_beams=4):
    """Generate caption untuk satu gambar menggunakan beam search."""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return ""

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# ─────────────────────────────────────────
# METRIK EVALUASI
# ─────────────────────────────────────────
def hitung_bleu(referensi_list, hipotesis_list):
    """
    Menghitung BLEU-1 sampai BLEU-4.
    referensi_list: list of list of list of tokens
    hipotesis_list: list of list of tokens
    """
    smoother = SmoothingFunction().method1

    bleu1 = corpus_bleu(referensi_list, hipotesis_list, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = corpus_bleu(referensi_list, hipotesis_list, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu3 = corpus_bleu(referensi_list, hipotesis_list, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(referensi_list, hipotesis_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    return {
        "BLEU-1": round(bleu1 * 100, 2),
        "BLEU-2": round(bleu2 * 100, 2),
        "BLEU-3": round(bleu3 * 100, 2),
        "BLEU-4": round(bleu4 * 100, 2),
    }


def hitung_meteor(referensi_list, hipotesis_list):
    """Menghitung rata-rata METEOR score."""
    scores = []
    for refs, hyp in zip(referensi_list, hipotesis_list):
        # METEOR menerima list referensi sebagai string
        ref_strings = [" ".join(r) for r in refs]
        hyp_string  = " ".join(hyp)
        score = meteor_score(ref_strings, hyp_string)
        scores.append(score)
    return {"METEOR": round(np.mean(scores) * 100, 2)}


def hitung_rouge(referensi_list, hipotesis_list):
    """Menghitung ROUGE-L score."""
    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores  = []
    for refs, hyp in zip(referensi_list, hipotesis_list):
        hyp_str    = " ".join(hyp)
        ref_scores = [scorer.score(" ".join(r), hyp_str)["rougeL"].fmeasure for r in refs]
        scores.append(max(ref_scores))  # ambil skor tertinggi dari semua referensi
    return {"ROUGE-L": round(np.mean(scores) * 100, 2)}


# ─────────────────────────────────────────
# EVALUASI PER KARAKTER
# ─────────────────────────────────────────
def evaluasi_per_karakter(hasil_detail):
    """Menghitung skor BLEU-4 per karakter aksara."""
    karakter_data = {}

    for item in hasil_detail:
        karakter = item.get("karakter", "unknown")
        if karakter not in karakter_data:
            karakter_data[karakter] = {"referensi": [], "hipotesis": []}

        refs = [ref.split() for ref in item["captions_referensi"]]
        hyp  = item["caption_prediksi"].split()

        karakter_data[karakter]["referensi"].append(refs)
        karakter_data[karakter]["hipotesis"].append(hyp)

    hasil_per_karakter = {}
    smoother = SmoothingFunction().method1

    for karakter, data in karakter_data.items():
        bleu4 = corpus_bleu(
            data["referensi"],
            data["hipotesis"],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother
        )
        hasil_per_karakter[karakter] = round(bleu4 * 100, 2)

    return hasil_per_karakter


# ─────────────────────────────────────────
# MAIN EVALUASI
# ─────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  EVALUASI MODEL BLIP - AKSARA LONTARA")
    print("=" * 55)

    device = CONFIG["device"]
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load model
    processor, model = load_model(CONFIG["model_dir"], device)

    # Load test data
    with open(CONFIG["test_json"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"📂 Data test: {len(test_data)} entri\n")

    # Kelompokkan per gambar (1 gambar bisa punya banyak caption referensi)
    gambar_dict = {}
    for item in test_data:
        key = item["image"]
        if key not in gambar_dict:
            gambar_dict[key] = {
                "captions": [],
                "karakter": item.get("karakter", ""),
                "variasi" : item.get("variasi", ""),
            }
        gambar_dict[key]["captions"].append(item["caption"])

    # Generate caption & kumpulkan untuk evaluasi
    print("🔄 Generating caption untuk data test...")
    referensi_list = []
    hipotesis_list = []
    hasil_detail   = []

    for image_key, info in tqdm(gambar_dict.items(), desc="Evaluasi"):
        img_name = os.path.basename(image_key)
        img_path = os.path.join(CONFIG["test_img_dir"], img_name)

        # Generate caption
        pred_caption = generate_caption(
            img_path, processor, model, device,
            CONFIG["max_length"], CONFIG["num_beams"]
        )

        # Tokenisasi
        refs = [cap.lower().split() for cap in info["captions"]]
        hyp  = pred_caption.lower().split()

        referensi_list.append(refs)
        hipotesis_list.append(hyp)

        hasil_detail.append({
            "image"               : image_key,
            "karakter"            : info["karakter"],
            "variasi"             : info["variasi"],
            "caption_prediksi"    : pred_caption,
            "captions_referensi"  : info["captions"],
        })

    # Hitung semua metrik
    print("\n📊 Menghitung metrik evaluasi...")
    bleu_scores   = hitung_bleu(referensi_list, hipotesis_list)
    meteor_scores = hitung_meteor(referensi_list, hipotesis_list)
    rouge_scores  = hitung_rouge(referensi_list, hipotesis_list)
    per_karakter  = evaluasi_per_karakter(hasil_detail)

    semua_metrik = {**bleu_scores, **meteor_scores, **rouge_scores}

    # Tampilkan hasil
    print("\n" + "=" * 55)
    print("  HASIL EVALUASI")
    print("=" * 55)
    for metrik, nilai in semua_metrik.items():
        bar = "█" * int(nilai / 5)
        print(f"  {metrik:<10}: {nilai:>6.2f}%  {bar}")

    print("\n📋 BLEU-4 per Karakter Aksara:")
    print("-" * 40)
    for karakter, skor in sorted(per_karakter.items()):
        bar = "█" * int(skor / 5)
        print(f"  {karakter:<6}: {skor:>6.2f}%  {bar}")

    # Simpan hasil
    hasil_final = {
        "metrik_keseluruhan" : semua_metrik,
        "bleu4_per_karakter" : per_karakter,
        "detail_prediksi"    : hasil_detail,
        "config"             : CONFIG,
    }

    output_path = os.path.join(CONFIG["output_dir"], "hasil_evaluasi.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hasil_final, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hasil evaluasi disimpan: {output_path}")

    # Tampilkan beberapa contoh prediksi
    print("\n📝 Contoh Prediksi:")
    print("-" * 55)
    for item in hasil_detail[:5]:
        print(f"  Gambar    : {item['image']}")
        print(f"  Karakter  : {item['karakter']} | {item['variasi']}")
        print(f"  Prediksi  : {item['caption_prediksi']}")
        print(f"  Referensi : {item['captions_referensi'][0]}")
        print()

    print("=" * 55)
    print("  ✅ EVALUASI SELESAI!")
    print("=" * 55)


if __name__ == "__main__":
    main()
