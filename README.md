# Image Captioning Aksara Lontara
## Fine-Tuning BLIP | Tesis S2

---

## 📁 Struktur Proyek

```
aksara_lontara_blip/
├── 1_prepare_dataset.py   → Persiapan & split dataset
├── 2_train_blip.py        → Fine-tuning model BLIP
├── 3_evaluate.py          → Evaluasi dengan BLEU, METEOR, ROUGE-L
├── 4_inference.py         → Prediksi caption gambar baru
├── requirements.txt       → Daftar library yang dibutuhkan
│
├── dataset/
│   ├── images/            → Gambar aksara mentah (per karakter)
│   │   ├── a/
│   │   ├── ba/
│   │   └── ... (19 karakter)
│   ├── captions.json      → Caption per gambar (dibuat otomatis)
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── model/
│   └── blip_lontara/
│       ├── best_model/    → Model terbaik
│       └── checkpoint_*/  → Checkpoint per epoch
│
└── hasil_evaluasi/
    ├── hasil_evaluasi.json
    └── hasil_inference.json
```

---

## 🚀 Cara Penggunaan

### 1. Install Library
```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset
```bash
python 1_prepare_dataset.py
```

### 3. Fine-Tuning Model
```bash
python 2_train_blip.py
```

### 4. Evaluasi Model
```bash
python 3_evaluate.py
```

### 5. Prediksi Caption (Inference)
```bash
# Satu gambar
python 4_inference.py --image dataset/images/ka/ka_var1.jpg

# Seluruh folder
python 4_inference.py --folder dataset/processed/test/images

# Mode demo
python 4_inference.py --demo
```

---

## 📊 Dataset

| Komponen         | Detail                         |
|------------------|-------------------------------|
| Karakter dasar   | 19 aksara Lontara              |
| Variasi          | 5 per karakter                 |
| Total gambar     | 95 gambar                      |
| Caption/gambar   | 5 variasi caption              |
| Total pasang     | 475 pasang (gambar + caption)  |
| Split Train/Val/Test | 80% / 10% / 10%           |

### 19 Karakter Aksara Lontara:
`a, ba, ca, da, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, wa, ya`

### 5 Variasi per Karakter:
1. `var1` - Aksara dasar
2. `var2` - Dengan tanda titik tunggal di atas
3. `var3` - Dengan tanda dua titik di atas
4. `var4` - Dengan tanda awalan (sisi kiri)
5. `var5` - Dengan tanda akhiran (kanan atas)

---

## 🤖 Model

- **Base Model**: `Salesforce/blip-image-captioning-base`
- **Task**: Image Captioning (Bahasa Indonesia)
- **Metode**: Fine-Tuning dengan dataset Aksara Lontara
- **Image Size**: 384 × 384 px
- **Optimizer**: AdamW (lr=2e-5)
- **Epoch**: 20
- **Batch Size**: 8

---

## 📈 Metrik Evaluasi

| Metrik   | Keterangan                                      |
|----------|-------------------------------------------------|
| BLEU-1   | Kemiripan unigram caption prediksi vs referensi |
| BLEU-2   | Kemiripan bigram                                |
| BLEU-3   | Kemiripan trigram                               |
| BLEU-4   | Kemiripan 4-gram (standar paper)                |
| METEOR   | Mempertimbangkan sinonim & stemming             |
| ROUGE-L  | Berbasis longest common subsequence             |

---

## 💡 Tips untuk Tesis S2

1. **Augmentasi data**: Gunakan rotasi, zoom, brightness untuk memperbanyak data
2. **Eksperimen**: Bandingkan hasil dengan ViT+GPT-2 sebagai baseline
3. **Analisis error**: Perhatikan karakter mana yang paling sulit diprediksi
4. **Ablation study**: Coba variasi jumlah epoch, learning rate, beam size

---

## 📚 Referensi

- Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. ICML.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
