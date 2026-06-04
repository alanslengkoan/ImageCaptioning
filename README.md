# Image Captioning Aksara Lontara
## Perbandingan BLIP vs GIT vs BLIP-2 | Tesis S2

---

## 📁 Struktur Proyek

```
ImageCaptioning/
├── training_blip.ipynb       → Fine-tuning BLIP
├── training_git.ipynb        → Fine-tuning GIT
├── training_blip2.ipynb      → Fine-tuning BLIP-2 (8-bit)
├── inference_blip.ipynb      → Inference multi-karakter (BLIP)
├── inference_git.ipynb       → Inference multi-karakter (GIT)
├── inference_blip2.ipynb     → Inference multi-karakter (BLIP-2)
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── images/               → 95 gambar aksara (PNG)
│   ├── captions.json         → 5 caption per gambar
│   └── processed/            → Hasil split (dibuat otomatis)
│       ├── train/
│       ├── val/
│       └── test/
│
├── datatest/                 → Gambar uji multi-karakter
│
├── model/                    → Hasil training (dibuat otomatis)
│   ├── blip/
│   │   ├── best_model/
│   │   └── checkpoint_*/
│   ├── git/
│   │   ├── best_model/
│   │   └── checkpoint_*/
│   └── blip2/
│       ├── best_model/
│       └── checkpoint_*/
│
└── hasil_evaluasi/           → Metrik & visualisasi
    ├── evaluasi_blip.json
    ├── evaluasi_git.json
    └── evaluasi_blip2.json
```

---

## 🚀 Cara Penggunaan (Google Colab)

### 1. Upload ke Google Drive
Upload folder `ImageCaptioning/` ke:
```
My Drive/ImageCaptioning/
```

### 2. Training (Jalankan Salah Satu)

> ⚠️ Pastikan runtime menggunakan **GPU T4** (Runtime → Change runtime type → GPU)

| Notebook | Model | Waktu ~T4 |
|----------|-------|-----------|
| `training_blip.ipynb` | Salesforce/blip-image-captioning-base | ~30 menit |
| `training_git.ipynb` | microsoft/git-base | ~25 menit |
| `training_blip2.ipynb` | Salesforce/blip2-opt-2.7b (8-bit) | ~60 menit |

Setiap notebook sudah mencakup: setup → dataset → training → evaluasi → simpan ke Drive.

### 3. Inference Multi-Karakter

Setelah training selesai, gunakan notebook inference yang sesuai:

| Notebook | Untuk Model |
|----------|-------------|
| `inference_blip.ipynb` | BLIP fine-tuned |
| `inference_git.ipynb` | GIT fine-tuned |
| `inference_blip2.ipynb` | BLIP-2 fine-tuned |

Fitur inference:
- Deteksi **1–4 karakter** per gambar
- Vertical Projection Segmentation
- Canvas 384×384 (konsisten dengan data training)
- Visualisasi debug (bounding box, projection profile)
- Batch inference seluruh folder

---

## 📊 Dataset

| Komponen             | Detail                                |
|----------------------|---------------------------------------|
| Karakter dasar       | 19 aksara Lontara                     |
| Variasi per karakter | 5 (a, i, u, e, o)                    |
| Total gambar         | 95 gambar (ltr-01.png s/d ltr-95.png) |
| Caption per gambar   | 5 variasi caption (Bahasa Indonesia)  |
| Total pasang         | 475 pasang (gambar + caption)         |
| Split Train/Val/Test | 80% / 10% / 10% (stratified)         |

### 19 Karakter Dasar:
`a, ba, ca, da, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, wa, ya`

---

## 🤖 Model

| Model | Base | Parameter | Batch | Teknik Khusus |
|-------|------|-----------|-------|---------------|
| BLIP | `Salesforce/blip-image-captioning-base` | ~247M | 8 | Full fine-tune |
| GIT | `microsoft/git-base` | ~177M | 8 | Full fine-tune |
| BLIP-2 | `Salesforce/blip2-opt-2.7b` | ~3.7B | 4 | 8-bit quantization, freeze ViT+LLM, tune Q-Former only |

**Konfigurasi training (sama untuk ketiga model):**
- Image Size: 384×384 px
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Epochs: 20
- Checkpoint: setiap 5 epoch
- Scheduler: Linear warmup

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

Evaluasi dilakukan terpisah untuk:
- **Keseluruhan** (semua gambar test)
- **1 karakter** (gambar tunggal)
- **2 karakter** (gambar multi-karakter)

---

## 📚 Referensi

- Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. ICML.
- Wang, J., et al. (2022). GIT: A Generative Image-to-text Transformer. TMLR.
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.
