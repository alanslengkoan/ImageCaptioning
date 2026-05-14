# Image Captioning Aksara Lontara
## Fine-Tuning BLIP | Tesis S2

---

## 📁 Struktur Proyek

```
ImageCaptioning/
├── Image_Captioning_Aksara_Lontara.ipynb  → Notebook utama (Google Colab)
├── requirements.txt                        → Daftar library
├── README.md
│
├── dataset/
│   ├── images/            → Gambar aksara mentah (95 file PNG)
│   │   ├── ltr-01.png     → a
│   │   ├── ltr-02.png     → i
│   │   ├── ltr-06.png     → ba
│   │   └── ... (ltr-01 s/d ltr-95)
│   ├── captions.json      → Caption per gambar (5 caption/gambar)
│   └── processed/         → Hasil split (dibuat oleh script)
│       ├── train/
│       │   ├── images/
│       │   └── captions_train.json
│       ├── val/
│       │   ├── images/
│       │   └── captions_val.json
│       └── test/
│           ├── images/
│           └── captions_test.json
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

## 🚀 Cara Penggunaan (Google Colab)

### 1. Upload Project ke Google Drive
Upload folder `ImageCaptioning/` ke Google Drive, misalnya di:
```
My Drive/ImageCaptioning/
```

### 2. Mount Google Drive & Pindah ke Folder Project
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ImageCaptioning
```

### 3. Install Library
```python
!pip install -r requirements.txt
```

### 4. Siapkan Dataset
```python
!python 1_prepare_dataset.py
```

### 5. Fine-Tuning Model
> ⚠️ Pastikan runtime Colab menggunakan **GPU** (Runtime → Change runtime type → GPU)
```python
!python 2_train_blip.py
```

### 6. Evaluasi Model
```python
!python 3_evaluate.py
```

### 7. Prediksi Caption (Inference)
```python
# Satu gambar
!python 4_inference.py --image dataset/processed/test/images/ltr-36.png

# Seluruh folder
!python 4_inference.py --folder dataset/processed/test/images

# Mode demo
!python 4_inference.py --demo
```

---

## 📊 Dataset

| Komponen             | Detail                                |
|----------------------|---------------------------------------|
| Karakter dasar       | 19 aksara Lontara                     |
| Vokal per karakter   | 5 (a, i, u, e, o)                    |
| Total gambar         | 95 gambar (ltr-01.png s/d ltr-95.png) |
| Caption per gambar   | 5 variasi caption                     |
| Total pasang         | 475 pasang (gambar + caption)         |
| Split Train/Val/Test | 80% / 10% / 10%                      |
| Format gambar        | PNG                                   |

### 19 Karakter Dasar Aksara Lontara:
`a, ba, ca, da, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, wa, ya`

### 5 Vokal per Karakter:
Setiap karakter dasar memiliki 5 varian vokal:
1. **a** - Vokal dasar (contoh: ba, ca, da)
2. **i** - Vokal i (contoh: bi, ci, di)
3. **u** - Vokal u (contoh: bu, cu, du)
4. **e** - Vokal e (contoh: be, ce, de)
5. **o** - Vokal o (contoh: bo, co, do)

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

1. **Augmentasi data**: Gunakan rotasi, zoom, brightness untuk memperbanyak 95 gambar
2. **Eksperimen**: Bandingkan hasil dengan ViT+GPT-2 sebagai baseline
3. **Analisis error**: Perhatikan karakter dasar mana yang paling sulit diprediksi (lihat BLEU-4 per karakter)
4. **Ablation study**: Coba variasi jumlah epoch, learning rate, beam size
5. **Analisis vokal**: Bandingkan akurasi prediksi antar varian vokal (a/i/u/e/o)

---

## 📚 Referensi

- Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. ICML.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
