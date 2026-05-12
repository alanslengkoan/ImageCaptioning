"""
=============================================================
TESIS S2 - IMAGE CAPTIONING AKSARA LONTARA
File 2: Fine-Tuning Model BLIP
=============================================================
Model  : Salesforce/blip-image-captioning-base
Tugas  : Image Captioning Aksara Lontara (Bahasa Indonesia)
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# KONFIGURASI TRAINING
# ─────────────────────────────────────────
CONFIG = {
    # Model
    "model_name"     : "Salesforce/blip-image-captioning-base",
    "output_dir"     : "model/blip_lontara",

    # Dataset
    "train_json"     : "dataset/processed/train/captions_train.json",
    "val_json"       : "dataset/processed/val/captions_val.json",
    "train_img_dir"  : "dataset/processed/train/images",
    "val_img_dir"    : "dataset/processed/val/images",

    # Training
    "num_epochs"     : 20,
    "batch_size"     : 8,
    "learning_rate"  : 2e-5,         # kecil agar tidak lupa pengetahuan lama
    "weight_decay"   : 0.01,
    "warmup_steps"   : 100,
    "max_length"     : 64,           # panjang maksimal caption
    "image_size"     : 384,

    # Logging
    "save_every"     : 5,            # simpan checkpoint setiap N epoch
    "log_every"      : 10,           # log setiap N step
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"🖥️  Device yang digunakan: {CONFIG['device']}")
print(f"📦 Model: {CONFIG['model_name']}")


# ─────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────
class AksaraLontaraDataset(Dataset):
    """
    Dataset class untuk Aksara Lontara.
    Setiap item berupa pasangan (gambar, caption).
    """
    def __init__(self, json_file, image_dir, processor, max_length=64):
        self.processor  = processor
        self.image_dir  = image_dir
        self.max_length = max_length

        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"   📂 Dataset dimuat: {len(self.data)} pasang dari {json_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item    = self.data[idx]
        caption = item["caption"]

        # Load gambar
        img_name = os.path.basename(item["image"])
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback: buat gambar putih jika tidak ditemukan (untuk testing)
            image = Image.new("RGB", (384, 384), color=(255, 255, 255))

        # Proses dengan BLIP processor
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Squeeze dimensi batch
        input_ids      = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        pixel_values   = encoding["pixel_values"].squeeze()

        # Labels = input_ids (untuk language modeling)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values"  : pixel_values,
            "input_ids"     : input_ids,
            "attention_mask": attention_mask,
            "labels"        : labels,
        }


# ─────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Satu epoch training."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for step, batch in enumerate(progress):
        # Pindahkan ke device
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss    = total_loss / (step + 1)

        progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    return total_loss / num_batches


def val_epoch(model, dataloader, device, epoch):
    """Satu epoch validasi."""
    model.eval()
    total_loss  = 0
    num_batches = len(dataloader)

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ", leave=False)

    with torch.no_grad():
        for batch in progress:
            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            progress.set_postfix({"val_loss": f"{total_loss / (len(progress)):.4f}"})

    return total_loss / num_batches


def simpan_checkpoint(model, processor, optimizer, epoch, val_loss, output_dir):
    """Menyimpan checkpoint model."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    # Simpan info training
    info = {
        "epoch"    : epoch,
        "val_loss" : val_loss,
    }
    with open(os.path.join(checkpoint_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"   💾 Checkpoint disimpan: {checkpoint_dir}")


# ─────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  FINE-TUNING BLIP - AKSARA LONTARA")
    print("  Image Captioning | Bahasa Indonesia")
    print("=" * 55)

    device = CONFIG["device"]
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── Load Processor & Model ──────────────────────────
    print("\n📥 Step 1: Memuat model BLIP...")
    processor = BlipProcessor.from_pretrained(CONFIG["model_name"])
    model     = BlipForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    model     = model.to(device)
    print(f"   ✅ Model dimuat ({sum(p.numel() for p in model.parameters()):,} parameter)")

    # ── Dataset & DataLoader ────────────────────────────
    print("\n📂 Step 2: Memuat dataset...")
    train_dataset = AksaraLontaraDataset(
        CONFIG["train_json"],
        CONFIG["train_img_dir"],
        processor,
        CONFIG["max_length"]
    )
    val_dataset = AksaraLontaraDataset(
        CONFIG["val_json"],
        CONFIG["val_img_dir"],
        processor,
        CONFIG["max_length"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    # ── Optimizer & Scheduler ───────────────────────────
    print("\n⚙️  Step 3: Menyiapkan optimizer & scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    total_steps   = len(train_loader) * CONFIG["num_epochs"]
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps
    )
    print(f"   Total training steps: {total_steps}")
    print(f"   Warmup steps        : {CONFIG['warmup_steps']}")

    # ── Training Loop ───────────────────────────────────
    print(f"\n🚀 Step 4: Mulai training ({CONFIG['num_epochs']} epoch)...")
    print("-" * 55)

    history      = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        # Validasi
        val_loss = val_epoch(model, val_loader, device, epoch)

        # Simpan history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"\n📊 Epoch {epoch:02d}/{CONFIG['num_epochs']:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # Simpan model terbaik
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(CONFIG["output_dir"], "best_model")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"   ⭐ Model terbaik disimpan! Val Loss: {best_val_loss:.4f}")

        # Simpan checkpoint berkala
        if epoch % CONFIG["save_every"] == 0:
            simpan_checkpoint(model, processor, optimizer, epoch, val_loss, CONFIG["output_dir"])

    # ── Simpan History ──────────────────────────────────
    history_path = os.path.join(CONFIG["output_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 55)
    print("  ✅ FINE-TUNING SELESAI!")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Model terbaik : {CONFIG['output_dir']}/best_model")
    print(f"  History       : {history_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
