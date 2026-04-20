"""
02_data_balancing.py — AgroDetect Dataset Balancing
=====================================================
Strategy (best-in-class for image classification):
  • Primary: Offline augmentation for minority classes → oversample to target_count
  • Secondary: WeightedRandomSampler at DataLoader level (class-frequency inverse weights)
  • Tertiary: Class-weighted CrossEntropyLoss during training (done in utils.py)

Augmentations used for oversampling:
  RandomHFlip, RandomVFlip, RandomRotation, ColorJitter,
  RandomCrop+Pad, GaussianBlur, CLAHE-like contrast, ShiftScaleRotate

Outputs:
  • Dataset/balanced_train.csv       — extended index (original + synthetic rows)
  • Dataset/aug_images/              — augmented images saved to disk
  • outputs/plots/balance_*.png      — visualisations
  • outputs/results/balance_report.json

Run:
  python 02_data_balancing.py
  (must run 01_data_cleaning.py first)
"""

import os, sys, json, time, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, os.path.dirname(__file__))
from config import *

# ═══════════════════════════════════════════════════════════════════════════
# Target count
# ═══════════════════════════════════════════════════════════════════════════
# Oversample minority classes to at least TARGET_COUNT images.
# We don't undersample majority — we rely on WeightedRandomSampler.
TARGET_COUNT   = 1700    # ~match the largest class (normal=1764, blast=1738)
AUG_IMAGES_DIR = os.path.join(DATASET_DIR, "aug_images")
os.makedirs(AUG_IMAGES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation pipeline
# ═══════════════════════════════════════════════════════════════════════════
def augment_image(img: Image.Image, seed: int = None) -> Image.Image:
    """Apply a random sequence of augmentations to produce a new variant."""
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    ops = []

    # Flip
    if random.random() > 0.5: ops.append(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT))
    if random.random() > 0.5: ops.append(lambda x: x.transpose(Image.FLIP_TOP_BOTTOM))

    # Rotation
    angle = random.uniform(-30, 30)
    ops.append(lambda x, a=angle: x.rotate(a, resample=Image.BILINEAR, expand=False))

    # Scale + Crop
    scale = random.uniform(0.8, 1.2)
    def scale_crop(x, s=scale):
        w, h = x.size
        nw, nh = int(w * s), int(h * s)
        x = x.resize((nw, nh), Image.BILINEAR)
        # centre crop back to original
        left = max(0, (nw - w) // 2)
        top  = max(0, (nh - h) // 2)
        return x.crop((left, top, left + w, top + h)).resize((w, h), Image.BILINEAR)
    ops.append(scale_crop)

    # Color jitter
    def color_jitter(x):
        x = ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3))
        x = ImageEnhance.Contrast(x).enhance(random.uniform(0.7, 1.3))
        x = ImageEnhance.Saturation(x).enhance(random.uniform(0.7, 1.3))
        x = ImageEnhance.Sharpness(x).enhance(random.uniform(0.5, 2.0))
        return x
    if random.random() > 0.3: ops.append(color_jitter)

    # Blur
    if random.random() > 0.7:
        r = random.uniform(0.5, 1.5)
        ops.append(lambda x, rr=r: x.filter(ImageFilter.GaussianBlur(radius=rr)))

    # Noise (grid-level — additive)
    if random.random() > 0.7:
        def add_noise(x):
            arr = np.array(x, dtype=np.float32)
            noise = np.random.normal(0, 10, arr.shape)
            return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        ops.append(add_noise)

    # Auto-contrast
    if random.random() > 0.5: ops.append(ImageOps.autocontrast)

    # Apply ops in sequence
    out = img
    for op in ops:
        try:
            out = op(out)
        except Exception:
            pass
    return out.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════
# Balancing
# ═══════════════════════════════════════════════════════════════════════════
def balance_dataset(
    cleaned_csv: str = CLEANED_CSV,
    images_dir:  str = TRAIN_IMAGES,
    aug_dir:     str = AUG_IMAGES_DIR,
    out_csv:     str = BALANCED_CSV,
    target_count:int = TARGET_COUNT,
) -> dict:

    print("\n" + "=" * 60)
    print("  AgroDetect — Dataset Balancing")
    print("=" * 60)

    if not os.path.exists(cleaned_csv):
        print(f"[!] cleaned_train.csv not found — falling back to train.csv")
        cleaned_csv = TRAIN_CSV

    df = pd.read_csv(cleaned_csv)

    # If cleaning produced an empty CSV (path mismatch), fall back to train.csv
    if len(df) == 0:
        print(f"[!] cleaned_train.csv is empty — falling back to train.csv")
        df = pd.read_csv(TRAIN_CSV)

    print(f"\n[1] Loaded {len(df):,} cleaned images")

    report = {
        "class_before": {},
        "class_after":  {},
        "augmented_per_class": {},
        "target_count": target_count,
    }

    for cls in CLASS_NAMES:
        report["class_before"][cls] = int((df["label"] == cls).sum())

    # ── Augment minority classes ────────────────────────────────────────────
    print(f"\n[2] Augmenting minority classes to target={target_count} …")
    aug_rows = []

    for cls in CLASS_NAMES:
        cls_df = df[df["label"] == cls]
        n_have = len(cls_df)
        n_need = max(0, target_count - n_have)

        if n_need == 0:
            print(f"    {cls:<35s} {n_have:>4d}  → no augmentation needed")
            report["augmented_per_class"][cls] = 0
            continue

        print(f"    {cls:<35s} {n_have:>4d}  → +{n_need} synthetic images")
        report["augmented_per_class"][cls] = n_need

        # Cycle over existing images to generate n_need new ones
        src_rows = cls_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_src    = len(src_rows)

        for i in tqdm(range(n_need), desc=f"  {cls}", ncols=70, leave=False):
            src_row  = src_rows.iloc[i % n_src]
            # Support both class-subfolder and flat layouts
            src_path = os.path.join(images_dir, src_row["label"], src_row["image_id"])
            if not os.path.exists(src_path):
                src_path = os.path.join(images_dir, src_row["image_id"])
            aug_id   = f"aug_{cls}_{i:06d}.jpg"
            aug_path = os.path.join(aug_dir, aug_id)

            if not os.path.exists(aug_path):   # skip if already generated
                try:
                    img  = Image.open(src_path).convert("RGB")
                    aug  = augment_image(img, seed=SEED + i)
                    aug.save(aug_path, "JPEG", quality=92)
                except Exception as e:
                    print(f"      [!] Failed {src_row['image_id']}: {e}")
                    continue

            aug_rows.append({
                "image_id": aug_id,
                "label":    cls,
                "variety":  src_row["variety"],
                "age":      src_row["age"],
                "is_augmented": True,
                "source_img": src_row["image_id"],
            })

    # ── Merge ──────────────────────────────────────────────────────────────
    df["is_augmented"] = False
    df["source_img"]   = ""
    aug_df   = pd.DataFrame(aug_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)
    final_df.to_csv(out_csv, index=False)

    for cls in CLASS_NAMES:
        report["class_after"][cls] = int((final_df["label"] == cls).sum())

    print(f"\n[3] Balanced dataset: {len(final_df):,} images")
    print(f"    Saved → {out_csv}")
    print(f"    Augmented images in → {aug_dir}")

    # ── Save report ────────────────────────────────────────────────────────
    rep_path = os.path.join(RESULTS_DIR, "balance_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Report → {rep_path}")

    return report, final_df


# ═══════════════════════════════════════════════════════════════════════════
# Weighted sampler stats helper
# ═══════════════════════════════════════════════════════════════════════════
def compute_effective_sampling(df: pd.DataFrame) -> dict:
    """Compute WeightedRandomSampler weights for a given balanced DF."""
    counts = df["label"].value_counts()
    weights = {cls: 1.0 / counts.get(cls, 1) for cls in CLASS_NAMES}
    total_w = sum(weights.values())
    effective = {cls: w / total_w for cls, w in weights.items()}
    return effective


# ═══════════════════════════════════════════════════════════════════════════
# Visualisations
# ═══════════════════════════════════════════════════════════════════════════
def plot_balance_results(report: dict, final_df: pd.DataFrame):
    sns.set_style("whitegrid")
    palette_before = sns.color_palette("Reds_r",   NUM_CLASSES)
    palette_after  = sns.color_palette("Greens_r", NUM_CLASSES)

    # ── 1. Before / After comparison ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, key, palette, title in zip(
            axes,
            ["class_before", "class_after"],
            [palette_before, palette_after],
            ["Before Balancing (cleaned)", f"After Balancing (target={report['target_count']})"]):
        vals  = [report[key][c] for c in CLASS_NAMES]
        bars  = ax.barh(CLASS_NAMES, vals, color=palette, edgecolor="black", linewidth=0.5)
        ax.set(title=title, xlabel="Image Count")
        ax.axvline(report["target_count"], color="red", linestyle="--",
                   linewidth=1.5, label=f"Target={report['target_count']}")
        ax.legend()
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=9)
    plt.suptitle("Dataset Balancing — Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "balance_class_dist.png"), dpi=150)
    plt.close()
    print(f"   Saved: balance_class_dist.png")

    # ── 2. Augmentation count per class bar ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    aug_counts = [report["augmented_per_class"].get(c, 0) for c in CLASS_NAMES]
    orig_counts = [report["class_before"][c] for c in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    ax.bar(x - w/2, orig_counts, w, label="Original",    color="#4e79a7")
    ax.bar(x + w/2, aug_counts,  w, label="Augmented",   color="#f28e2b")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set(title="Original vs Augmented Images per Class", ylabel="Count")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "balance_aug_per_class.png"), dpi=150)
    plt.close()
    print(f"   Saved: balance_aug_per_class.png")

    # ── 3. Imbalance ratio before/after ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    before_vals = [report["class_before"][c] for c in CLASS_NAMES]
    after_vals  = [report["class_after"][c]  for c in CLASS_NAMES]
    ratio_before = [max(before_vals) / (v + 1e-6) for v in before_vals]
    ratio_after  = [max(after_vals)  / (v + 1e-6) for v in after_vals]
    x = np.arange(len(CLASS_NAMES))
    ax.plot(x, ratio_before, "o-", label="Before", color="red",   linewidth=2)
    ax.plot(x, ratio_after,  "s-", label="After",  color="green", linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.axhline(1, color="gray", linestyle="--")
    ax.set(title="Class Imbalance Ratio (max_count / class_count)",
           ylabel="Imbalance Ratio")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "balance_imbalance_ratio.png"), dpi=150)
    plt.close()
    print(f"   Saved: balance_imbalance_ratio.png")

    # ── 4. Effective sampling weights ──────────────────────────────────────
    eff = compute_effective_sampling(final_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(CLASS_NAMES, [eff[c] for c in CLASS_NAMES],
           color=sns.color_palette("coolwarm", NUM_CLASSES), edgecolor="black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set(title="Effective WeightedRandomSampler Probabilities",
           ylabel="Sampling Weight (normalised)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "balance_sampler_weights.png"), dpi=150)
    plt.close()
    print(f"   Saved: balance_sampler_weights.png")

    # ── 5. Sample augmented images ─────────────────────────────────────────
    aug_df = final_df[final_df["is_augmented"] == True].sample(
        min(20, int(final_df["is_augmented"].sum())), random_state=SEED)
    if len(aug_df) > 0:
        fig, axes = plt.subplots(4, 5, figsize=(18, 15))
        for ax, (_, row) in zip(axes.flat, aug_df.iterrows()):
            img_path = os.path.join(AUG_IMAGES_DIR, row["image_id"])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{row['label']}\n(aug)", fontsize=7)
            ax.axis("off")
        plt.suptitle("Sample Augmented Images", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "balance_aug_samples.png"), dpi=150)
        plt.close()
        print(f"   Saved: balance_aug_samples.png")

    # ── 6. Pie charts ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, key, title in zip(axes,
                              ["class_before", "class_after"],
                              ["Before Balancing", "After Balancing"]):
        vals   = [report[key][c] for c in CLASS_NAMES]
        labels = [f"{c}\n({v})" for c, v in zip(CLASS_NAMES, vals)]
        ax.pie(vals, labels=labels, autopct="%1.1f%%",
               colors=sns.color_palette("tab10", NUM_CLASSES),
               startangle=90, textprops={"fontsize": 7})
        ax.set_title(title, fontsize=12)
    plt.suptitle("Class Proportion — Before vs After Balancing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "balance_pie.png"), dpi=150)
    plt.close()
    print(f"   Saved: balance_pie.png")

    print(f"\n[4] All plots saved to {PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    report, final_df = balance_dataset()
    plot_balance_results(report, final_df)
    print("\n✓ Dataset balancing complete.\n")
