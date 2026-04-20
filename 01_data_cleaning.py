"""
01_data_cleaning.py — AgroDetect Data Cleaning Pipeline
========================================================
Checks:
  1. Image readability / corruption
  2. Image mode (forces RGB)
  3. Perceptual-hash duplicate detection
  4. Minimum size threshold
  5. Aspect-ratio sanity check
Outputs:
  • Dataset/cleaned_train.csv      — cleaned index
  • outputs/plots/cleaning_*.png   — visualisations
  • outputs/results/cleaning_report.json

Run:
  python 01_data_cleaning.py
"""

import os, sys, json, hashlib, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import *

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def md5_hash(path: str) -> str:
    """MD5 hash of raw file bytes — fast duplicate check."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def phash(img: Image.Image, hash_size: int = 16) -> str:
    """Perceptual hash for near-duplicate detection."""
    img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    mean = arr.mean()
    bits = (arr > mean).flatten()
    return "".join(["1" if b else "0" for b in bits])


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ═══════════════════════════════════════════════════════════════════════════
# Main Cleaning
# ═══════════════════════════════════════════════════════════════════════════
def clean_dataset(
    csv_path: str = TRAIN_CSV,
    images_dir: str = TRAIN_IMAGES,
    out_csv: str = CLEANED_CSV,
    min_size: int = 32,
    phash_threshold: int = 8,   # Hamming distance ≤ this → near-duplicate
) -> dict:

    print("\n" + "=" * 60)
    print("  AgroDetect — Data Cleaning")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"\n[1] Loaded {len(df):,} rows from {csv_path}")

    report = {
        "total_original":       len(df),
        "corrupted":            [],
        "wrong_mode":           [],
        "too_small":            [],
        "bad_aspect_ratio":     [],
        "exact_duplicates":     [],
        "near_duplicates":      [],
        "kept":                 0,
        "dropped":              0,
        "class_before":         {},
        "class_after":          {},
        "image_sizes":          [],
    }

    # ── Per-class count before ─────────────────────────────────────────────
    for cls in CLASS_NAMES:
        report["class_before"][cls] = int((df["label"] == cls).sum())

    # ── Scan every image ───────────────────────────────────────────────────
    print("\n[2] Scanning images …")
    t0 = time.time()

    md5_seen   = {}    # md5 → first image_id
    phash_seen = []    # list of (phash_str, image_id)

    bad_rows = set()

    widths, heights = [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        img_id   = row["image_id"]
        label    = row["label"]

        # Support both layouts:
        #   train_images/<label>/<image_id>   ← class-subfolder layout
        #   train_images/<image_id>           ← flat layout
        img_path_sub  = os.path.join(images_dir, label, img_id)
        img_path_flat = os.path.join(images_dir, img_id)
        if os.path.exists(img_path_sub):
            img_path = img_path_sub
        elif os.path.exists(img_path_flat):
            img_path = img_path_flat
        else:
            img_path = img_path_sub   # will fail existence check below

        # ── existence ──────────────────────────────────────────────────────
        if not os.path.exists(img_path):
            report["corrupted"].append(img_id)
            bad_rows.add(idx)
            continue

        # ── readability ────────────────────────────────────────────────────
        try:
            img = Image.open(img_path)
            img.verify()           # catches truncated / corrupt files
            img = Image.open(img_path)  # re-open after verify
        except Exception:
            report["corrupted"].append(img_id)
            bad_rows.add(idx)
            continue

        # ── mode ───────────────────────────────────────────────────────────
        if img.mode not in ("RGB", "RGBA", "L"):
            report["wrong_mode"].append(img_id)
            # Convertible — we keep but note it
        img = img.convert("RGB")

        w, h = img.size
        widths.append(w); heights.append(h)
        report["image_sizes"].append([w, h])

        # ── minimum size ───────────────────────────────────────────────────
        if w < min_size or h < min_size:
            report["too_small"].append(img_id)
            bad_rows.add(idx)
            continue

        # ── aspect ratio (extreme landscape / portrait > 10:1) ─────────────
        ar = max(w, h) / (min(w, h) + 1e-6)
        if ar > 10:
            report["bad_aspect_ratio"].append(img_id)
            bad_rows.add(idx)
            continue

        # ── exact duplicate (MD5) ──────────────────────────────────────────
        h_md5 = md5_hash(img_path)
        if h_md5 in md5_seen:
            report["exact_duplicates"].append(img_id)
            bad_rows.add(idx)
            continue
        md5_seen[h_md5] = img_id

        # ── near-duplicate (pHash) ─────────────────────────────────────────
        ph = phash(img)
        is_near_dup = False
        for prev_ph, prev_id in phash_seen:
            if hamming_distance(ph, prev_ph) <= phash_threshold:
                report["near_duplicates"].append(img_id)
                bad_rows.add(idx)
                is_near_dup = True
                break
        if not is_near_dup:
            phash_seen.append((ph, img_id))

    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s")

    # ── Build cleaned DataFrame ────────────────────────────────────────────
    clean_df = df.drop(index=list(bad_rows)).reset_index(drop=True)
    clean_df.to_csv(out_csv, index=False)

    report["kept"]    = len(clean_df)
    report["dropped"] = len(df) - len(clean_df)

    for cls in CLASS_NAMES:
        report["class_after"][cls] = int((clean_df["label"] == cls).sum())

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n[3] Cleaning Summary")
    print(f"    Corrupted / missing : {len(report['corrupted'])}")
    print(f"    Wrong mode          : {len(report['wrong_mode'])}")
    print(f"    Too small           : {len(report['too_small'])}")
    print(f"    Bad aspect ratio    : {len(report['bad_aspect_ratio'])}")
    print(f"    Exact duplicates    : {len(report['exact_duplicates'])}")
    print(f"    Near-duplicates     : {len(report['near_duplicates'])}")
    print(f"    ─────────────────────────────────")
    print(f"    Kept  : {report['kept']:,}  /  Dropped : {report['dropped']:,}")
    print(f"    Cleaned CSV → {out_csv}")

    # ── Save JSON report ───────────────────────────────────────────────────
    rep_path = os.path.join(RESULTS_DIR, "cleaning_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Report JSON → {rep_path}")

    return report, clean_df, widths, heights


# ═══════════════════════════════════════════════════════════════════════════
# Visualisations
# ═══════════════════════════════════════════════════════════════════════════
def plot_cleaning_results(report: dict, widths: list, heights: list):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", 10)

    # ── 1. Class distribution before vs after ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, title in zip(axes,
                              ["class_before", "class_after"],
                              ["Before Cleaning", "After Cleaning"]):
        vals = [report[key].get(c, 0) for c in CLASS_NAMES]
        bars = ax.barh(CLASS_NAMES, vals, color=palette)
        ax.set(title=title, xlabel="Image Count")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=9)
    plt.suptitle("Class Distribution — Before vs After Cleaning", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cleaning_class_dist.png"), dpi=150)
    plt.close()

    # ── 2. Issues breakdown pie ────────────────────────────────────────────
    issue_labels = ["Corrupted", "Wrong Mode", "Too Small", "Bad Aspect",
                    "Exact Dup.", "Near Dup."]
    issue_sizes  = [
        len(report["corrupted"]),
        len(report["wrong_mode"]),
        len(report["too_small"]),
        len(report["bad_aspect_ratio"]),
        len(report["exact_duplicates"]),
        len(report["near_duplicates"]),
    ]
    total_issues = sum(issue_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if total_issues > 0:
        non_zero = [(l, s) for l, s in zip(issue_labels, issue_sizes) if s > 0]
        lbl, sz  = zip(*non_zero)
        axes[0].pie(sz, labels=lbl, autopct="%1.1f%%",
                    colors=sns.color_palette("Set3", len(lbl)))
        axes[0].set_title("Issue Breakdown")
    else:
        axes[0].text(0.5, 0.5, "No issues found!", ha="center", va="center",
                     fontsize=14, transform=axes[0].transAxes)
        axes[0].set_title("Issue Breakdown")

    # Kept vs Dropped bar
    axes[1].bar(["Kept", "Dropped"],
                [report["kept"], report["dropped"]],
                color=["#4CAF50", "#F44336"], edgecolor="black")
    axes[1].set(title="Kept vs Dropped Images", ylabel="Count")
    for i, v in enumerate([report["kept"], report["dropped"]]):
        axes[1].text(i, v + 20, str(v), ha="center", fontweight="bold")

    plt.suptitle("Data Cleaning Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cleaning_summary.png"), dpi=150)
    plt.close()

    # ── 3. Image size distribution ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(widths,  bins=30, color="steelblue",  edgecolor="black", alpha=0.7)
    axes[0].set(title="Width Distribution", xlabel="Pixels", ylabel="Count")
    axes[1].hist(heights, bins=30, color="darkorange", edgecolor="black", alpha=0.7)
    axes[1].set(title="Height Distribution", xlabel="Pixels")
    axes[2].scatter(widths, heights, alpha=0.3, s=5, color="purple")
    axes[2].set(title="Width vs Height", xlabel="Width", ylabel="Height")
    plt.suptitle("Image Size Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cleaning_image_sizes.png"), dpi=150)
    plt.close()

    # ── 4. Sample cleaned images ───────────────────────────────────────────
    clean_df = pd.read_csv(CLEANED_CSV)
    n_show = min(20, len(clean_df))
    sample  = clean_df.sample(n_show, random_state=SEED)
    fig, axes = plt.subplots(4, 5, figsize=(18, 15))
    for ax, (_, row) in zip(axes.flat, sample.iterrows()):
        p = os.path.join(TRAIN_IMAGES, row["label"], row["image_id"])
        if not os.path.exists(p):
            p = os.path.join(TRAIN_IMAGES, row["image_id"])
        img = Image.open(p).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{row['label']}\n{row['image_id']}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Sample Cleaned Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cleaning_samples.png"), dpi=150)
    plt.close()

    print(f"\n[4] Plots saved to {PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    report, clean_df, widths, heights = clean_dataset()
    plot_cleaning_results(report, widths, heights)
    print("\n✓ Data cleaning complete.\n")
