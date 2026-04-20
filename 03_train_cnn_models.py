"""
03_train_cnn_models.py — Train All 6 CNN Baseline Models
==========================================================
Models: ConvNeXt_base · DenseNet161 · GoogLeNet
        MobileNet_v3_large · ResNet50 · ShuffleNet_v2

Per model, reports:
  Train / Val / Test accuracy, class-wise accuracies, F1 (macro+weighted),
  confusion matrix, classification report, accuracy+loss curves,
  inference time, training time, 95% confidence intervals,
  5-fold cross-validation, memory profiling, Grad-CAM heatmaps,
  SHAP plots (if shap installed).

Best model checkpoint saved to outputs/models/<name>_best.pth
All metrics saved to outputs/results/<name>.json

Run:
  python 03_train_cnn_models.py
  (run 01 and 02 first)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import (
    set_seed, get_model, get_feature_extractor,
    PlantDiseaseDataset, make_dataloaders, split_dataframe,
    full_train, compute_full_metrics, cross_validate_model,
    profile_memory, compute_and_save_shap, grad_cam_heatmap,
    plot_training_curves, plot_confusion_matrix, plot_class_accuracies,
    save_results, load_results,
    get_train_transform, get_val_transform, compute_class_weights,
    train_one_epoch, evaluate, EarlyStopping,
    DEVICE
)

# ═══════════════════════════════════════════════════════════════════════════
# Data preparation helper — resolves images directory for augmented images
# ═══════════════════════════════════════════════════════════════════════════
class MultiDirDataset(PlantDiseaseDataset):
    """Dataset that looks up images in multiple directories.
    Supports both class-subfolder layout (train_images/<label>/<file>)
    and flat layout (aug_images/<file>).
    """
    def __init__(self, df, dirs: list, transform=None):
        super().__init__(df, dirs[0], transform)
        self.dirs = dirs

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        row = self.df.iloc[idx]
        label = row.get("label", "")
        for d in self.dirs:
            # 1) class-subfolder layout
            p = os.path.join(d, label, row["image_id"])
            if os.path.exists(p):
                img = PILImage.open(p).convert("RGB")
                if self.transform: img = self.transform(img)
                return img, self.class_to_idx[label]
            # 2) flat layout (augmented images)
            p = os.path.join(d, row["image_id"])
            if os.path.exists(p):
                img = PILImage.open(p).convert("RGB")
                if self.transform: img = self.transform(img)
                return img, self.class_to_idx[label]
        raise FileNotFoundError(f"Image not found: {row['image_id']}")


def prepare_data(use_balanced: bool = True):
    """Return train / val / test DataFrames and image directories."""
    csv = BALANCED_CSV if (use_balanced and os.path.exists(BALANCED_CSV)) else CLEANED_CSV
    if not os.path.exists(csv):
        csv = TRAIN_CSV
    df = pd.read_csv(csv)

    # Drop rows where image file doesn't exist in any dir
    aug_dir = os.path.join(DATASET_DIR, "aug_images")
    dirs = [TRAIN_IMAGES]
    if os.path.isdir(aug_dir):
        dirs.append(aug_dir)

    def exists(row):
        label = row.get("label", "")
        return any(
            os.path.exists(os.path.join(d, label, row["image_id"])) or
            os.path.exists(os.path.join(d, row["image_id"]))
            for d in dirs
        )

    df = df[df.apply(exists, axis=1)].reset_index(drop=True)
    print(f"  [Data] {len(df):,} valid images from {csv}")

    train_df, val_df, test_df = split_dataframe(df)
    return train_df, val_df, test_df, dirs


def make_loaders_multi(train_df, val_df, test_df, dirs):
    from torch.utils.data import DataLoader
    from utils import get_train_transform, get_val_transform
    from torch.utils.data import WeightedRandomSampler

    aug_dir = os.path.join(DATASET_DIR, "aug_images")
    train_ds = MultiDirDataset(train_df, dirs, get_train_transform())
    val_ds   = MultiDirDataset(val_df,   dirs, get_val_transform())
    test_ds  = MultiDirDataset(test_df,  dirs, get_val_transform())

    # Weighted sampler on training set
    targets = np.array(train_ds.targets)
    counts  = np.bincount(targets, minlength=NUM_CLASSES)
    sw      = 1.0 / (counts + 1e-6)
    sw      = sw[targets]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ld   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_ld, val_ld, test_ld, train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
# Full training routine (overrides utils.full_train for multi-dir support)
# ═══════════════════════════════════════════════════════════════════════════
def train_model(model_name: str, train_df, val_df, test_df, dirs,
                num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE,
                tag: str = "") -> dict:
    set_seed()
    print(f"\n{'─'*60}")
    print(f"  Training: {model_name}  (tag={tag or 'baseline'})")
    print(f"{'─'*60}")

    train_ld, val_ld, test_ld, train_ds, val_ds, test_ds = make_loaders_multi(
        train_df, val_df, test_df, dirs)

    model = get_model(model_name).to(DEVICE)

    cw = compute_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    criterion_plain = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=ETA_MIN)
    scaler    = GradScaler(device="cuda", enabled=USE_AMP)
    es        = EarlyStopping(patience=PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc, best_epoch = 0.0, 0

    save_tag  = f"{model_name}{'_' + tag if tag else ''}"
    best_path = os.path.join(MODELS_DIR, f"{save_tag}_best.pth")

    # ── Training loop ──────────────────────────────────────────────────────
    train_start = time.time()
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, criterion, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_ld, criterion_plain)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        if va_acc > best_val_acc:
            best_val_acc = va_acc; best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": va_acc, "model_name": model_name}, best_path)

        print(f"  [{epoch:3d}/{num_epochs}] "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc:.4f}/{va_acc:.4f}  lr {lr_now:.2e}")

        if es(va_acc):
            print(f"  Early stopping at epoch {epoch}.")
            break

    training_time = time.time() - train_start

    # ── Load best weights ──────────────────────────────────────────────────
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # ── Test evaluation ────────────────────────────────────────────────────
    ts_loss, ts_acc, preds, labels = evaluate(model, test_ld, criterion_plain)

    # ── Inference time ─────────────────────────────────────────────────────
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for imgs, _ in test_ld:
            _ = model(imgs.to(DEVICE))
    inf_time = (time.time() - t0) / len(test_ld.dataset)

    # ── Results dict ──────────────────────────────────────────────────────
    results = {
        "model_name":    model_name,
        "tag":           tag,
        "best_epoch":    best_epoch,
        "num_epochs":    num_epochs,
        "batch_size":    BATCH_SIZE,
        "learning_rate": lr,
        "dropout":       DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "train_acc":     history["train_acc"][best_epoch - 1],
        "val_acc":       best_val_acc,
        "test_acc":      ts_acc,
        "test_loss":     ts_loss,
        "training_time_s": training_time,
        "inference_time_per_img_s": inf_time,
        "history":       history,
        "preds":         preds.tolist(),
        "labels":        labels.tolist(),
    }
    results.update(compute_full_metrics(preds, labels, model_name))

    # ── Memory profiling ───────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(DEVICE)
    mem = profile_memory(model, test_ld)
    results["memory"] = mem

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_training_curves(history, model_name, tag=tag)
    plot_confusion_matrix(np.array(results["confusion_matrix"]), model_name, tag=tag)
    plot_class_accuracies(results["per_class_acc"], model_name, tag=tag)

    # Grad-CAM
    try:
        grad_cam_heatmap(model, model_name, test_ds)
    except Exception as e:
        print(f"  [Grad-CAM] skipped: {e}")

    # SHAP
    try:
        compute_and_save_shap(model, test_ds, model_name)
    except Exception as e:
        print(f"  [SHAP] skipped: {e}")

    # ── Save results ──────────────────────────────────────────────────────
    save_results(results, save_tag)

    print(f"\n  ✓ {model_name} | test_acc={ts_acc:.4f} | "
          f"f1_macro={results['f1_macro']:.4f} | "
          f"train_time={training_time:.0f}s")

    return results, model


# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation (lightweight version)
# ═══════════════════════════════════════════════════════════════════════════
def run_cross_validation(model_name: str, df: pd.DataFrame, dirs: list) -> dict:
    """Run 5-fold CV using only original (non-augmented) images."""
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler

    print(f"\n  [CV] {model_name} — {CV_FOLDS}-fold cross-validation …")
    # Use only original images for CV to avoid data leakage
    orig_df = df[df.get("is_augmented", pd.Series([False]*len(df))) != True].reset_index(drop=True)
    if len(orig_df) == 0:
        orig_df = df

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    labels = orig_df["label"].values
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(orig_df, labels), 1):
        tr_df = orig_df.iloc[tr_idx]
        va_df = orig_df.iloc[va_idx]

        tr_ds = MultiDirDataset(tr_df, dirs, get_train_transform())
        va_ds = MultiDirDataset(va_df, dirs, get_val_transform())
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model     = get_model(model_name).to(DEVICE)
        cw        = compute_class_weights(tr_df).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        scaler    = GradScaler(device="cuda", enabled=USE_AMP)

        for ep in range(CV_FOLD_EPOCHS):
            train_one_epoch(model, tr_ld, optimizer, criterion, scaler)

        from sklearn.metrics import f1_score as sk_f1
        _, acc, p, l = evaluate(model, va_ld, criterion)
        f1 = sk_f1(l, p, average="macro", zero_division=0)
        fold_accs.append(acc); fold_f1s.append(f1)
        print(f"     Fold {fold}: acc={acc:.4f}  f1={f1:.4f}")
        del model; torch.cuda.empty_cache()

    cv_results = {
        "cv_acc_mean": float(np.mean(fold_accs)),
        "cv_acc_std":  float(np.std(fold_accs)),
        "cv_f1_mean":  float(np.mean(fold_f1s)),
        "cv_f1_std":   float(np.std(fold_f1s)),
        "cv_fold_accs": [float(x) for x in fold_accs],
        "cv_fold_f1s":  [float(x) for x in fold_f1s],
    }
    print(f"  [CV] {model_name}: acc={cv_results['cv_acc_mean']:.4f}±{cv_results['cv_acc_std']:.4f}")
    return cv_results


# ═══════════════════════════════════════════════════════════════════════════
# Summary plots across all models
# ═══════════════════════════════════════════════════════════════════════════
def plot_all_model_comparison(all_results: list):
    sns.set_style("whitegrid")
    names = [r["model_name"] for r in all_results]

    # ── 1. Test accuracy comparison ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("test_acc",       "Test Accuracy"),
        ("val_acc",        "Val Accuracy"),
        ("f1_macro",       "F1 Macro"),
        ("f1_weighted",    "F1 Weighted"),
        ("precision_macro","Precision Macro"),
        ("recall_macro",   "Recall Macro"),
    ]
    palette = sns.color_palette("tab10", len(names))
    for ax, (key, title) in zip(axes.flat, metrics):
        vals = [r.get(key, 0) for r in all_results]
        bars = ax.bar(names, vals, color=palette, edgecolor="black", linewidth=0.5)
        ax.set(title=title, ylim=(0, 1.05), ylabel="Score")
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Baseline CNN Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_comparison_metrics.png"), dpi=150)
    plt.close()

    # ── 2. Training time vs accuracy ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [r.get("training_time_s", 0) / 60 for r in all_results]
    accs  = [r.get("test_acc", 0) for r in all_results]
    for i, (n, t, a) in enumerate(zip(names, times, accs)):
        ax.scatter(t, a, s=150, color=palette[i], zorder=5, label=n)
        ax.annotate(n, (t, a), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set(title="Training Time vs Test Accuracy",
           xlabel="Training Time (min)", ylabel="Test Accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_time_vs_acc.png"), dpi=150)
    plt.close()

    # ── 3. Inference time comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    inf_times = [r.get("inference_time_per_img_s", 0) * 1000 for r in all_results]
    ax.bar(names, inf_times, color=palette, edgecolor="black")
    ax.set(title="Inference Time per Image (ms)", ylabel="ms")
    ax.set_xticklabels(names, rotation=30, ha="right")
    for i, v in enumerate(inf_times):
        ax.text(i, v + 0.05, f"{v:.2f}ms", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_inference_time.png"), dpi=150)
    plt.close()

    # ── 4. Confidence intervals ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    accs_ci = [r.get("test_acc", 0) for r in all_results]
    lo      = [r.get("ci_lower", 0) for r in all_results]
    hi      = [r.get("ci_upper", 0) for r in all_results]
    lo_err  = [a - l for a, l in zip(accs_ci, lo)]
    hi_err  = [h - a for a, h in zip(accs_ci, hi)]
    x = np.arange(len(names))
    ax.bar(x, accs_ci, color=palette, edgecolor="black", alpha=0.8)
    ax.errorbar(x, accs_ci, yerr=[lo_err, hi_err], fmt="none",
                color="black", capsize=6, linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(title="Test Accuracy with 95% Bootstrap CI", ylabel="Accuracy", ylim=(0, 1.1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_confidence_intervals.png"), dpi=150)
    plt.close()

    # ── 5. All training curves on one plot ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, r in zip(axes.flat, all_results):
        h = r.get("history", {})
        if not h or not h.get("train_acc"):
            ax.axis("off"); continue
        ep = range(1, len(h["train_acc"]) + 1)
        ax.plot(ep, h["train_acc"], label="Train")
        ax.plot(ep, h["val_acc"],   label="Val")
        ax.set(title=r["model_name"], xlabel="Epoch", ylabel="Accuracy",
               ylim=(0, 1.05))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Training Curves — All CNN Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_all_training_curves.png"), dpi=150)
    plt.close()

    # ── 6. Heatmap: per-class accuracy across models ───────────────────────
    pca_matrix = []
    for r in all_results:
        pca_matrix.append([r["per_class_acc"].get(c, 0) for c in CLASS_NAMES])
    pca_matrix = np.array(pca_matrix)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pca_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=CLASS_NAMES, yticklabels=names, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set(title="Per-class Accuracy — All CNN Models",
           xlabel="Class", ylabel="Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_per_class_acc_heatmap.png"), dpi=150)
    plt.close()

    # ── 7. CV results ─────────────────────────────────────────────────────
    cv_accs = [r.get("cv_acc_mean", 0) for r in all_results]
    cv_stds = [r.get("cv_acc_std",  0) for r in all_results]
    if any(v > 0 for v in cv_accs):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, cv_accs, yerr=cv_stds, capsize=6,
               color=palette, edgecolor="black", alpha=0.85)
        ax.set(title=f"{CV_FOLDS}-Fold Cross-Validation Accuracy",
               ylabel="Accuracy ± Std", ylim=(0, 1.1))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "cnn_cross_validation.png"), dpi=150)
        plt.close()

    print(f"\n[Plots] All comparison plots saved to {PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_seed()
    print("\n" + "=" * 60)
    print("  AgroDetect — Train Baseline CNN Models")
    print("=" * 60)

    # ── Data ───────────────────────────────────────────────────────────────
    train_df, val_df, test_df, dirs = prepare_data(use_balanced=True)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── Train all models ───────────────────────────────────────────────────
    all_results = []
    for model_name in MODEL_NAMES:
        save_tag = model_name
        existing = load_results(save_tag)
        if existing and os.path.exists(os.path.join(MODELS_DIR, f"{save_tag}_best.pth")):
            print(f"\n  [Skip] {model_name} already trained. Loading results.")
            # Re-run CV if not present
            if "cv_acc_mean" not in existing:
                all_df = pd.concat([train_df, val_df, test_df])
                cv_r = run_cross_validation(model_name, all_df, dirs)
                existing.update(cv_r)
                save_results(existing, save_tag)
            all_results.append(existing)
            continue

        results, model = train_model(
            model_name, train_df, val_df, test_df, dirs,
            num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
        )

        # Cross-validation
        all_df = pd.concat([train_df, val_df, test_df])
        cv_r = run_cross_validation(model_name, all_df, dirs)
        results.update(cv_r)
        save_results(results, model_name)

        all_results.append(results)
        del model; torch.cuda.empty_cache()

    # ── Cross-model comparison plots ───────────────────────────────────────
    plot_all_model_comparison(all_results)

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  {'Model':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'F1-Macro':>10} {'CV-Acc':>10}")
    print("  " + "-" * 78)
    for r in all_results:
        print(f"  {r['model_name']:<25} "
              f"{r.get('train_acc', 0):>8.4f} "
              f"{r.get('val_acc', 0):>8.4f} "
              f"{r.get('test_acc', 0):>8.4f} "
              f"{r.get('f1_macro', 0):>10.4f} "
              f"{r.get('cv_acc_mean', 0):>10.4f}")
    print("=" * 80)

    print("\n✓ All CNN models trained and evaluated.\n")
