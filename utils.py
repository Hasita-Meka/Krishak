"""
utils.py — Shared utilities for the AgroDetect pipeline.
Covers: Dataset, transforms, model factory, training loop, metrics,
        visualization, memory profiling, SHAP, confidence intervals,
        cross-validation, early stopping.
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import os, time, json, random, copy, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from PIL import Image
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
try:
    from torch.amp import GradScaler, autocast          # PyTorch >= 2.4
except ImportError:
    from torch.cuda.amp import GradScaler, autocast     # PyTorch < 2.4

import tracemalloc, psutil

# ── Project config ─────────────────────────────────────────────────────────
from config import *


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.2),
    ])

def get_val_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════
class PlantDiseaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df          = df.reset_index(drop=True)
        self.images_dir  = images_dir
        self.transform   = transform
        self.class_to_idx = CLASS_TO_IDX

    def __len__(self):
        return len(self.df)

    def _find_image(self, row) -> str:
        """
        Try both layouts:
          1. images_dir / label / image_id   (class-subfolder layout)
          2. images_dir / image_id           (flat layout, e.g. aug_images)
        """
        p1 = os.path.join(self.images_dir, str(row.get("label", "")), row["image_id"])
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(self.images_dir, row["image_id"])
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(
            f"Image not found in either layout:\n  {p1}\n  {p2}")

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self._find_image(row)
        img      = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label

    @property
    def targets(self):
        return [self.class_to_idx[l] for l in self.df["label"]]


def make_dataloaders(
    train_df, val_df, test_df, images_dir: str,
    batch_size: int = BATCH_SIZE, use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = PlantDiseaseDataset(train_df, images_dir, get_train_transform())
    val_ds   = PlantDiseaseDataset(val_df,   images_dir, get_val_transform())
    test_ds  = PlantDiseaseDataset(test_df,  images_dir, get_val_transform())

    if use_weighted_sampler:
        targets = np.array(train_ds.targets)
        class_counts = np.bincount(targets, minlength=NUM_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, val_loader, test_loader


def split_dataframe(df: pd.DataFrame, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    from sklearn.model_selection import train_test_split
    train_val, test_df = train_test_split(
        df, test_size=test_split, stratify=df["label"], random_state=seed)
    val_ratio = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════
def get_model(name: str, num_classes: int = NUM_CLASSES,
              dropout: float = DROPOUT, pretrained: bool = True) -> nn.Module:
    """Return a torchvision model with modified classifier head."""

    if name == "convnext_base":
        w = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        m = models.convnext_base(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name == "densenet161":
        w = models.DenseNet161_Weights.DEFAULT if pretrained else None
        m = models.densenet161(weights=w)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name == "googlenet":
        w = models.GoogLeNet_Weights.DEFAULT if pretrained else None
        m = models.googlenet(weights=w, aux_logits=True)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        if m.aux1 is not None:
            m.aux1.fc2 = nn.Linear(1024, num_classes)
        if m.aux2 is not None:
            m.aux2.fc2 = nn.Linear(1024, num_classes)

    elif name == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    elif name == "shufflenet_v2":
        w = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        m = models.shufflenet_v2_x1_0(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    else:
        raise ValueError(f"Unknown model: {name}")

    return m


def get_feature_extractor(model: nn.Module, name: str) -> nn.Module:
    """Return model with identity final layer for feature extraction."""
    m = copy.deepcopy(model)
    if name == "convnext_base":
        m.classifier[2] = nn.Identity()
    elif name == "densenet161":
        m.classifier = nn.Identity()
    elif name == "googlenet":
        m.fc = nn.Identity()
    elif name == "mobilenet_v3_large":
        m.classifier[3] = nn.Identity()
    elif name == "resnet50":
        m.fc = nn.Identity()
    elif name == "shufflenet_v2":
        m.fc = nn.Identity()
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Class weights for loss
# ═══════════════════════════════════════════════════════════════════════════
def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["label"].value_counts()
    weights = []
    for cls in CLASS_NAMES:
        n = counts.get(cls, 1)
        weights.append(1.0 / n)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum() * NUM_CLASSES
    return w


# ═══════════════════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA,
                 mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = -np.inf if mode == "max" else np.inf
        self.counter    = 0
        self.stop       = False

    def __call__(self, metric: float) -> bool:
        improved = (self.mode == "max" and metric > self.best + self.min_delta) or \
                   (self.mode == "min" and metric < self.best - self.min_delta)
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ═══════════════════════════════════════════════════════════════════════════
# Training / Evaluation loops
# ═══════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device=DEVICE) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            out = model(imgs)
            # GoogLeNet returns (logits, aux1, aux2) during training
            if isinstance(out, (tuple, list)):
                loss = criterion(out[0], labels)
                for aux in out[1:]:
                    if aux is not None:
                        loss += 0.3 * criterion(aux, labels)
                out = out[0]
            else:
                loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type="cuda", enabled=USE_AMP):
            out = model(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ═══════════════════════════════════════════════════════════════════════════
# Full training loop with all bells and whistles
# ═══════════════════════════════════════════════════════════════════════════
def full_train(
    model_name: str,
    train_loader, val_loader, test_loader,
    train_df: pd.DataFrame,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    save_dir: str = MODELS_DIR,
    tag: str = "",
) -> Dict:
    set_seed()
    model = get_model(model_name).to(DEVICE)

    # Class-weighted loss + label smoothing
    cw = compute_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=ETA_MIN)
    scaler    = GradScaler(device="cuda", enabled=USE_AMP)
    es        = EarlyStopping(patience=PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc, best_epoch = 0.0, 0

    save_tag  = f"{model_name}{'_' + tag if tag else ''}"
    best_path = os.path.join(save_dir, f"{save_tag}_best.pth")

    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)

        if va_acc > best_val_acc:
            best_val_acc  = va_acc
            best_epoch    = epoch
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_acc": va_acc,
                        "model_name": model_name}, best_path)

        print(f"  [{epoch:3d}/{num_epochs}] "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc:.4f}/{va_acc:.4f}  lr {current_lr:.2e}")

        if es(va_acc):
            print(f"  Early stopping at epoch {epoch}.")
            break

    training_time = time.time() - train_start

    # Load best weights and evaluate on test set
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    criterion_test = nn.CrossEntropyLoss()
    ts_loss, ts_acc, preds, labels = evaluate(model, test_loader, criterion_test)

    # Inference time (mean per image)
    inf_start = time.time()
    with torch.no_grad():
        for imgs, _ in test_loader:
            _ = model(imgs.to(DEVICE))
    inf_time = (time.time() - inf_start) / len(test_loader.dataset)

    results = {
        "model_name":    model_name,
        "tag":           tag,
        "best_epoch":    best_epoch,
        "train_acc":     history["train_acc"][best_epoch - 1],
        "val_acc":       best_val_acc,
        "test_acc":      ts_acc,
        "test_loss":     ts_loss,
        "training_time": training_time,
        "inference_time_per_img": inf_time,
        "history":       history,
        "preds":         preds.tolist(),
        "labels":        labels.tolist(),
    }

    # Detailed metrics
    results.update(compute_full_metrics(preds, labels, model_name))
    return results, model


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_full_metrics(preds: np.ndarray, labels: np.ndarray, name: str = "") -> Dict:
    cm   = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    f1_macro   = f1_score(labels, preds, average="macro",   zero_division=0)
    f1_weighted= f1_score(labels, preds, average="weighted", zero_division=0)
    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    rec_macro  = recall_score(labels,  preds, average="macro", zero_division=0)

    ci_lower, ci_upper = bootstrap_confidence_interval(labels, preds)

    return {
        "confusion_matrix":   cm.tolist(),
        "per_class_acc":      {c: float(a) for c, a in zip(CLASS_NAMES, per_class_acc)},
        "f1_macro":           f1_macro,
        "f1_weighted":        f1_weighted,
        "precision_macro":    prec_macro,
        "recall_macro":       rec_macro,
        "classification_report": report,
        "ci_lower":           ci_lower,
        "ci_upper":           ci_upper,
    }


def bootstrap_confidence_interval(labels, preds, n_boot=1000, alpha=0.95) -> Tuple[float, float]:
    accs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        accs.append(accuracy_score(labels[idx], preds[idx]))
    lo = np.percentile(accs, (1 - alpha) / 2 * 100)
    hi = np.percentile(accs, (1 + alpha) / 2 * 100)
    return float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════════════════
# Memory Profiling
# ═══════════════════════════════════════════════════════════════════════════
def profile_memory(model: nn.Module, loader: DataLoader) -> Dict:
    tracemalloc.start()
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1e6

    gpu_before = torch.cuda.memory_allocated(DEVICE) / 1e6 if torch.cuda.is_available() else 0

    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            _ = model(imgs.to(DEVICE))
            break   # one batch is enough

    ram_after = proc.memory_info().rss / 1e6
    gpu_after = torch.cuda.memory_allocated(DEVICE) / 1e6 if torch.cuda.is_available() else 0
    peak_gpu  = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if torch.cuda.is_available() else 0

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "ram_delta_mb":  ram_after - ram_before,
        "peak_ram_mb":   peak_ram / 1e6,
        "gpu_delta_mb":  gpu_after - gpu_before,
        "peak_gpu_mb":   peak_gpu,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cross Validation (feature-level for speed)
# ═══════════════════════════════════════════════════════════════════════════
def cross_validate_model(model_name: str, df: pd.DataFrame, images_dir: str,
                         n_folds: int = CV_FOLDS) -> Dict:
    """K-fold CV — trains lightweight version for speed."""
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    labels = df["label"].values
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, labels), 1):
        tr_df = df.iloc[tr_idx]; va_df = df.iloc[va_idx]
        tr_ds = PlantDiseaseDataset(tr_df, images_dir, get_train_transform())
        va_ds = PlantDiseaseDataset(va_df, images_dir, get_val_transform())
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

        for epoch in range(CV_FOLD_EPOCHS):   # brief training per fold
            train_one_epoch(model, tr_ld, optimizer, criterion, scaler)

        _, acc, preds, lbls = evaluate(model, va_ld, criterion)
        f1 = f1_score(lbls, preds, average="macro", zero_division=0)
        fold_accs.append(acc); fold_f1s.append(f1)
        print(f"  Fold {fold}/{n_folds}: acc={acc:.4f}, f1={f1:.4f}")

        del model; torch.cuda.empty_cache()

    return {
        "cv_acc_mean": float(np.mean(fold_accs)),
        "cv_acc_std":  float(np.std(fold_accs)),
        "cv_f1_mean":  float(np.mean(fold_f1s)),
        "cv_f1_std":   float(np.std(fold_f1s)),
        "cv_fold_accs": fold_accs,
        "cv_fold_f1s":  fold_f1s,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════
def compute_and_save_shap(model: nn.Module, dataset: PlantDiseaseDataset,
                          model_name: str, save_dir: str = PLOTS_DIR):
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — skipping.")
        return

    model.eval()
    # Background and test samples (kept small for speed)
    bg_idx  = np.random.choice(len(dataset), min(SHAP_BACKGROUND, len(dataset)), replace=False)
    bg_imgs = torch.stack([dataset[i][0] for i in bg_idx]).to(DEVICE)
    te_idx  = np.random.choice(len(dataset), min(SHAP_TEST_SAMPLES, len(dataset)), replace=False)
    te_imgs = torch.stack([dataset[i][0] for i in te_idx]).to(DEVICE)

    # Wrap model to always return plain tensor (handles GoogLeNet tuple output)
    class _SafeWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, (tuple, list)) else out

    safe_model = _SafeWrapper(model)

    try:
        # GradientExplainer is more stable than DeepExplainer on PyTorch ≥ 2.x
        explainer = shap.GradientExplainer(safe_model, bg_imgs)
        shap_vals = explainer.shap_values(te_imgs)   # list[C] of (N,C,H,W)
    except Exception as ex:
        print(f"  [SHAP] GradientExplainer failed ({ex}); trying DeepExplainer …")
        try:
            explainer = shap.DeepExplainer(safe_model, bg_imgs)
            shap_vals = explainer.shap_values(te_imgs)
        except Exception as ex2:
            print(f"  [SHAP] DeepExplainer also failed ({ex2}); skipping SHAP.")
            return

    # Mean absolute SHAP per class across spatial + channel dims → (num_classes,)
    fig, ax = plt.subplots(figsize=(12, 6))
    if isinstance(shap_vals, list):
        # shap_vals: list of (N, C, H, W)  — one entry per output class
        mean_abs = np.array([np.abs(sv).mean(axis=(0, 2, 3)) for sv in shap_vals])
        # → (num_classes, C)  — heatmap: classes × input channels
        col_labels = [f"Ch{i}" for i in range(mean_abs.shape[1])]
    else:
        # shap_vals: (N, num_classes, C, H, W)  — newer shap API
        mean_abs = np.abs(shap_vals).mean(axis=(0, 2, 3))  # (num_classes, C)
        col_labels = [f"Ch{i}" for i in range(mean_abs.shape[1])]

    sns.heatmap(mean_abs, xticklabels=col_labels[:50],   # cap x-tick labels
                yticklabels=CLASS_NAMES, ax=ax, cmap="viridis")
    ax.set_title(f"SHAP Feature Importance — {model_name}")
    ax.set_xlabel("Input channels (mean over spatial dims)")
    plt.tight_layout()
    path = os.path.join(save_dir, f"shap_{model_name}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [SHAP] saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Grad-CAM feature importance heatmap (fallback to GradCAM)
# ═══════════════════════════════════════════════════════════════════════════
def grad_cam_heatmap(model: nn.Module, model_name: str,
                     dataset: PlantDiseaseDataset, save_dir: str = PLOTS_DIR,
                     n_samples: int = 3):
    """Generate Grad-CAM heatmaps for sample images."""
    model.eval()
    # Pick target conv layer
    _layer_map = {
        "convnext_base":      "features.7.2",
        "densenet161":        "features.denseblock4",
        "googlenet":          "inception5b",
        "mobilenet_v3_large": "features.16",
        "resnet50":           "layer4",
        "shufflenet_v2":      "conv5",
    }
    target_layer_name = _layer_map.get(model_name, None)
    if target_layer_name is None:
        return

    activation, gradient = {}, {}

    def forward_hook(m, inp, out):
        # Clone to avoid inplace-view conflicts with autograd
        activation["feat"] = out.clone().detach()

    def backward_hook(m, grad_in, grad_out):
        g = grad_out[0]
        gradient["feat"] = g.clone().detach() if g is not None else None

    # Traverse to target layer
    layer = dict(model.named_modules()).get(target_layer_name, None)
    if layer is None:
        return
    fh = layer.register_forward_hook(forward_hook)
    # Use register_backward_hook for broader compatibility
    bh = layer.register_backward_hook(backward_hook)

    inv_norm = T.Compose([
        T.Normalize(mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
                    std=[1/s for s in IMAGENET_STD]),
    ])

    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, idx in enumerate(idxs):
        try:
            img_t, label = dataset[idx]
            inp = img_t.unsqueeze(0).to(DEVICE)
            inp.requires_grad_(True)
            activation.clear(); gradient.clear()

            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out[0, label].backward()

            if "feat" not in gradient or gradient["feat"] is None:
                axes[i][0].axis("off"); axes[i][1].axis("off")
                continue

            weights = gradient["feat"].mean(dim=[2, 3], keepdim=True)
            cam     = (weights * activation["feat"]).sum(dim=1, keepdim=True)
            cam     = F.relu(cam).squeeze().cpu().numpy()
            cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            orig = inv_norm(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[i][0].imshow(orig); axes[i][0].set_title(f"Original: {CLASS_NAMES[label]}")
            axes[i][1].imshow(orig); axes[i][1].imshow(cam, alpha=0.5, cmap="jet")
            axes[i][1].set_title("Grad-CAM")
            for ax in axes[i]: ax.axis("off")
        except Exception as _e:
            axes[i][0].axis("off"); axes[i][1].axis("off")

    fh.remove(); bh.remove()
    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"gradcam_{model_name}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [Grad-CAM] saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_curves(history: Dict, model_name: str, save_dir: str = PLOTS_DIR, tag: str = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set(title=f"{model_name} — Loss", xlabel="Epoch", ylabel="Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set(title=f"{model_name} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"curves_{model_name}{'_' + tag if tag else ''}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150); plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                          save_dir: str = PLOTS_DIR, tag: str = ""):
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set(title=f"Confusion Matrix — {model_name}", xlabel="Predicted", ylabel="True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"cm_{model_name}{'_' + tag if tag else ''}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150); plt.close()


def plot_class_accuracies(per_class_acc: Dict, model_name: str,
                          save_dir: str = PLOTS_DIR, tag: str = ""):
    fig, ax = plt.subplots(figsize=(12, 5))
    classes = list(per_class_acc.keys())
    accs    = list(per_class_acc.values())
    bars = ax.barh(classes, accs, color=sns.color_palette("viridis", len(classes)))
    ax.set(title=f"Per-class Accuracy — {model_name}", xlabel="Accuracy", xlim=(0, 1))
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fname = f"classacc_{model_name}{'_' + tag if tag else ''}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150); plt.close()


def save_results(results: Dict, name: str, save_dir: str = RESULTS_DIR):
    path = os.path.join(save_dir, f"{name}.json")
    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [convert(i) for i in obj]
        return obj
    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  [Results] saved → {path}")


def load_results(name: str, save_dir: str = RESULTS_DIR) -> Optional[Dict]:
    path = os.path.join(save_dir, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
