"""
04_qefs.py — Quantum-Inspired Evolutionary Feature Selection (QEFS)
====================================================================
Adapts the QEFS paper (https://github.com/kanand2003/QEFS…) to PyTorch:

  Step 1: Quantum-Inspired Feature Extraction
    • Load each pretrained CNN backbone (remove classifier head)
    • Project features through a Quantum-Inspired Layer:
        – Parameterised rotation gates (Rx, Ry, Rz analogues)
        – Entanglement-inspired inter-feature mixing (Hadamard + CNOT-like)
    • Result: QCNN feature vectors per sample

  Step 2: HFSEA — Hybrid Firefly-Swallow Evolutionary Algorithm
    • Population of binary masks (1 = feature selected)
    • Fitness = 5-fold CV accuracy of a shallow MLP on selected features
    • Firefly update: position guided by relative brightness (inv-distance)
    • Swallow update: random walk with social attraction
    • Best mask = selected feature subset

  Step 3: Classification with selected features
    • Train shallow MLP / SVM on the selected feature subset
    • Full evaluation: accuracy, F1, confusion matrix, inference time, etc.

  Step 4: Comparison (baseline CNN vs QEFS-enhanced)
    • For every model: test_acc, f1_macro, selected feature count, etc.

Run:
  python 04_qefs.py
  (run 03_train_cnn_models.py first)
"""

import os, sys, json, time, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import (
    set_seed, get_model, get_feature_extractor,
    PlantDiseaseDataset, split_dataframe,
    compute_full_metrics, bootstrap_confidence_interval,
    plot_confusion_matrix, plot_class_accuracies,
    save_results, load_results,
    get_val_transform, get_train_transform,
    DEVICE, CLASS_NAMES, NUM_CLASSES
)

# ── Multi-dir dataset (same as script 03) ─────────────────────────────────
class MultiDirDataset(PlantDiseaseDataset):
    def __init__(self, df, dirs, transform=None):
        super().__init__(df, dirs[0], transform)
        self.dirs = dirs
    def __getitem__(self, idx):
        from PIL import Image as PILImage
        row   = self.df.iloc[idx]
        label = row.get("label", "")
        for d in self.dirs:
            for p in [os.path.join(d, label, row["image_id"]),
                      os.path.join(d, row["image_id"])]:
                if os.path.exists(p):
                    img = PILImage.open(p).convert("RGB")
                    if self.transform: img = self.transform(img)
                    return img, self.class_to_idx[label]
        raise FileNotFoundError(row["image_id"])


def get_image_dirs():
    aug_dir = os.path.join(DATASET_DIR, "aug_images")
    dirs = [TRAIN_IMAGES]
    if os.path.isdir(aug_dir): dirs.append(aug_dir)
    return dirs


# ═══════════════════════════════════════════════════════════════════════════
# Quantum-Inspired Feature Extraction Layer
# ═══════════════════════════════════════════════════════════════════════════
class QuantumInspiredLayer(nn.Module):
    """
    Simulates a parameterised quantum circuit on classical feature vectors.

    Operations (all differentiable classical analogues):
      1. Rotation layer  : element-wise Rx/Ry/Rz-like learned rotations
      2. Hadamard layer  : HxH mixing matrix (sign-structure)
      3. Entanglement    : CNOT-like pairwise product features
      4. Amplitude norm  : normalise output (Born-rule inspired)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Learnable rotation angles θ (Rx = cos θ I + i sin θ X)
        self.theta_x = nn.Parameter(torch.randn(in_dim))
        self.theta_y = nn.Parameter(torch.randn(in_dim))
        self.theta_z = nn.Parameter(torch.randn(in_dim))

        # Hadamard-inspired weight matrix (initialised with ±1/√n)
        H = (torch.randint(0, 2, (out_dim, in_dim)).float() * 2 - 1) / (in_dim ** 0.5)
        self.H = nn.Parameter(H)

        # Entanglement mixing
        self.ent_fc = nn.Linear(in_dim, out_dim)

        # Batch norm + activation
        self.bn  = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rotation gate: R(θ) x = cos(θ) * x_real + sin(θ) * x_imag
        # We treat x as real part; imaginary part simulated via phase shift
        x_r = torch.cos(self.theta_x) * x - torch.sin(self.theta_x) * x.roll(1, -1)
        x_r = torch.cos(self.theta_y) * x_r + torch.sin(self.theta_y) * x_r.roll(2, -1)
        x_r = torch.cos(self.theta_z) * x_r - torch.sin(self.theta_z) * x_r.roll(3, -1)

        # Hadamard mixing
        h_out = x_r @ self.H.T                  # (B, out_dim)

        # Entanglement (pairwise product via learned projection)
        e_out = self.ent_fc(x_r)               # (B, out_dim)

        # Superposition: combine real + entangled
        out = h_out + e_out

        # Amplitude normalisation (Born-rule: probability ∝ |amplitude|²)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)

        return self.act(self.bn(out))


class QCNNHead(nn.Module):
    """Full QCNN: CNN backbone → QuantumInspiredLayer → classifier."""
    def __init__(self, backbone: nn.Module, feat_dim: int,
                 q_dim: int = QEFS_FEATURE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.q_layer  = QuantumInspiredLayer(feat_dim, q_dim)
        self.dropout  = nn.Dropout(0.4)
        self.head     = nn.Linear(q_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        q = self.q_layer(f)
        return self.head(self.dropout(q))

    def get_q_features(self, x):
        with torch.no_grad():
            f = self.backbone(x)
            if isinstance(f, (tuple, list)):
                f = f[0]
            return self.q_layer(f)


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction utilities
# ═══════════════════════════════════════════════════════════════════════════
def extract_features(model: nn.Module, loader: DataLoader,
                     use_quantum: bool = False,
                     q_layer: nn.Module = None) -> tuple:
    """Extract feature vectors and labels from a DataLoader."""
    model.eval()
    if q_layer is not None:
        q_layer.eval()

    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="  Extracting features", ncols=70):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if q_layer is not None:
                feats = q_layer(feats)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)


def get_backbone_dim(model_name: str) -> int:
    _dims = {
        "convnext_base":      1024,
        "densenet161":        2208,
        "googlenet":          1024,
        "mobilenet_v3_large": 960,
        "resnet50":           2048,
        "shufflenet_v2":      1024,
    }
    return _dims.get(model_name, 512)


# ═══════════════════════════════════════════════════════════════════════════
# HFSEA — Hybrid Firefly-Swallow Evolutionary Algorithm
# ═══════════════════════════════════════════════════════════════════════════
class HFSEA:
    """
    Hybrid Firefly-Swallow Evolutionary Algorithm for binary feature selection.

    Each individual encodes a binary mask over the feature space.
    Fitness = cross-validated accuracy of a fast MLPClassifier on selected features.
    """
    def __init__(self, n_features: int,
                 pop_size:  int = QEFS_POP_SIZE,
                 max_iter:  int = QEFS_MAX_ITER,
                 alpha:     float = QEFS_ALPHA,
                 gamma:     float = QEFS_GAMMA,
                 beta0:     float = QEFS_BETA0,
                 delta:     float = QEFS_DELTA,
                 min_features: int = 10):
        self.n_features   = n_features
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.alpha        = alpha    # randomness / absorption
        self.gamma        = gamma    # light absorption coefficient
        self.beta0        = beta0    # attractiveness at r=0
        self.delta        = delta    # step decay (swallow)
        self.min_features = min_features

        # Initialise population: each row = binary mask
        self.pop      = (np.random.rand(pop_size, n_features) > 0.5).astype(float)
        # Ensure at least min_features selected per individual
        for i in range(pop_size):
            if self.pop[i].sum() < min_features:
                self.pop[i, np.random.choice(n_features, min_features, replace=False)] = 1

        self.fitness  = np.zeros(pop_size)
        self.best_mask = None
        self.best_fit  = -np.inf
        self.history   = []

    def evaluate_fitness(self, mask: np.ndarray, X_tr: np.ndarray,
                          y_tr: np.ndarray) -> float:
        """Fast 3-fold CV accuracy on selected features."""
        sel = mask > 0.5
        if sel.sum() < 1:
            return 0.0
        X_sel = X_tr[:, sel]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_sel)
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50,
                            random_state=SEED, early_stopping=True)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        accs = []
        for tr_i, va_i in skf.split(X_s, y_tr):
            clf.fit(X_s[tr_i], y_tr[tr_i])
            accs.append(accuracy_score(y_tr[va_i], clf.predict(X_s[va_i])))
        return float(np.mean(accs)) - 0.001 * sel.sum() / self.n_features  # penalise size

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _binarise(self, pos: np.ndarray) -> np.ndarray:
        prob = self._sigmoid(pos)
        mask = (np.random.rand(*pos.shape) < prob).astype(float)
        # Always keep at least min_features
        if mask.sum() < self.min_features:
            mask[np.random.choice(self.n_features, self.min_features, replace=False)] = 1
        return mask

    def _firefly_update(self, i: int, j: int, pos_i: np.ndarray,
                         pos_j: np.ndarray) -> np.ndarray:
        """Move firefly i towards brighter firefly j."""
        r_sq  = np.sum((pos_i - pos_j) ** 2)
        beta  = self.beta0 * np.exp(-self.gamma * r_sq)
        noise = self.alpha * (np.random.rand(self.n_features) - 0.5)
        return pos_i + beta * (pos_j - pos_i) + noise

    def _swallow_update(self, pos: np.ndarray, best: np.ndarray,
                         step: float) -> np.ndarray:
        """Social attraction towards best + random walk."""
        social = step * (best - pos)
        random_walk = self.alpha * (np.random.rand(self.n_features) - 0.5)
        return pos + social + random_walk

    def run(self, X_tr: np.ndarray, y_tr: np.ndarray,
            verbose: bool = True) -> tuple:
        """
        Run HFSEA and return (best_mask, best_fitness, history).
        """
        # Continuous positions for update rules
        pos = self.pop.copy().astype(float) * 2 - 1   # map {0,1} → {-1,1}

        # Initial fitness
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate_fitness(self._binarise(pos[i]), X_tr, y_tr)
        best_idx = np.argmax(self.fitness)
        self.best_mask = self._binarise(pos[best_idx])
        self.best_fit  = self.fitness[best_idx]
        self.history.append(self.best_fit)

        step = 1.0   # swallow step size

        for it in range(self.max_iter):
            # ── Firefly phase: pairwise comparison ────────────────────────
            order = np.argsort(self.fitness)
            for i_ord in range(self.pop_size):
                i = order[i_ord]
                for j_ord in range(i_ord + 1, self.pop_size):
                    j = order[j_ord]
                    if self.fitness[j] > self.fitness[i]:
                        pos[i] = self._firefly_update(i, j, pos[i], pos[j])
                        mask_i  = self._binarise(pos[i])
                        fit_i   = self.evaluate_fitness(mask_i, X_tr, y_tr)
                        if fit_i > self.fitness[i]:
                            self.fitness[i] = fit_i

            # ── Swallow phase: all move towards best ──────────────────────
            best_idx = np.argmax(self.fitness)
            best_pos = pos[best_idx]
            step    *= self.delta   # decay step
            for i in range(self.pop_size):
                if i != best_idx:
                    pos[i] = self._swallow_update(pos[i], best_pos, step)
                    mask_i  = self._binarise(pos[i])
                    fit_i   = self.evaluate_fitness(mask_i, X_tr, y_tr)
                    self.fitness[i] = fit_i

            # ── Global best ───────────────────────────────────────────────
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fit:
                self.best_fit  = self.fitness[best_idx]
                self.best_mask = self._binarise(pos[best_idx])

            self.history.append(self.best_fit)
            if verbose:
                print(f"    HFSEA iter {it+1:3d}/{self.max_iter}: "
                      f"best_fit={self.best_fit:.4f}  "
                      f"selected={int(self.best_mask.sum())} features")

        return self.best_mask.astype(bool), self.best_fit, self.history


# ═══════════════════════════════════════════════════════════════════════════
# QEFS pipeline for one model
# ═══════════════════════════════════════════════════════════════════════════
def run_qefs_on_model(model_name: str, train_df: pd.DataFrame,
                       val_df: pd.DataFrame, test_df: pd.DataFrame,
                       dirs: list) -> dict:
    print(f"\n{'─'*60}")
    print(f"  QEFS: {model_name}")
    print(f"{'─'*60}")

    # ── Load pretrained backbone ──────────────────────────────────────────
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [!] No checkpoint for {model_name}. Run 03 first. Skipping.")
        return {}

    model = get_model(model_name).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Strip classifier to get backbone feature extractor
    backbone = get_feature_extractor(model, model_name).to(DEVICE).eval()
    feat_dim = get_backbone_dim(model_name)

    # ── Build QuantumInspiredLayer ────────────────────────────────────────
    q_layer = QuantumInspiredLayer(feat_dim, QEFS_FEATURE_DIM).to(DEVICE)

    # Fine-tune the quantum layer while keeping backbone frozen
    print(f"  Fine-tuning Quantum layer ({QEFS_FEATURE_DIM}d) …")
    for p in backbone.parameters():
        p.requires_grad = False

    # Classifier for fine-tuning
    qcnn = QCNNHead(backbone, feat_dim, QEFS_FEATURE_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(qcnn.q_layer.parameters()) + list(qcnn.head.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    from utils import train_one_epoch, evaluate, EarlyStopping, compute_class_weights
    train_ds = MultiDirDataset(train_df, dirs, get_train_transform())
    val_ds   = MultiDirDataset(val_df,   dirs, get_val_transform())
    test_ds  = MultiDirDataset(test_df,  dirs, get_val_transform())

    from torch.utils.data import DataLoader
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ld   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    cw = compute_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    es = EarlyStopping(patience=8)

    best_val_acc = 0.0
    qcnn_path = os.path.join(MODELS_DIR, f"qcnn_{model_name}_best.pth")

    for epoch in range(QEFS_FINETUNE_EPOCHS):
        train_one_epoch(qcnn, train_ld, optimizer, criterion, scaler)
        _, va_acc, _, _ = evaluate(qcnn, val_ld, criterion)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            # Plain state dict (used for reload within this script)
            torch.save(qcnn.state_dict(), qcnn_path)
            # Standard-format checkpoint consumed by downstream scripts (05_genetic_nas)
            torch.save({
                "model_state": qcnn.state_dict(),
                "model_name":  model_name,
                "feat_dim":    feat_dim,        # raw CNN backbone dim
                "q_dim":       QEFS_FEATURE_DIM, # quantum projection dim (512)
            }, os.path.join(MODELS_DIR, f"qefs_{model_name}_best.pth"))
        print(f"    Quantum fine-tune [{epoch+1:2d}/{QEFS_FINETUNE_EPOCHS}]: val_acc={va_acc:.4f}")
        if es(va_acc): break

    if os.path.exists(qcnn_path):
        qcnn.load_state_dict(torch.load(qcnn_path, map_location=DEVICE, weights_only=False))

    # ── Extract quantum features ──────────────────────────────────────────
    print(f"  Extracting QCNN features …")
    qcnn.eval()

    def extract_q(loader):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, lbl in loader:
                q = qcnn.get_q_features(imgs.to(DEVICE))
                feats.append(q.cpu().numpy())
                labels.append(lbl.numpy())
        return np.vstack(feats), np.concatenate(labels)

    X_tr, y_tr = extract_q(train_ld)
    X_va, y_va = extract_q(val_ld)
    X_te, y_te = extract_q(test_ld)

    # Combine train + val for HFSEA
    X_all = np.concatenate([X_tr, X_va])
    y_all = np.concatenate([y_tr, y_va])

    # ── HFSEA Feature Selection ───────────────────────────────────────────
    print(f"\n  Running HFSEA on {X_all.shape[1]}-dim quantum features …")
    hfsea = HFSEA(
        n_features   = X_all.shape[1],
        pop_size     = QEFS_POP_SIZE,
        max_iter     = QEFS_MAX_ITER,
        alpha        = QEFS_ALPHA,
        gamma        = QEFS_GAMMA,
        beta0        = QEFS_BETA0,
        delta        = QEFS_DELTA,
        min_features = max(10, X_all.shape[1] // 10),
    )
    best_mask, best_fitness, hfsea_history = hfsea.run(X_all, y_all, verbose=True)

    n_selected = int(best_mask.sum())
    print(f"\n  HFSEA selected {n_selected}/{X_all.shape[1]} features  (fitness={best_fitness:.4f})")

    # ── Train final classifier on selected features ───────────────────────
    scaler_fs = StandardScaler()
    X_tr_sel  = scaler_fs.fit_transform(X_tr[:, best_mask])
    X_va_sel  = scaler_fs.transform(X_va[:, best_mask])
    X_te_sel  = scaler_fs.transform(X_te[:, best_mask])

    # MLP classifier
    print(f"  Training MLP classifier on {n_selected} selected features …")
    t0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=200, random_state=SEED,
        early_stopping=True, validation_fraction=0.1,
        learning_rate_init=0.001
    )
    clf.fit(X_tr_sel, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    preds_te = clf.predict(X_te_sel)
    inf_time = (time.time() - t1) / len(y_te)

    acc_te = accuracy_score(y_te, preds_te)
    f1_te  = f1_score(y_te, preds_te, average="macro", zero_division=0)

    print(f"  QEFS test_acc={acc_te:.4f}  f1_macro={f1_te:.4f}")

    # Also get QCNN (before HFSEA) accuracy for comparison
    _, qcnn_acc, qcnn_preds, qcnn_lbls = evaluate(qcnn, test_ld, nn.CrossEntropyLoss())

    # Load baseline results
    baseline = load_results(model_name) or {}

    results = {
        "model_name":          model_name,
        "method":              "QEFS",
        "baseline_test_acc":   baseline.get("test_acc", 0),
        "qcnn_test_acc":       float(qcnn_acc),
        "qefs_test_acc":       float(acc_te),
        "qefs_f1_macro":       float(f1_te),
        "n_features_original": int(X_all.shape[1]),
        "n_features_selected": n_selected,
        "feature_reduction_%": float(100 * (1 - n_selected / X_all.shape[1])),
        "hfsea_best_fitness":  float(best_fitness),
        "hfsea_history":       [float(x) for x in hfsea_history],
        "training_time_s":     float(train_time),
        "inference_time_per_img_s": float(inf_time),
        "preds":               preds_te.tolist(),
        "labels":              y_te.tolist(),
    }
    results.update(compute_full_metrics(preds_te, y_te, f"qefs_{model_name}"))

    # ── Plots ──────────────────────────────────────────────────────────────
    # HFSEA convergence
    plt.figure(figsize=(10, 5))
    plt.plot(hfsea_history, "b-o", markersize=3)
    plt.title(f"HFSEA Convergence — {model_name}")
    plt.xlabel("Iteration"); plt.ylabel("Best Fitness (CV accuracy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"qefs_hfsea_convergence_{model_name}.png"), dpi=150)
    plt.close()

    # Feature selection mask
    plt.figure(figsize=(14, 3))
    plt.bar(range(len(best_mask)), best_mask.astype(int),
            color=["#2196F3" if b else "#BDBDBD" for b in best_mask], width=1)
    plt.title(f"HFSEA Selected Features ({n_selected}/{len(best_mask)}) — {model_name}")
    plt.xlabel("Feature Index"); plt.ylabel("Selected (1/0)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"qefs_feature_mask_{model_name}.png"), dpi=150)
    plt.close()

    # Confusion matrix
    plot_confusion_matrix(
        np.array(results["confusion_matrix"]), model_name,
        tag="qefs"
    )
    plot_class_accuracies(results["per_class_acc"], model_name, tag="qefs")

    save_results(results, f"qefs_{model_name}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Comparison plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_qefs_comparison(all_qefs: list):
    if not all_qefs:
        return
    names    = [r["model_name"] for r in all_qefs]
    baseline = [r.get("baseline_test_acc", 0) for r in all_qefs]
    qcnn     = [r.get("qcnn_test_acc",    0) for r in all_qefs]
    qefs     = [r.get("qefs_test_acc",    0) for r in all_qefs]
    n_feat_o = [r.get("n_features_original", 0) for r in all_qefs]
    n_feat_s = [r.get("n_features_selected", 0) for r in all_qefs]

    x = np.arange(len(names)); w = 0.25

    # ── Accuracy comparison ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, baseline, w, label="Baseline CNN",    color="#4e79a7", edgecolor="black")
    ax.bar(x,     qcnn,     w, label="QCNN (no HFSEA)", color="#f28e2b", edgecolor="black")
    ax.bar(x + w, qefs,     w, label="QEFS (QCNN+HFSEA)", color="#59a14f", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(title="QEFS vs Baseline Accuracy Comparison",
           ylabel="Test Accuracy", ylim=(0, 1.1))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qefs_accuracy_comparison.png"), dpi=150)
    plt.close()

    # ── Feature reduction ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(x - 0.2, n_feat_o, 0.4, label="Original",  color="#e15759", edgecolor="black")
    axes[0].bar(x + 0.2, n_feat_s, 0.4, label="Selected",  color="#76b7b2", edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set(title="Feature Count Before/After HFSEA", ylabel="# Features")
    axes[0].legend()

    reductions = [r.get("feature_reduction_%", 0) for r in all_qefs]
    axes[1].bar(names, reductions, color=sns.color_palette("Oranges_r", len(names)),
                edgecolor="black")
    axes[1].set(title="Feature Reduction % by HFSEA", ylabel="Reduction (%)")
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    for i, v in enumerate(reductions):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    plt.suptitle("QEFS Feature Selection Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qefs_feature_reduction.png"), dpi=150)
    plt.close()

    # ── Accuracy gain ─────────────────────────────────────────────────────
    gains = [q - b for q, b in zip(qefs, baseline)]
    colors = ["#4CAF50" if g >= 0 else "#F44336" for g in gains]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, gains, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Accuracy Gain: QEFS vs Baseline", ylabel="Δ Accuracy")
    ax.set_xticklabels(names, rotation=30, ha="right")
    for i, g in enumerate(gains):
        ax.text(i, g + (0.002 if g >= 0 else -0.004),
                f"{g:+.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qefs_accuracy_gain.png"), dpi=150)
    plt.close()

    # ── F1-score comparison ────────────────────────────────────────────────
    f1_baseline = [load_results(r["model_name"]).get("f1_macro", 0)
                   if load_results(r["model_name"]) else 0 for r in all_qefs]
    f1_qefs     = [r.get("qefs_f1_macro", 0) for r in all_qefs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(names, f1_baseline, "o-", label="Baseline F1", color="blue", linewidth=2)
    ax.plot(names, f1_qefs,     "s-", label="QEFS F1",     color="green", linewidth=2)
    ax.set(title="F1-Macro Score: QEFS vs Baseline", ylabel="F1 Macro")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qefs_f1_comparison.png"), dpi=150)
    plt.close()

    print(f"\n[QEFS Plots] saved to {PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_seed()
    print("\n" + "=" * 60)
    print("  AgroDetect — QEFS")
    print("=" * 60)

    csv = BALANCED_CSV if os.path.exists(BALANCED_CSV) else CLEANED_CSV
    if not os.path.exists(csv):
        csv = TRAIN_CSV
    df = pd.read_csv(csv)
    dirs = get_image_dirs()

    def file_exists(row):
        label = row.get("label", "")
        return any(
            os.path.exists(os.path.join(d, label, row["image_id"])) or
            os.path.exists(os.path.join(d, row["image_id"]))
            for d in dirs
        )
    df = df[df.apply(file_exists, axis=1)].reset_index(drop=True)

    train_df, val_df, test_df = split_dataframe(df)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    all_qefs = []
    for model_name in MODEL_NAMES:
        res_path = os.path.join(RESULTS_DIR, f"qefs_{model_name}.json")
        if os.path.exists(res_path):
            print(f"\n  [Skip] QEFS for {model_name} already done.")
            r = load_results(f"qefs_{model_name}")
            if r: all_qefs.append(r)
            continue
        r = run_qefs_on_model(model_name, train_df, val_df, test_df, dirs)
        if r: all_qefs.append(r)

    plot_qefs_comparison(all_qefs)

    print("\n" + "=" * 70)
    print(f"  {'Model':<25} {'Baseline':>10} {'QCNN':>10} {'QEFS':>10} {'ΔAcc':>10} {'#Feat':>8}")
    print("  " + "-" * 68)
    for r in all_qefs:
        delta = r.get("qefs_test_acc", 0) - r.get("baseline_test_acc", 0)
        print(f"  {r['model_name']:<25} "
              f"{r.get('baseline_test_acc', 0):>10.4f} "
              f"{r.get('qcnn_test_acc', 0):>10.4f} "
              f"{r.get('qefs_test_acc', 0):>10.4f} "
              f"{delta:>+10.4f} "
              f"{r.get('n_features_selected', 0):>8d}")
    print("=" * 70)

    print("\n✓ QEFS complete.\n")
