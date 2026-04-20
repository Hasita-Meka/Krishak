"""
06_continual_learning.py — Continual Learning on AgroDetect
=============================================================
Implements 5 Continual Learning methods from the Mammoth framework
(https://github.com/aimagelab/mammoth) adapted to PyTorch + our dataset:

  Method       | Description
  ─────────────┼──────────────────────────────────────────────────────
  GDumb        | Store balanced memory, retrain from scratch each task
  DER++        | Dark Experience Replay++ (soft + hard targets distillation)
  X-DER        | Extended DER (bilateral soft-label distillation)
  A-GEM        | Averaged Gradient Episodic Memory
  ER-ACE       | Experience Replay with Asymmetric Cross-Entropy

Setup:
  • 10 classes split into 5 tasks (2 classes per task)
  • Backbone: ResNet50 (pretrained, frozen) + learnable task head
  • Replay buffer: 500 samples (reservoir sampling)
  • Per-method: forward transfer, backward transfer, avg accuracy

Metrics per method:
  • Task accuracy matrix (ACC[i][j] = acc on task j after training task i)
  • Average Accuracy (AA), Backward Transfer (BWT), Forward Transfer (FWT)
  • Confusion matrix on all classes after all tasks
  • Memory usage, training time, inference time

Run:
  python 06_continual_learning.py
"""

import os, sys, json, time, copy, random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import (
    set_seed, get_model, get_feature_extractor,
    PlantDiseaseDataset, split_dataframe,
    compute_full_metrics, compute_class_weights,
    get_train_transform, get_val_transform,
    train_one_epoch, evaluate, EarlyStopping,
    plot_confusion_matrix, plot_class_accuracies,
    save_results, load_results, DEVICE, CLASS_NAMES, NUM_CLASSES,
    DROPOUT, LABEL_SMOOTHING, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, USE_AMP, ETA_MIN
)

# ── Import QEFS building blocks to reconstruct GNAS backbone ──────────────
import importlib.util as _ilu
_qefs_spec = _ilu.spec_from_file_location(
    "qefs_module",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_qefs.py")
)
_qefs_mod = _ilu.module_from_spec(_qefs_spec)
_qefs_spec.loader.exec_module(_qefs_mod)
QuantumInspiredLayer = _qefs_mod.QuantumInspiredLayer
QCNNHead             = _qefs_mod.QCNNHead
_get_backbone_dim    = _qefs_mod.get_backbone_dim   # raw CNN feature dims


class _QEFSFeatureExtractor(nn.Module):
    """backbone (CNN) + QuantumInspiredLayer → QEFS_FEATURE_DIM-dim features."""
    def __init__(self, qcnn: QCNNHead):
        super().__init__()
        self.backbone = qcnn.backbone
        self.q_layer  = qcnn.q_layer
    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)): f = f[0]
        return self.q_layer(f)

# ═══════════════════════════════════════════════════════════════════════════
# Task setup
# ═══════════════════════════════════════════════════════════════════════════
N_TASKS  = CL_N_TASKS      # 5
TASKS    = [CLASS_NAMES[i*2:(i+1)*2] for i in range(N_TASKS)]
# e.g. Task 0: [bacterial_leaf_blight, bacterial_leaf_streak], ...

# CL_MODEL_NAME comes from config (default "resnet50"); override via config patch for tests
BUFFER_SIZE     = CL_BUFFER_SIZE
EPOCHS_PER_TASK = CL_EPOCHS_PER_TASK


def print_tasks():
    print("\n  Task split:")
    for i, t in enumerate(TASKS):
        print(f"    Task {i}: {t}")


# ─── Multi-dir dataset ─────────────────────────────────────────────────────
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
    aug = os.path.join(DATASET_DIR, "aug_images")
    dirs = [TRAIN_IMAGES]
    if os.path.isdir(aug): dirs.append(aug)
    return dirs


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning backbone
# ═══════════════════════════════════════════════════════════════════════════
class CLModel(nn.Module):
    """
    Pretrained backbone (frozen) + learnable all-class head.
    Checkpoint priority (best → fallback):
      1. gnas_{name}_best.pth   — GNAS-optimised (may wrap a QEFS backbone)
      2. qefs_{name}_best.pth   — QEFS quantum-enhanced backbone
      3. {name}_best.pth        — baseline CNN checkpoint
    We use a shared head over all NUM_CLASSES to enable backward-transfer measurement.
    """
    def __init__(self, backbone_name: str = CL_MODEL_NAME):
        super().__init__()

        gnas_path = os.path.join(MODELS_DIR, f"gnas_{backbone_name}_best.pth")
        qefs_path = os.path.join(MODELS_DIR, f"qefs_{backbone_name}_best.pth")
        base_path = os.path.join(MODELS_DIR, f"{backbone_name}_best.pth")

        if os.path.exists(gnas_path):
            self._load_from_gnas(backbone_name, gnas_path)
        elif os.path.exists(qefs_path):
            self._load_from_qefs(backbone_name, qefs_path)
        else:
            self._load_from_baseline(backbone_name, base_path)

        self.fc      = nn.Linear(self.feat_dim, NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT)
        # Ensure all submodules (backbone + fc + dropout) are on the same device.
        # The backbone loaders each call .to(DEVICE) internally; fc/dropout are
        # created on CPU by default, so we synchronise everything here.
        self.to(DEVICE)

    # ── Backbone loaders ─────────────────────────────────────────────────

    def _load_from_gnas(self, backbone_name: str, gnas_path: str):
        """Reconstruct the GNAS model, extract its backbone, discard the head."""
        ckpt      = torch.load(gnas_path, map_location=DEVICE, weights_only=False)
        uses_qefs = ckpt.get("uses_qefs", False)
        feat_dim  = ckpt.get("feat_dim",  2048)

        if uses_qefs:
            # GNAS backbone = QEFSFeatureExtractor (CNN + QuantumInspiredLayer → 512-dim)
            raw_dim  = _get_backbone_dim(backbone_name)
            base     = get_model(backbone_name).to(DEVICE)
            base_ext = get_feature_extractor(base, backbone_name)
            qcnn     = QCNNHead(base_ext, raw_dim, feat_dim).to(DEVICE)
            extractor = _QEFSFeatureExtractor(qcnn).to(DEVICE)
        else:
            # GNAS backbone = plain CNN feature extractor
            base      = get_model(backbone_name).to(DEVICE)
            extractor = get_feature_extractor(base, backbone_name).to(DEVICE)

        # Pluck only the backbone.* keys from the full GNAS state dict
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in ckpt["model_state"].items()
            if k.startswith("backbone.")
        }
        extractor.load_state_dict(backbone_state)
        self.backbone = extractor
        self.feat_dim = feat_dim
        print(f"  [CL] Loaded GNAS backbone (uses_qefs={uses_qefs}, "
              f"feat_dim={feat_dim}) from {os.path.basename(gnas_path)}")

    def _load_from_qefs(self, backbone_name: str, qefs_path: str):
        """Build a QEFS backbone (CNN + QuantumInspiredLayer → 512-dim)."""
        raw_dim  = _get_backbone_dim(backbone_name)
        base     = get_model(backbone_name).to(DEVICE)
        base_ext = get_feature_extractor(base, backbone_name)
        qcnn     = QCNNHead(base_ext, raw_dim, QEFS_FEATURE_DIM).to(DEVICE)
        ckpt     = torch.load(qefs_path, map_location=DEVICE, weights_only=False)
        state    = ckpt.get("model_state", ckpt)
        qcnn.load_state_dict(state)
        extractor = _QEFSFeatureExtractor(qcnn).to(DEVICE)
        self.backbone = extractor
        self.feat_dim = QEFS_FEATURE_DIM
        print(f"  [CL] Loaded QEFS backbone (feat_dim=512) from {os.path.basename(qefs_path)}")

    def _load_from_baseline(self, backbone_name: str, base_path: str):
        """Standard CNN feature extractor (baseline fallback)."""
        base = get_model(backbone_name).to(DEVICE)
        if os.path.exists(base_path):
            ckpt = torch.load(base_path, map_location=DEVICE, weights_only=False)
            base.load_state_dict(ckpt["model_state"])
            print(f"  [CL] Loaded baseline backbone from {os.path.basename(base_path)}")
        else:
            print(f"  [CL] WARNING: no checkpoint found for {backbone_name}; using random weights")
        self.backbone = get_feature_extractor(base, backbone_name)
        self.feat_dim = _get_backbone_dim(backbone_name)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)): f = f[0]
        return self.fc(self.dropout(f))

    def features(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)): f = f[0]
        return f


# ═══════════════════════════════════════════════════════════════════════════
# Replay Buffer (Reservoir Sampling)
# ═══════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data:   list = []   # list of (img_tensor, label, logits)
        self.n_seen: int  = 0

    def add(self, imgs: torch.Tensor, labels: torch.Tensor,
            logits: torch.Tensor = None):
        """Reservoir sampling."""
        for i in range(imgs.size(0)):
            item = (imgs[i].cpu(), labels[i].item(),
                    logits[i].cpu() if logits is not None else None)
            if len(self.data) < self.capacity:
                self.data.append(item)
            else:
                j = random.randint(0, self.n_seen)
                if j < self.capacity:
                    self.data[j] = item
            self.n_seen += 1

    def sample(self, n: int) -> tuple:
        n = min(n, len(self.data))
        if n == 0:
            return None, None, None
        batch = random.sample(self.data, n)
        imgs   = torch.stack([b[0] for b in batch]).to(DEVICE)
        labels = torch.tensor([b[1] for b in batch], device=DEVICE)
        logits = torch.stack([b[2] for b in batch]).to(DEVICE) \
                 if batch[0][2] is not None else None
        return imgs, labels, logits

    def __len__(self):
        return len(self.data)


# ═══════════════════════════════════════════════════════════════════════════
# Task data helpers
# ═══════════════════════════════════════════════════════════════════════════
def get_task_loaders(df, dirs, task_idx, split="train"):
    task_classes = TASKS[task_idx]
    task_df = df[df["label"].isin(task_classes)].reset_index(drop=True)
    transform = get_train_transform() if split == "train" else get_val_transform()
    ds = MultiDirDataset(task_df, dirs, transform)
    shuffle = (split == "train")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def get_all_loaders(test_df, dirs):
    """Return per-task test loaders for accuracy matrix."""
    loaders = []
    for t in range(N_TASKS):
        loaders.append(get_task_loaders(test_df, dirs, t, split="test"))
    return loaders


# ═══════════════════════════════════════════════════════════════════════════
# Continual Learning metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_cl_metrics(acc_matrix: np.ndarray) -> dict:
    """
    acc_matrix[i][j] = accuracy on task j after training task i.
    Shape: (N_TASKS, N_TASKS).
    """
    N = N_TASKS
    AA  = float(np.mean([acc_matrix[N-1, j] for j in range(N)]))
    BWT = float(np.mean([acc_matrix[N-1, j] - acc_matrix[j, j] for j in range(N-1)]))
    # FWT: acc on task j before training it, compared to random (0)
    FWT = float(np.mean([acc_matrix[j-1, j] for j in range(1, N)])) if N > 1 else 0.0
    return {"AA": AA, "BWT": BWT, "FWT": FWT}


def task_accuracy(model, loader) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())
    return accuracy_score(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════════════════
# 1. GDumb
# ═══════════════════════════════════════════════════════════════════════════
class GDumb:
    """
    Store a balanced memory of M samples.
    At each task, retrain the model from scratch on the memory alone.
    """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = {}  # class → list of (img_tensor, label)
        self.max_per_class = buffer_size // NUM_CLASSES

    def update_memory(self, loader):
        for imgs, labels in loader:
            for img, lbl in zip(imgs, labels):
                c = lbl.item()
                if c not in self.buffer: self.buffer[c] = []
                if len(self.buffer[c]) < self.max_per_class:
                    self.buffer[c].append((img.clone(), c))

    def get_memory_loader(self):
        all_items = []
        for c, items in self.buffer.items():
            all_items.extend(items)
        if not all_items: return None
        random.shuffle(all_items)
        imgs   = torch.stack([x[0] for x in all_items])
        labels = torch.tensor([x[1] for x in all_items])
        ds = torch.utils.data.TensorDataset(imgs, labels)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    def train_from_memory(self, model, n_epochs=EPOCHS_PER_TASK):
        loader = self.get_memory_loader()
        if loader is None: return
        model.train()
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = GradScaler(device="cuda", enabled=USE_AMP)
        for ep in range(n_epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=USE_AMP):
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()


# ═══════════════════════════════════════════════════════════════════════════
# 2. DER++ (Dark Experience Replay++)
# ═══════════════════════════════════════════════════════════════════════════
class DERpp:
    """
    Combines:
      • CE loss on current task
      • MSE distillation on buffer logits (dark knowledge)
      • CE on buffer labels
    """
    def __init__(self, alpha=0.5, beta=0.5, buffer_size=BUFFER_SIZE):
        self.alpha  = alpha   # weight for logit distillation
        self.beta   = beta    # weight for buffer CE
        self.buffer = ReplayBuffer(buffer_size)

    def observe(self, model, imgs, labels, opt, scaler):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss_ce = F.cross_entropy(logits, labels)

            loss = loss_ce
            if len(self.buffer) > 0:
                b_imgs, b_labels, b_logits = self.buffer.sample(BATCH_SIZE)
                b_out = model(b_imgs)
                # Logit distillation (dark knowledge)
                loss_dark = F.mse_loss(b_out, b_logits) if b_logits is not None else 0.0
                # CE on buffer labels
                loss_buf  = F.cross_entropy(b_out, b_labels)
                loss = loss_ce + self.alpha * loss_dark + self.beta * loss_buf

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        # Store current batch in buffer WITH logits (no grad)
        with torch.no_grad():
            lg = model(imgs).detach()
        self.buffer.add(imgs.cpu(), labels.cpu(), lg.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 3. X-DER (Extended DER)
# ═══════════════════════════════════════════════════════════════════════════
class XDER:
    """
    X-DER adds bilateral distillation:
      • Forward: current model predicts buffer logits
      • Backward: buffer model predicts current logits (stored at add-time)
    Plus a semantic regularisation term.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.1, buffer_size=BUFFER_SIZE):
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma   # semantic regularisation weight
        self.buffer = ReplayBuffer(buffer_size)

    def observe(self, model, imgs, labels, opt, scaler, task_id: int = 0):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss_ce = F.cross_entropy(logits, labels)
            loss = loss_ce

            if len(self.buffer) > 0:
                b_imgs, b_labels, b_logits = self.buffer.sample(BATCH_SIZE)
                b_out = model(b_imgs)

                # Forward distillation (DER++ style)
                loss_fwd = F.mse_loss(b_out, b_logits) if b_logits is not None else 0.0

                # Backward: penalise current logits changing on buffer data
                # (soft targets from stored logits)
                loss_bwd = F.kl_div(
                    F.log_softmax(b_out, dim=-1),
                    F.softmax(b_logits, dim=-1),
                    reduction="batchmean"
                ) if b_logits is not None else 0.0

                # Semantic: current features should be consistent
                loss_sem = F.cross_entropy(b_out, b_labels)

                loss = loss_ce + self.alpha * loss_fwd + self.beta * loss_bwd + \
                       self.gamma * loss_sem

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        with torch.no_grad():
            lg = model(imgs).detach()
        self.buffer.add(imgs.cpu(), labels.cpu(), lg.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 4. A-GEM (Averaged Gradient Episodic Memory)
# ═══════════════════════════════════════════════════════════════════════════
class AGEM:
    """
    Projects gradients so that the loss on a reference buffer subset
    does not increase. Uses averaged episodic memory constraint.
    """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = ReplayBuffer(buffer_size)

    def _project_grad(self, model, ref_grad: torch.Tensor) -> None:
        """Project current gradients so they don't increase buffer loss."""
        params = [p for p in model.parameters() if p.grad is not None]
        cur_grads = torch.cat([p.grad.view(-1) for p in params])

        dot = torch.dot(cur_grads, ref_grad)
        if dot < 0:
            # Project onto the half-space
            proj = cur_grads - dot / (ref_grad.dot(ref_grad) + 1e-8) * ref_grad
            offset = 0
            for p in params:
                n = p.grad.numel()
                p.grad.copy_(proj[offset: offset + n].view_as(p.grad))
                offset += n

    def observe(self, model, imgs, labels, opt, scaler):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # ── Reference gradient from buffer ─────────────────────────────────
        ref_grad = None
        if len(self.buffer) > 0:
            b_imgs, b_labels, _ = self.buffer.sample(BATCH_SIZE)
            opt.zero_grad(set_to_none=True)
            b_loss = F.cross_entropy(model(b_imgs), b_labels)
            b_loss.backward()
            ref_grad = torch.cat([p.grad.view(-1).clone()
                                  for p in model.parameters()
                                  if p.grad is not None])
            opt.zero_grad(set_to_none=True)

        # ── Current task gradient ──────────────────────────────────────────
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            loss = F.cross_entropy(model(imgs), labels)
        scaler.scale(loss).backward()

        if ref_grad is not None:
            self._project_grad(model, ref_grad)

        scaler.step(opt); scaler.update()
        self.buffer.add(imgs.cpu(), labels.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 5. ER-ACE (Experience Replay + Asymmetric Cross-Entropy)
# ═══════════════════════════════════════════════════════════════════════════
class ERACE:
    """
    ER-ACE masks out old-class logits when computing the CE loss on
    current task data, preventing interference. Buffer replayed with
    standard CE.
    """
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = ReplayBuffer(buffer_size)
        self.seen_classes = set()

    def _masked_ce(self, logits: torch.Tensor, labels: torch.Tensor,
                   present_classes: set) -> torch.Tensor:
        """CE only over classes seen in the current task (ACE component)."""
        mask = torch.full((NUM_CLASSES,), float("-inf"), device=DEVICE)
        for c in present_classes:
            mask[c] = 0.0
        return F.cross_entropy(logits + mask, labels)

    def observe(self, model, imgs, labels, opt, scaler, present_classes: set):
        model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        self.seen_classes.update(present_classes)
        opt.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(imgs)
            # Asymmetric CE (mask out unseen old classes)
            loss = self._masked_ce(logits, labels, present_classes)

            if len(self.buffer) > 0:
                b_imgs, b_labels, _ = self.buffer.sample(BATCH_SIZE)
                b_out  = model(b_imgs)
                loss  += F.cross_entropy(b_out, b_labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        self.buffer.add(imgs.cpu(), labels.cpu())
        return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# Generic training loop for all CL methods
# ═══════════════════════════════════════════════════════════════════════════
def cl_train_loop(method_name: str, method_obj, model: nn.Module,
                  train_df, test_df, dirs) -> dict:
    """Run the full CL sequence for a given method."""
    print(f"\n{'─'*60}")
    print(f"  CL Method: {method_name}")
    print(f"{'─'*60}")

    set_seed()
    opt    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    task_loaders_test = get_all_loaders(test_df, dirs)

    acc_matrix  = np.zeros((N_TASKS, N_TASKS))
    loss_curves = []  # per task
    t_total     = time.time()

    for task_id in range(N_TASKS):
        task_classes = TASKS[task_id]
        task_class_ids = {CLASS_TO_IDX[c] for c in task_classes}
        print(f"\n  Task {task_id}: {task_classes}")

        train_ld = get_task_loaders(train_df, dirs, task_id, split="train")
        task_losses = []

        # Special case: GDumb — update memory, then retrain from scratch
        if isinstance(method_obj, GDumb):
            method_obj.update_memory(train_ld)
            model_clone = copy.deepcopy(model)
            method_obj.train_from_memory(model_clone, n_epochs=EPOCHS_PER_TASK)
            model.load_state_dict(model_clone.state_dict())
        else:
            # Standard iterative observe loop
            for epoch in range(EPOCHS_PER_TASK):
                ep_loss = 0.0; n_batches = 0
                for imgs, labels in train_ld:
                    if isinstance(method_obj, AGEM):
                        l = method_obj.observe(model, imgs, labels, opt, scaler)
                    elif isinstance(method_obj, ERACE):
                        l = method_obj.observe(model, imgs, labels, opt, scaler,
                                               task_class_ids)
                    elif isinstance(method_obj, (DERpp, XDER)):
                        l = method_obj.observe(model, imgs, labels, opt, scaler)
                    else:
                        l = 0.0
                    ep_loss += l; n_batches += 1
                avg_loss = ep_loss / max(n_batches, 1)
                task_losses.append(avg_loss)
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{EPOCHS_PER_TASK} | loss={avg_loss:.4f}")

        loss_curves.append(task_losses)

        # Evaluate on ALL tasks seen so far
        model.eval()
        for prev_task in range(N_TASKS):
            acc = task_accuracy(model, task_loaders_test[prev_task])
            acc_matrix[task_id, prev_task] = acc
        print(f"  Acc after Task {task_id}: {acc_matrix[task_id, :task_id+1].tolist()}")

    total_time = time.time() - t_total

    cl_metrics = compute_cl_metrics(acc_matrix)
    print(f"\n  AA={cl_metrics['AA']:.4f}  BWT={cl_metrics['BWT']:.4f}  FWT={cl_metrics['FWT']:.4f}")

    # Final evaluation: all classes
    print(f"\n  Final evaluation on all classes …")
    model.eval()
    all_preds, all_labels = [], []
    full_test_ds = MultiDirDataset(test_df, dirs, get_val_transform())
    full_test_ld = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    with torch.no_grad():
        for imgs, lbls in full_test_ld:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc_all = accuracy_score(labels, preds)
    f1_all  = f1_score(labels, preds, average="macro", zero_division=0)

    # Inference time
    t0 = time.time()
    with torch.no_grad():
        for imgs, _ in full_test_ld:
            _ = model(imgs.to(DEVICE))
    inf_time = (time.time() - t0) / len(full_test_ld.dataset)

    results = {
        "method":             method_name,
        "model_name":         CL_MODEL_NAME,
        "acc_matrix":         acc_matrix.tolist(),
        "AA":                 cl_metrics["AA"],
        "BWT":                cl_metrics["BWT"],
        "FWT":                cl_metrics["FWT"],
        "final_test_acc":     float(acc_all),
        "final_f1_macro":     float(f1_all),
        "total_training_time_s": total_time,
        "inference_time_per_img_s": float(inf_time),
        "buffer_size":        BUFFER_SIZE,
        "n_tasks":            N_TASKS,
        "epochs_per_task":    EPOCHS_PER_TASK,
        "preds":              preds.tolist(),
        "labels":             labels.tolist(),
    }
    results.update(compute_full_metrics(preds, labels, method_name))
    save_results(results, f"cl_{method_name.lower().replace('-','_').replace('+','p')}")

    # Plots
    plot_confusion_matrix(np.array(results["confusion_matrix"]),
                          method_name, tag="cl")
    plot_task_accuracy_matrix(acc_matrix, method_name)
    plot_task_loss_curves(loss_curves, method_name)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_task_accuracy_matrix(acc_matrix: np.ndarray, method_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(acc_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[f"Task {i}" for i in range(N_TASKS)],
                yticklabels=[f"After Task {i}" for i in range(N_TASKS)],
                ax=ax, vmin=0, vmax=1)
    ax.set(title=f"Task Accuracy Matrix — {method_name}",
           xlabel="Evaluated on", ylabel="Trained up to")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_acc_matrix_{method_name}.png"), dpi=150)
    plt.close()


def plot_task_loss_curves(loss_curves: list, method_name: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("tab10", N_TASKS)
    offset = 0
    for t, lc in enumerate(loss_curves):
        x = range(offset, offset + len(lc))
        ax.plot(x, lc, color=palette[t], label=f"Task {t}", linewidth=2)
        ax.axvline(offset, color=palette[t], linestyle=":", alpha=0.5)
        offset += len(lc)
    ax.set(title=f"Training Loss Curves — {method_name}",
           xlabel="Epoch (cumulative)", ylabel="Loss")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cl_loss_curves_{method_name}.png"), dpi=150)
    plt.close()


def plot_cl_comparison(all_results: list):
    if not all_results: return
    methods = [r["method"] for r in all_results]
    AA      = [r["AA"]  for r in all_results]
    BWT     = [r["BWT"] for r in all_results]
    FWT     = [r["FWT"] for r in all_results]
    final   = [r["final_test_acc"] for r in all_results]
    f1      = [r["final_f1_macro"] for r in all_results]
    times   = [r["total_training_time_s"] / 60 for r in all_results]

    palette = sns.color_palette("Set2", len(methods))
    x = np.arange(len(methods))

    # ── 1. AA / BWT / FWT ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, vals, title, color in zip(
            axes, [AA, BWT, FWT],
            ["Average Accuracy (AA)", "Backward Transfer (BWT)", "Forward Transfer (FWT)"],
            ["#4e79a7", "#f28e2b", "#59a14f"]):
        bars = ax.bar(methods, vals, color=color, edgecolor="black")
        ax.set(title=title, ylabel="Score")
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.002 if v >= 0 else -0.005),
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Continual Learning Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cl_aa_bwt_fwt.png"), dpi=150)
    plt.close()

    # ── 2. Final accuracy & F1 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, vals, title in zip(axes, [final, f1],
                               ["Final Test Accuracy", "Final F1-Macro"]):
        bars = ax.bar(methods, vals, color=palette, edgecolor="black")
        ax.set(title=title, ylabel="Score", ylim=(0, 1.1))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("CL Methods — Final Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cl_final_performance.png"), dpi=150)
    plt.close()

    # ── 3. Radar chart ─────────────────────────────────────────────────────
    cats = ["AA", "BWT+1", "FWT+1", "Test Acc", "F1 Macro"]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (method, r) in enumerate(zip(methods, all_results)):
        vals_raw = [r["AA"], r["BWT"] + 1, r["FWT"] + 1,
                    r["final_test_acc"], r["final_f1_macro"]]
        vals_raw += [vals_raw[0]]
        ax.plot(angles, vals_raw, "o-", linewidth=2, label=method,
                color=palette[i])
        ax.fill(angles, vals_raw, alpha=0.1, color=palette[i])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_title("CL Methods Radar Chart", fontsize=14, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cl_radar.png"), dpi=150)
    plt.close()

    # ── 4. Training time ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, times, color=palette, edgecolor="black")
    ax.set(title="Total Training Time (minutes)", ylabel="Minutes")
    ax.set_xticklabels(methods, rotation=30, ha="right")
    for i, v in enumerate(times): ax.text(i, v + 0.1, f"{v:.1f}m", ha="center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cl_training_time.png"), dpi=150)
    plt.close()

    # ── 5. Task accuracy progression heatmap grid ─────────────────────────
    n_m = len(all_results)
    cols = min(3, n_m); rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    if n_m == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]
    else: axes = axes.tolist()

    flat_axes = [ax for row in axes for ax in (row if isinstance(row, list) else [row])]
    for ax, r in zip(flat_axes, all_results):
        mat = np.array(r["acc_matrix"])
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"T{i}" for i in range(N_TASKS)],
                    yticklabels=[f"T{i}" for i in range(N_TASKS)],
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(r["method"], fontsize=12)
    for ax in flat_axes[n_m:]: ax.axis("off")
    plt.suptitle("Task Accuracy Matrices — All CL Methods", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cl_all_acc_matrices.png"), dpi=150)
    plt.close()

    print(f"\n[CL Plots] saved to {PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_seed()
    print("\n" + "=" * 60)
    print("  AgroDetect — Continual Learning")
    print("=" * 60)
    print_tasks()

    csv = BALANCED_CSV if os.path.exists(BALANCED_CSV) else CLEANED_CSV
    if not os.path.exists(csv): csv = TRAIN_CSV
    df   = pd.read_csv(csv)
    dirs = get_image_dirs()

    def exists(row):
        label = row.get("label", "")
        return any(
            os.path.exists(os.path.join(d, label, row["image_id"])) or
            os.path.exists(os.path.join(d, row["image_id"]))
            for d in dirs
        )
    df = df[df.apply(exists, axis=1)].reset_index(drop=True)

    train_df, val_df, test_df = split_dataframe(df)
    all_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"  Train: {len(all_df):,}  Test: {len(test_df):,}")

    methods = {
        "GDumb":  (GDumb,  {}),
        "DER++":  (DERpp,  {"alpha": 0.5, "beta": 0.5}),
        "X-DER":  (XDER,   {"alpha": 0.5, "beta": 0.5, "gamma": 0.1}),
        "A-GEM":  (AGEM,   {}),
        "ER-ACE": (ERACE,  {}),
    }

    all_results = []
    for method_name, (MethodClass, kwargs) in methods.items():
        safe_name = method_name.lower().replace("-", "_").replace("+", "p")
        res_path  = os.path.join(RESULTS_DIR, f"cl_{safe_name}.json")
        if os.path.exists(res_path):
            print(f"\n  [Skip] {method_name} already done.")
            r = load_results(f"cl_{safe_name}")
            if r: all_results.append(r)
            continue

        # Fresh model for each method
        model = CLModel(CL_MODEL_NAME).to(DEVICE)
        method_obj = MethodClass(buffer_size=BUFFER_SIZE, **kwargs)
        r = cl_train_loop(method_name, method_obj, model, all_df, test_df, dirs)
        if r: all_results.append(r)
        del model; torch.cuda.empty_cache()

    plot_cl_comparison(all_results)

    print("\n" + "=" * 75)
    print(f"  {'Method':<12} {'AA':>8} {'BWT':>8} {'FWT':>8} {'TestAcc':>10} {'F1':>8}")
    print("  " + "-" * 70)
    for r in all_results:
        print(f"  {r['method']:<12} "
              f"{r.get('AA', 0):>8.4f} "
              f"{r.get('BWT', 0):>8.4f} "
              f"{r.get('FWT', 0):>8.4f} "
              f"{r.get('final_test_acc', 0):>10.4f} "
              f"{r.get('final_f1_macro', 0):>8.4f}")
    print("=" * 75)

    print("\n✓ Continual Learning complete.\n")
