"""
05_genetic_nas.py — Genetic Neural Architecture Search (Genetic NAS)
=====================================================================
Adapts the Genetic NAS approach (https://github.com/sypsyp97/Genetic_NAS)
to PyTorch for the AgroDetect plant disease dataset.

For each of the 6 pretrained CNN backbones, a Genetic Algorithm searches
for the optimal classifier HEAD architecture, since modifying entire
backbones would be prohibitively expensive.

Search Space (per model):
  • Number of FC layers      : {1, 2, 3}
  • Hidden dim per layer     : {128, 256, 512, 1024}
  • Dropout rate             : {0.2, 0.3, 0.4, 0.5}
  • Activation function      : {ReLU, GELU, SiLU, Mish}
  • Batch Norm use           : {True, False}
  • Skip connection use      : {True, False}   (residual in head)
  • Learning rate multiplier : {0.5, 1.0, 2.0, 5.0} × base LR

Chromosome encoding: 14-bit binary string
  [n_layers(2)] [hidden_dim(2)] [dropout(2)] [act(2)] [bn(1)] [skip(1)] [lr_mult(2)]
   total = 12 bits   (extra bits ignored / padded)

Genetic Operations:
  • Tournament Selection (k=3)
  • Single-point Crossover (p_c = 0.8)
  • Bit-flip Mutation     (p_m = 0.15)

Fitness: validation accuracy after GNAS_EVAL_EPOCHS epochs of training.
Best architecture is then fully trained and evaluated on the test set.

Run:
  python 05_genetic_nas.py
  (run 03_train_cnn_models.py first)
"""

import os, sys, json, time, copy, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    set_seed, get_model, get_feature_extractor, PlantDiseaseDataset,
    split_dataframe, compute_full_metrics, compute_class_weights,
    bootstrap_confidence_interval,
    plot_confusion_matrix, plot_class_accuracies,
    plot_training_curves, save_results, load_results,
    get_train_transform, get_val_transform,
    train_one_epoch, evaluate, EarlyStopping,
    DEVICE
)

# ── Import QEFS classes for building the quantum-enhanced backbone ─────────
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

# ── Multi-dir dataset (same as 03) ────────────────────────────────────────
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
# Search space
# ═══════════════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    "n_layers":   [1, 2, 3],
    "hidden_dim": [128, 256, 512, 1024],
    "dropout":    [0.2, 0.3, 0.4, 0.5],
    "activation": ["relu", "gelu", "silu", "mish"],
    "use_bn":     [False, True],
    "use_skip":   [False, True],
    "lr_mult":    [0.5, 1.0, 2.0, 5.0],
}

CHROM_BITS = {
    # (gene_name, n_bits, options_list)
    "n_layers":   (2, [1, 2, 3]),          # 00=1, 01=2, 10=3, 11=3
    "hidden_dim": (2, [128, 256, 512, 1024]),
    "dropout":    (2, [0.2, 0.3, 0.4, 0.5]),
    "activation": (2, ["relu", "gelu", "silu", "mish"]),
    "use_bn":     (1, [False, True]),
    "use_skip":   (1, [False, True]),
    "lr_mult":    (2, [0.5, 1.0, 2.0, 5.0]),
}
CHROM_LEN = sum(b for _, (b, _) in CHROM_BITS.items())   # = 12


# ─── Chromosome ↔ architecture dict conversions ───────────────────────────
def decode_chromosome(chrom: list) -> dict:
    arch = {}
    pos = 0
    for gene, (n_bits, options) in CHROM_BITS.items():
        bits = chrom[pos: pos + n_bits]
        idx  = min(int("".join(str(b) for b in bits), 2), len(options) - 1)
        arch[gene] = options[idx]
        pos += n_bits
    return arch


def encode_chromosome(arch: dict) -> list:
    chrom = []
    for gene, (n_bits, options) in CHROM_BITS.items():
        val = arch[gene]
        idx = options.index(val) if val in options else 0
        bits = list(map(int, format(idx, f"0{n_bits}b")))
        chrom.extend(bits)
    return chrom


def random_chromosome() -> list:
    return [random.randint(0, 1) for _ in range(CHROM_LEN)]


# ═══════════════════════════════════════════════════════════════════════════
# Classifier head builder
# ═══════════════════════════════════════════════════════════════════════════
def build_activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(),
            "silu": nn.SiLU(), "mish": nn.Mish()}.get(name, nn.GELU())


class SearchableHead(nn.Module):
    """Dynamic FC head with optional BN and skip connections."""
    def __init__(self, in_dim: int, arch: dict, num_classes: int = NUM_CLASSES):
        super().__init__()
        layers = []
        dims   = [in_dim]
        for _ in range(arch["n_layers"]):
            dims.append(arch["hidden_dim"])

        self.fcs      = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.acts     = nn.ModuleList()
        self.use_bn   = arch["use_bn"]
        self.use_skip = arch["use_skip"]
        self.dropout  = nn.Dropout(arch["dropout"])
        # Plain Python list — Identity has no params, None means no skip;
        # a plain list avoids nn.ModuleList strict-mode state_dict issues with None.
        self._skip_flags: list = []

        for i in range(arch["n_layers"]):
            self.fcs.append(nn.Linear(dims[i], dims[i+1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]) if arch["use_bn"] else nn.Identity())
            self.acts.append(build_activation(arch["activation"]))
            # Skip only when input and output dims match
            self._skip_flags.append(
                arch["use_skip"] and dims[i] == dims[i+1]
            )

        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (fc, bn, act) in enumerate(zip(self.fcs, self.bns, self.acts)):
            residual = x
            x = self.dropout(act(bn(fc(x))))
            if self._skip_flags[i]:
                x = x + residual
        return self.head(x)


# Raw CNN output dims (before quantum projection)
_cnn_feat_dims = {
    "convnext_base":      1024,
    "densenet161":        2208,
    "googlenet":          1024,
    "mobilenet_v3_large": 1280,   # FIX: classifier[0] expands 960→1280 before Identity
    "resnet50":           2048,
    "shufflenet_v2":      1024,
}


class QEFSFeatureExtractor(nn.Module):
    """Wraps backbone + QuantumInspiredLayer as a 512-dim feature extractor.

    When GNAS runs on a QEFS-enhanced model, the CNN backbone is first
    projected through the learned quantum layer to produce 512-dim features.
    This class exposes that combined extractor so NAS can attach a
    SearchableHead on top, and gradients flow through both layers.
    """
    def __init__(self, qcnn: QCNNHead):
        super().__init__()
        self.backbone = qcnn.backbone   # CNN feature extractor (Identity head)
        self.q_layer  = qcnn.q_layer   # QuantumInspiredLayer

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        return self.q_layer(f)          # → (B, QEFS_FEATURE_DIM=512)


# ═══════════════════════════════════════════════════════════════════════════
# Model with searchable head
# ═══════════════════════════════════════════════════════════════════════════
def build_model_with_arch(model_name: str, arch: dict,
                           pretrained_path: str = None,
                           qefs_path: str = None) -> nn.Module:
    """Load QEFS-enhanced backbone (or baseline CNN) + attach searchable head.

    Priority: if qefs_path exists → use backbone + QuantumInspiredLayer (512-dim)
              else                 → use baseline CNN backbone (native dim)
    """
    if qefs_path and os.path.exists(qefs_path):
        # ── QEFS backbone: CNN + QuantumInspiredLayer → 512-dim ──────────
        raw_dim  = _cnn_feat_dims[model_name]
        base     = get_model(model_name).to(DEVICE)
        base_ext = get_feature_extractor(base, model_name)
        qcnn     = QCNNHead(base_ext, raw_dim, QEFS_FEATURE_DIM).to(DEVICE)
        ckpt     = torch.load(qefs_path, map_location=DEVICE, weights_only=False)
        state    = ckpt.get("model_state", ckpt)   # handles both formats
        qcnn.load_state_dict(state)
        backbone = QEFSFeatureExtractor(qcnn)
        feat_dim = QEFS_FEATURE_DIM                 # 512
    else:
        # ── Baseline CNN backbone ─────────────────────────────────────────
        base = get_model(model_name).to(DEVICE)
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
            base.load_state_dict(ckpt["model_state"])
        backbone = get_feature_extractor(base, model_name)
        feat_dim = _cnn_feat_dims[model_name]

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head     = SearchableHead(feat_dim, arch)
        def forward(self, x):
            f = self.backbone(x)
            if isinstance(f, (tuple, list)): f = f[0]
            return self.head(f)

    return FullModel().to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
# Fitness evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_architecture(
    model_name: str, arch: dict, pretrained_path: str,
    train_loader, val_loader, train_df: pd.DataFrame,
    n_epochs: int = GNAS_EVAL_EPOCHS,
    qefs_path: str = None
) -> float:
    """Train candidate architecture for n_epochs, return val accuracy."""
    model = build_model_with_arch(model_name, arch, pretrained_path, qefs_path)

    # Freeze backbone, only train head
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False

    cw        = compute_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    lr        = LEARNING_RATE * arch["lr_mult"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    for ep in range(n_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, scaler)

    _, val_acc, _, _ = evaluate(model, val_loader, criterion)
    del model; torch.cuda.empty_cache()
    return float(val_acc)


# ═══════════════════════════════════════════════════════════════════════════
# Genetic Algorithm
# ═══════════════════════════════════════════════════════════════════════════
class GeneticNAS:
    def __init__(self, pop_size: int = GNAS_POP_SIZE,
                 n_generations: int = GNAS_GENERATIONS,
                 p_crossover: float = GNAS_CROSSOVER_RATE,
                 p_mutation:  float = GNAS_MUTATION_RATE,
                 tournament_k: int  = GNAS_TOURNAMENT_K):
        self.pop_size     = pop_size
        self.n_gen        = n_generations
        self.p_c          = p_crossover
        self.p_m          = p_mutation
        self.k            = tournament_k

        # Population of chromosomes
        self.population = [random_chromosome() for _ in range(pop_size)]
        self.fitness    = [0.0] * pop_size
        self.best_chrom = None
        self.best_fit   = -np.inf
        self.gen_best   = []   # best fitness per generation
        self.gen_mean   = []   # mean fitness per generation

    def tournament_select(self) -> list:
        candidates = random.sample(range(self.pop_size), self.k)
        winner     = max(candidates, key=lambda i: self.fitness[i])
        return copy.deepcopy(self.population[winner])

    def single_point_crossover(self, p1: list, p2: list) -> tuple:
        if random.random() < self.p_c:
            pt = random.randint(1, CHROM_LEN - 1)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        else:
            c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        return c1, c2

    def bit_flip_mutation(self, chrom: list) -> list:
        return [1 - b if random.random() < self.p_m else b for b in chrom]

    def evolve(self, fitness_fn) -> tuple:
        """
        Run GA.
        fitness_fn: callable(chromosome) → float (val accuracy)
        Returns: (best_chromosome, best_fitness, gen_best_history, gen_mean_history)
        """
        print(f"\n  [GA] Evaluating initial population ({self.pop_size} individuals) …")
        for i in range(self.pop_size):
            self.fitness[i] = fitness_fn(self.population[i])
            print(f"    Individual {i+1:2d}/{self.pop_size}: "
                  f"fit={self.fitness[i]:.4f}  "
                  f"arch={decode_chromosome(self.population[i])}")

        best_idx = int(np.argmax(self.fitness))
        self.best_chrom = copy.deepcopy(self.population[best_idx])
        self.best_fit   = self.fitness[best_idx]
        self.gen_best.append(self.best_fit)
        self.gen_mean.append(float(np.mean(self.fitness)))

        for gen in range(1, self.n_gen + 1):
            print(f"\n  [GA] Generation {gen}/{self.n_gen} "
                  f"(best so far: {self.best_fit:.4f}) …")
            new_pop = []

            # Elitism: carry over top-2
            sorted_idx = np.argsort(self.fitness)[::-1]
            new_pop.extend([copy.deepcopy(self.population[sorted_idx[0]]),
                            copy.deepcopy(self.population[sorted_idx[1]])])
            new_fit = [self.fitness[sorted_idx[0]], self.fitness[sorted_idx[1]]]

            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select()
                p2 = self.tournament_select()
                c1, c2 = self.single_point_crossover(p1, p2)
                c1 = self.bit_flip_mutation(c1)
                c2 = self.bit_flip_mutation(c2)

                f1 = fitness_fn(c1)
                f2 = fitness_fn(c2)
                print(f"    Child: fit={f1:.4f}  arch={decode_chromosome(c1)}")
                new_pop.append(c1); new_fit.append(f1)

                if len(new_pop) < self.pop_size:
                    print(f"    Child: fit={f2:.4f}  arch={decode_chromosome(c2)}")
                    new_pop.append(c2); new_fit.append(f2)

            self.population = new_pop
            self.fitness    = new_fit

            gen_best_idx = int(np.argmax(self.fitness))
            if self.fitness[gen_best_idx] > self.best_fit:
                self.best_fit   = self.fitness[gen_best_idx]
                self.best_chrom = copy.deepcopy(self.population[gen_best_idx])

            self.gen_best.append(self.best_fit)
            self.gen_mean.append(float(np.mean(self.fitness)))
            print(f"  [GA] Gen {gen} best={self.best_fit:.4f}  mean={self.gen_mean[-1]:.4f}")

        return self.best_chrom, self.best_fit, self.gen_best, self.gen_mean


# ═══════════════════════════════════════════════════════════════════════════
# Full NAS pipeline for one model
# ═══════════════════════════════════════════════════════════════════════════
def run_nas_on_model(model_name: str, train_df, val_df, test_df, dirs) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Genetic NAS: {model_name}")
    print(f"{'─'*60}")

    # Prefer QEFS-enhanced checkpoint; fall back to baseline CNN
    qefs_path = os.path.join(MODELS_DIR, f"qefs_{model_name}_best.pth")
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    use_qefs  = os.path.exists(qefs_path)

    if not use_qefs and not os.path.exists(ckpt_path):
        print(f"  [!] No checkpoint for {model_name}. Run 03/04 first. Skipping.")
        return {}

    feat_dim = QEFS_FEATURE_DIM if use_qefs else _cnn_feat_dims[model_name]
    print(f"  Backbone source: {'QEFS (512-dim)' if use_qefs else 'baseline CNN'}")

    # Build loaders
    train_ds = MultiDirDataset(train_df, dirs, get_train_transform())
    val_ds   = MultiDirDataset(val_df,   dirs, get_val_transform())
    test_ds  = MultiDirDataset(test_df,  dirs, get_val_transform())
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ld   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Fitness function closure
    def fitness_fn(chrom: list) -> float:
        arch = decode_chromosome(chrom)
        return evaluate_architecture(
            model_name, arch, ckpt_path,
            train_ld, val_ld, train_df,
            n_epochs=GNAS_EVAL_EPOCHS,
            qefs_path=qefs_path if use_qefs else None
        )

    # Run GA
    ga = GeneticNAS(
        pop_size    = GNAS_POP_SIZE,
        n_generations = GNAS_GENERATIONS,
        p_crossover = GNAS_CROSSOVER_RATE,
        p_mutation  = GNAS_MUTATION_RATE,
        tournament_k = GNAS_TOURNAMENT_K,
    )
    best_chrom, best_fit, gen_best, gen_mean = ga.evolve(fitness_fn)
    best_arch = decode_chromosome(best_chrom)

    print(f"\n  Best architecture found: {best_arch}")
    print(f"  Best validation accuracy: {best_fit:.4f}")

    # ── Full training of best architecture ────────────────────────────────
    print(f"\n  Full training of best architecture for {NUM_EPOCHS} epochs …")
    model = build_model_with_arch(model_name, best_arch, ckpt_path,
                                  qefs_path=qefs_path if use_qefs else None)

    # Unfreeze everything for full fine-tune
    for p in model.parameters():
        p.requires_grad = True

    cw        = compute_class_weights(train_df).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    lr        = LEARNING_RATE * best_arch["lr_mult"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)
    scaler    = GradScaler(device="cuda", enabled=USE_AMP)
    es        = EarlyStopping(patience=PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0
    best_path = os.path.join(MODELS_DIR, f"gnas_{model_name}_best.pth")

    t_train = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, criterion, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_ld, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc; best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "arch":        best_arch,
                "model_name":  model_name,
                "epoch":       epoch,
                "uses_qefs":   use_qefs,           # consumed by 06_continual_learning
                "feat_dim":    feat_dim,            # backbone output dim (512 or CNN native)
            }, best_path)

        print(f"  [{epoch:3d}/{NUM_EPOCHS}] loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc {tr_acc:.4f}/{va_acc:.4f}")
        if es(va_acc):
            print(f"  Early stopping at epoch {epoch}.")
            break

    training_time = time.time() - t_train

    # Load best and test
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    criterion_plain = nn.CrossEntropyLoss()
    ts_loss, ts_acc, preds, labels = evaluate(model, test_ld, criterion_plain)

    t0 = time.time()
    with torch.no_grad():
        for imgs, _ in test_ld:
            _ = model(imgs.to(DEVICE))
    inf_time = (time.time() - t0) / len(test_ld.dataset)

    baseline = load_results(model_name) or {}
    results = {
        "model_name":        model_name,
        "method":            "GeneticNAS",
        "best_architecture": best_arch,
        "best_chromosome":   best_chrom,
        "nas_best_val_acc":  best_fit,
        "baseline_test_acc": baseline.get("test_acc", 0),
        "gnas_val_acc":      best_val_acc,
        "gnas_test_acc":     float(ts_acc),
        "gnas_f1_macro":     0.0,  # filled below
        "best_epoch":        best_epoch,
        "num_epochs":        NUM_EPOCHS,
        "training_time_s":   training_time,
        "inference_time_per_img_s": inf_time,
        "history":           history,
        "ga_gen_best":       gen_best,
        "ga_gen_mean":       gen_mean,
        "preds":             preds.tolist(),
        "labels":            labels.tolist(),
    }
    metrics = compute_full_metrics(preds, labels, f"gnas_{model_name}")
    results.update(metrics)
    results["gnas_f1_macro"] = metrics["f1_macro"]

    # ── Plots ──────────────────────────────────────────────────────────────
    # GA convergence
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = range(len(gen_best))
    ax.plot(gens, gen_best, "b-o", label="Best", linewidth=2, markersize=4)
    ax.plot(gens, gen_mean, "r--s", label="Mean", linewidth=1.5, markersize=4)
    ax.set(title=f"Genetic NAS Convergence — {model_name}",
           xlabel="Generation", ylabel="Validation Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"gnas_convergence_{model_name}.png"), dpi=150)
    plt.close()

    # Training curves
    plot_training_curves(history, model_name, tag="gnas")
    plot_confusion_matrix(np.array(results["confusion_matrix"]), model_name, tag="gnas")
    plot_class_accuracies(results["per_class_acc"], model_name, tag="gnas")

    # Architecture visualisation
    _vis_architecture(model_name, best_arch)

    save_results(results, f"gnas_{model_name}")
    del model; torch.cuda.empty_cache()

    print(f"\n  ✓ {model_name} GNAS: test_acc={ts_acc:.4f}  "
          f"(baseline={baseline.get('test_acc', 0):.4f})")
    return results


def _vis_architecture(model_name: str, arch: dict):
    """Simple text diagram of the best-found architecture."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    text = (f"Model:      {model_name}\n"
            f"FC Layers:  {arch['n_layers']}\n"
            f"Hidden dim: {arch['hidden_dim']}\n"
            f"Dropout:    {arch['dropout']}\n"
            f"Activation: {arch['activation']}\n"
            f"BatchNorm:  {arch['use_bn']}\n"
            f"Skip Conn.: {arch['use_skip']}\n"
            f"LR mult:    {arch['lr_mult']}×")
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
            fontsize=14, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title(f"Best Architecture — {model_name} (Genetic NAS)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"gnas_arch_{model_name}.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Comparison plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_gnas_comparison(all_gnas: list):
    if not all_gnas: return
    names    = [r["model_name"] for r in all_gnas]
    baseline = [r.get("baseline_test_acc", 0) for r in all_gnas]
    gnas     = [r.get("gnas_test_acc",    0) for r in all_gnas]
    f1_base  = [load_results(n).get("f1_macro", 0) if load_results(n) else 0 for n in names]
    f1_gnas  = [r.get("gnas_f1_macro", 0) for r in all_gnas]

    x = np.arange(len(names)); w = 0.35
    palette = sns.color_palette("tab10", len(names))

    # ── Accuracy comparison ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(x - w/2, baseline, w, label="Baseline", color="#4e79a7", edgecolor="black")
    axes[0].bar(x + w/2, gnas,     w, label="GNAS",     color="#59a14f", edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set(title="Test Accuracy: GNAS vs Baseline", ylabel="Accuracy", ylim=(0, 1.1))
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - w/2, f1_base, w, label="Baseline F1", color="#4e79a7", edgecolor="black")
    axes[1].bar(x + w/2, f1_gnas, w, label="GNAS F1",     color="#59a14f", edgecolor="black")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set(title="F1-Macro: GNAS vs Baseline", ylabel="F1 Macro", ylim=(0, 1.1))
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Genetic NAS — Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gnas_accuracy_comparison.png"), dpi=150)
    plt.close()

    # ── Accuracy delta ────────────────────────────────────────────────────
    gains = [g - b for g, b in zip(gnas, baseline)]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if g >= 0 else "#F44336" for g in gains]
    ax.bar(names, gains, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Accuracy Gain: GNAS vs Baseline", ylabel="Δ Test Accuracy")
    ax.set_xticklabels(names, rotation=30, ha="right")
    for i, g in enumerate(gains):
        ax.text(i, g + (0.001 if g >= 0 else -0.003),
                f"{g:+.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gnas_accuracy_gain.png"), dpi=150)
    plt.close()

    # ── Per-model GA convergence grid ─────────────────────────────────────
    n_models = len(all_gnas)
    cols = 3; rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    for ax, r in zip(axes.flat, all_gnas):
        gens = range(len(r["ga_gen_best"]))
        ax.plot(gens, r["ga_gen_best"], "b-o", markersize=4, label="Best")
        ax.plot(gens, r["ga_gen_mean"], "r--", markersize=3, label="Mean")
        ax.set(title=r["model_name"], xlabel="Generation", ylabel="Val Acc")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    for ax in axes.flat[n_models:]:
        ax.axis("off")
    plt.suptitle("GA Convergence Curves — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gnas_all_convergence.png"), dpi=150)
    plt.close()

    # ── Architecture hyperparameter radar ─────────────────────────────────
    # Compare chosen architectures across models
    fig, ax = plt.subplots(figsize=(14, 6))
    hp_names = list(CHROM_BITS.keys())
    hp_vals  = []
    for r in all_gnas:
        arch = r.get("best_architecture", {})
        row  = []
        for hp, (_, opts) in CHROM_BITS.items():
            val = arch.get(hp, opts[0])
            row.append(opts.index(val) if val in opts else 0)
        hp_vals.append(row)

    hp_matrix = np.array(hp_vals, dtype=float)
    # Normalise each column
    hp_matrix = hp_matrix / (hp_matrix.max(axis=0, keepdims=True) + 1e-8)

    sns.heatmap(hp_matrix, annot=True, fmt=".2f",
                xticklabels=hp_names, yticklabels=names,
                cmap="Blues", ax=ax, linewidths=0.5)
    ax.set(title="Best Architecture Hyperparameters (normalised rank)",
           xlabel="Hyperparameter", ylabel="Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gnas_arch_heatmap.png"), dpi=150)
    plt.close()

    print(f"\n[GNAS Plots] saved to {PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    set_seed()
    print("\n" + "=" * 60)
    print("  AgroDetect — Genetic NAS")
    print("=" * 60)

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
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    all_gnas = []
    for model_name in MODEL_NAMES:
        res_path = os.path.join(RESULTS_DIR, f"gnas_{model_name}.json")
        if os.path.exists(res_path):
            print(f"\n  [Skip] GNAS for {model_name} already done.")
            r = load_results(f"gnas_{model_name}")
            if r: all_gnas.append(r)
            continue
        r = run_nas_on_model(model_name, train_df, val_df, test_df, dirs)
        if r: all_gnas.append(r)

    plot_gnas_comparison(all_gnas)

    print("\n" + "=" * 75)
    print(f"  {'Model':<25} {'Baseline':>10} {'GNAS':>10} {'ΔAcc':>10} {'F1-GNAS':>10}")
    print("  " + "-" * 73)
    for r in all_gnas:
        delta = r.get("gnas_test_acc", 0) - r.get("baseline_test_acc", 0)
        print(f"  {r['model_name']:<25} "
              f"{r.get('baseline_test_acc', 0):>10.4f} "
              f"{r.get('gnas_test_acc',    0):>10.4f} "
              f"{delta:>+10.4f} "
              f"{r.get('gnas_f1_macro',   0):>10.4f}")
    print("=" * 75)

    print("\n✓ Genetic NAS complete.\n")
