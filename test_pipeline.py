"""
test_pipeline.py — AgroDetect Pipeline Smoke Test
===================================================
Validates every stage of the pipeline without running full training.
Creates tiny synthetic checkpoints so the QEFS → GNAS → CL loading
chain can be exercised even if previous stages haven't run yet.

Run:   python test_pipeline.py
Time:  ~3–8 minutes on GPU (mostly model download on first run)
Pass criterion: "ALL TESTS PASSED" printed at the end with no FAIL lines.
"""

import os, sys, json, time, tempfile, traceback, warnings, importlib.util
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colours for terminal output ────────────────────────────────────────────
GREEN  = "\033[92m"; RED  = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"
BOLD   = "\033[1m"

results_log: list = []

def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")
    results_log.append(("PASS", msg))

def fail(msg, exc=None):
    print(f"  {RED}✗{RESET} {msg}")
    if exc:
        print(f"      {RED}{type(exc).__name__}: {exc}{RESET}")
        traceback.print_exc()
    results_log.append(("FAIL", msg))

def section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — Imports & hardware
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 1 · Imports & Hardware")

try:
    import numpy as np; ok("numpy")
except Exception as e: fail("numpy", e)

try:
    import pandas as pd; ok("pandas")
except Exception as e: fail("pandas", e)

try:
    import torch; ok(f"torch {torch.__version__}")
except Exception as e: fail("torch", e); sys.exit(1)

try:
    import torchvision; ok(f"torchvision {torchvision.__version__}")
except Exception as e: fail("torchvision", e)

try:
    import PIL; ok("Pillow")
except Exception as e: fail("Pillow", e)

try:
    import sklearn; ok("scikit-learn")
except Exception as e: fail("scikit-learn", e)

try:
    import matplotlib; ok("matplotlib")
except Exception as e: fail("matplotlib", e)

try:
    import seaborn; ok("seaborn")
except Exception as e: fail("seaborn", e)

try:
    import shap; ok("shap")
except Exception as e: fail("shap (optional — SHAP plots will be skipped)", e)

try:
    import scipy; ok("scipy")
except Exception as e: fail("scipy", e)

try:
    import psutil; ok("psutil")
except Exception as e: fail("psutil", e)

# CUDA
try:
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"CUDA available: {dev} ({vram:.1f} GB VRAM)")
    else:
        fail("CUDA not available — training will run on CPU (very slow)")
except Exception as e: fail("CUDA check", e)

# AMP
try:
    from torch.amp import GradScaler, autocast
    scaler_test = GradScaler(device="cuda", enabled=False)
    ok("torch.amp (PyTorch ≥ 2.4 API)")
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        ok("torch.cuda.amp (fallback API)")
    except Exception as e: fail("AMP import", e)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Config & project imports
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 2 · Config & Project Imports")

try:
    from config import (
        DATASET_DIR, TRAIN_CSV, TRAIN_IMAGES, CLEANED_CSV, BALANCED_CSV,
        OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR,
        NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX,
        IMAGE_SIZE, SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
        DROPOUT, PATIENCE, LABEL_SMOOTHING, USE_AMP, DEVICE,
        QEFS_FEATURE_DIM, MODEL_NAMES,
    )
    ok(f"config.py  (device={DEVICE})")
except Exception as e: fail("config.py", e); sys.exit(1)

try:
    from utils import (
        set_seed, get_model, get_feature_extractor,
        PlantDiseaseDataset, split_dataframe, compute_class_weights,
        get_train_transform, get_val_transform,
        train_one_epoch, evaluate, EarlyStopping,
        compute_full_metrics, bootstrap_confidence_interval,
        profile_memory, compute_and_save_shap, grad_cam_heatmap,
        plot_training_curves, plot_confusion_matrix, plot_class_accuracies,
        save_results, load_results, full_train,
    )
    ok("utils.py")
except Exception as e: fail("utils.py", e); sys.exit(1)

# Dataset directory
if os.path.isdir(DATASET_DIR):
    ok(f"Dataset directory found: {DATASET_DIR}")
else:
    fail(f"Dataset directory MISSING: {DATASET_DIR}"); sys.exit(1)

# CSV files
csv_to_use = None
for csv_path, label in [(BALANCED_CSV, "balanced_train.csv"),
                         (CLEANED_CSV,  "cleaned_train.csv"),
                         (TRAIN_CSV,    "train.csv")]:
    if os.path.exists(csv_path):
        ok(f"{label} found")
        csv_to_use = csv_path
        break
if csv_to_use is None:
    fail("No CSV file found (train.csv / cleaned_train.csv / balanced_train.csv)")
    sys.exit(1)

# Image directory
if os.path.isdir(TRAIN_IMAGES):
    subdirs = [d for d in os.listdir(TRAIN_IMAGES)
               if os.path.isdir(os.path.join(TRAIN_IMAGES, d))]
    ok(f"TRAIN_IMAGES found with {len(subdirs)} subdirectories: {subdirs[:3]}…")
else:
    fail(f"TRAIN_IMAGES directory missing: {TRAIN_IMAGES}"); sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Dataset loading & image resolution
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 3 · Dataset Loading & Image Access")

set_seed()
try:
    df_full = pd.read_csv(csv_to_use)
    ok(f"Loaded CSV: {len(df_full):,} rows, columns={list(df_full.columns)}")
except Exception as e: fail("Reading CSV", e); sys.exit(1)

# Verify image_id + label columns exist
required_cols = {"image_id", "label"}
missing_cols  = required_cols - set(df_full.columns)
if missing_cols:
    fail(f"CSV missing columns: {missing_cols}"); sys.exit(1)
ok("CSV has required columns: image_id, label")

# Check label distribution
label_counts = df_full["label"].value_counts()
ok(f"Labels: {dict(label_counts.head(3))} …")

# Try loading 10 images via PlantDiseaseDataset
try:
    sample_df = df_full.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(1, len(g)), random_state=42)
    ).reset_index(drop=True)
    aug_dir = os.path.join(DATASET_DIR, "aug_images")
    dirs = [TRAIN_IMAGES] + ([aug_dir] if os.path.isdir(aug_dir) else [])

    # Build a small Dataset manually using the MultiDirDataset approach
    found, missing = 0, 0
    for _, row in sample_df.iterrows():
        label = row.get("label", "")
        located = False
        for d in dirs:
            for p in [os.path.join(d, label, row["image_id"]),
                      os.path.join(d, row["image_id"])]:
                if os.path.exists(p):
                    located = True
                    break
            if located: break
        if located: found += 1
        else: missing += 1

    if missing == 0:
        ok(f"All {found} sampled images located (dual-layout resolver working)")
    else:
        fail(f"{missing}/{found+missing} images not found — check Dataset folder structure")
except Exception as e: fail("Image resolution check", e)

# DataLoader round-trip (1 batch)
try:
    from torch.utils.data import DataLoader

    class _MultiDirDS(PlantDiseaseDataset):
        def __init__(self, df, dirs, transform=None):
            super().__init__(df, dirs[0], transform); self.dirs = dirs
        def __getitem__(self, idx):
            from PIL import Image as _PIL
            row = self.df.iloc[idx]; label = row.get("label", "")
            for d in self.dirs:
                for p in [os.path.join(d, label, row["image_id"]),
                          os.path.join(d, row["image_id"])]:
                    if os.path.exists(p):
                        img = _PIL.open(p).convert("RGB")
                        if self.transform: img = self.transform(img)
                        return img, self.class_to_idx[label]
            raise FileNotFoundError(row["image_id"])

    mini_df = df_full.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(2, len(g)), random_state=42)
    ).reset_index(drop=True)
    mini_ds = _MultiDirDS(mini_df, dirs, get_val_transform())
    mini_ld = DataLoader(mini_ds, batch_size=4, shuffle=False, num_workers=0)
    imgs, labels = next(iter(mini_ld))
    assert imgs.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE), f"Unexpected shape {imgs.shape}"
    ok(f"DataLoader: batch shape {tuple(imgs.shape)}, labels {labels.tolist()[:4]}")
except Exception as e: fail("DataLoader round-trip", e)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — All 6 CNN model forward passes
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 4 · CNN Model Construction & Forward Passes")

dummy_batch = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
model_checkpoints = {}   # model_name → path (populated as checkpoints are found)

for mname in MODEL_NAMES:
    try:
        m = get_model(mname, pretrained=False).to(DEVICE)
        m.eval()
        with torch.no_grad():
            out = m(dummy_batch)
            if isinstance(out, (tuple, list)): out = out[0]
        assert out.shape == (2, NUM_CLASSES), f"Output shape {out.shape}"

        # Check if a real checkpoint exists
        ckpt_path = os.path.join(MODELS_DIR, f"{mname}_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            model_checkpoints[mname] = ckpt_path
            ok(f"{mname:<25} ✓ forward pass  ✓ checkpoint loaded")
        else:
            ok(f"{mname:<25} ✓ forward pass  (no checkpoint yet — run 03 first)")
        del m
    except Exception as e:
        fail(f"{mname}", e)

torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — QEFS classes
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 5 · QEFS Classes (04_qefs.py)")

try:
    _spec = importlib.util.spec_from_file_location(
        "qefs_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_qefs.py")
    )
    _qefs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_qefs)
    ok("04_qefs.py imported via importlib")

    QuantumInspiredLayer = _qefs.QuantumInspiredLayer
    QCNNHead             = _qefs.QCNNHead
    get_backbone_dim     = _qefs.get_backbone_dim
except Exception as e:
    fail("04_qefs.py importlib load", e)
    QuantumInspiredLayer = QCNNHead = get_backbone_dim = None

if QCNNHead is not None:
    try:
        # QuantumInspiredLayer standalone
        ql = QuantumInspiredLayer(in_dim=2048, out_dim=QEFS_FEATURE_DIM).to(DEVICE)
        feat = torch.randn(2, 2048).to(DEVICE)
        q_out = ql(feat)
        assert q_out.shape == (2, QEFS_FEATURE_DIM), f"QL output {q_out.shape}"
        ok(f"QuantumInspiredLayer: (2,2048) → (2,{QEFS_FEATURE_DIM})")
    except Exception as e: fail("QuantumInspiredLayer forward", e)

    try:
        # QCNNHead: backbone (dummy) → QL → head
        import torch.nn as nn
        dummy_backbone = nn.Identity()
        qcnn = QCNNHead(dummy_backbone, feat_dim=2048, q_dim=QEFS_FEATURE_DIM).to(DEVICE)
        dummy_feat = torch.randn(2, 2048).to(DEVICE)
        # Temporarily patch backbone to output 2048-dim directly
        qcnn.backbone = nn.Identity()
        out = qcnn(dummy_feat)
        assert out.shape == (2, NUM_CLASSES), f"QCNNHead output {out.shape}"
        ok(f"QCNNHead: (2,2048) → (2,{NUM_CLASSES})")
    except Exception as e: fail("QCNNHead forward", e)

    # Save a synthetic QEFS checkpoint for the GNAS test below
    try:
        import torch.nn as nn
        base = get_model("resnet50", pretrained=False).to(DEVICE)
        bext = get_feature_extractor(base, "resnet50")
        raw_dim = get_backbone_dim("resnet50")  # 2048
        qcnn_real = QCNNHead(bext, raw_dim, QEFS_FEATURE_DIM).to(DEVICE)
        _qefs_ckpt_path = os.path.join(MODELS_DIR, "_test_qefs_resnet50_best.pth")
        torch.save({
            "model_state": qcnn_real.state_dict(),
            "model_name":  "resnet50",
            "feat_dim":    raw_dim,
            "q_dim":       QEFS_FEATURE_DIM,
        }, _qefs_ckpt_path)
        ok(f"Synthetic QEFS checkpoint saved: {os.path.basename(_qefs_ckpt_path)}")
        del qcnn_real
        torch.cuda.empty_cache()
    except Exception as e: fail("Save synthetic QEFS checkpoint", e); _qefs_ckpt_path = None


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Genetic NAS classes + QEFS→GNAS loading chain
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 6 · Genetic NAS Classes (05_genetic_nas.py)")

try:
    _spec5 = importlib.util.spec_from_file_location(
        "gnas_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_genetic_nas.py")
    )
    _gnas = importlib.util.module_from_spec(_spec5)
    _spec5.loader.exec_module(_gnas)
    ok("05_genetic_nas.py imported via importlib")

    SearchableHead      = _gnas.SearchableHead
    decode_chromosome   = _gnas.decode_chromosome
    random_chromosome   = _gnas.random_chromosome
    build_model_with_arch = _gnas.build_model_with_arch
    QEFSFeatureExtractor  = _gnas.QEFSFeatureExtractor
    _cnn_feat_dims        = _gnas._cnn_feat_dims
except Exception as e:
    fail("05_genetic_nas.py importlib load", e)
    SearchableHead = decode_chromosome = random_chromosome = None
    build_model_with_arch = QEFSFeatureExtractor = _cnn_feat_dims = None

if SearchableHead is not None:
    # Test chromosome decode
    try:
        import torch.nn as nn
        chrom = random_chromosome()
        arch  = decode_chromosome(chrom)
        assert set(arch.keys()) == {"n_layers","hidden_dim","dropout","activation",
                                    "use_bn","use_skip","lr_mult"}, f"arch keys: {arch.keys()}"
        ok(f"Chromosome decode: {arch}")
    except Exception as e: fail("Chromosome decode", e)

    # Test SearchableHead with a few arch configurations
    test_archs = [
        {"n_layers":1,"hidden_dim":256,"dropout":0.3,"activation":"relu",
         "use_bn":False,"use_skip":False,"lr_mult":1.0},
        {"n_layers":2,"hidden_dim":512,"dropout":0.4,"activation":"gelu",
         "use_bn":True,"use_skip":True,"lr_mult":1.0},
        {"n_layers":3,"hidden_dim":128,"dropout":0.2,"activation":"silu",
         "use_bn":True,"use_skip":True,"lr_mult":0.5},
    ]
    for arch_t in test_archs:
        try:
            import torch.nn as nn
            head = SearchableHead(QEFS_FEATURE_DIM, arch_t).to(DEVICE)
            x    = torch.randn(2, QEFS_FEATURE_DIM).to(DEVICE)
            out  = head(x)
            assert out.shape == (2, NUM_CLASSES), f"head output {out.shape}"
            ok(f"SearchableHead n_layers={arch_t['n_layers']} use_skip={arch_t['use_skip']}: OK")
        except Exception as e:
            fail(f"SearchableHead arch={arch_t}", e)

    # Test build_model_with_arch with QEFS backbone (needs _test checkpoint)
    if _qefs_ckpt_path is not None and os.path.exists(_qefs_ckpt_path):
        try:
            arch_t = {"n_layers":1,"hidden_dim":256,"dropout":0.3,"activation":"relu",
                      "use_bn":False,"use_skip":False,"lr_mult":1.0}
            # Use our synthetic QEFS checkpoint
            model = build_model_with_arch("resnet50", arch_t,
                                          pretrained_path=None,
                                          qefs_path=_qefs_ckpt_path)
            model.eval()
            with torch.no_grad():
                out = model(dummy_batch)
            assert out.shape == (2, NUM_CLASSES), f"full model output {out.shape}"
            ok("build_model_with_arch (QEFS backbone): forward OK")

            # Check backbone is QEFSFeatureExtractor
            assert isinstance(model.backbone, QEFSFeatureExtractor), \
                f"Expected QEFSFeatureExtractor, got {type(model.backbone)}"
            ok("build_model_with_arch backbone is QEFSFeatureExtractor ✓")

            # Save a synthetic GNAS checkpoint (uses_qefs=True)
            _gnas_ckpt_path = os.path.join(MODELS_DIR, "_test_gnas_resnet50_best.pth")
            torch.save({
                "model_state": model.state_dict(),
                "arch":        arch_t,
                "model_name":  "resnet50",
                "epoch":       1,
                "uses_qefs":   True,
                "feat_dim":    QEFS_FEATURE_DIM,
            }, _gnas_ckpt_path)
            ok(f"Synthetic GNAS checkpoint (uses_qefs=True) saved")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            fail("build_model_with_arch (QEFS backbone)", e)
            _gnas_ckpt_path = None
    else:
        # Also test without QEFS (baseline CNN backbone)
        try:
            arch_t = {"n_layers":1,"hidden_dim":256,"dropout":0.3,"activation":"relu",
                      "use_bn":False,"use_skip":False,"lr_mult":1.0}
            ckpt_p = model_checkpoints.get("resnet50")
            model  = build_model_with_arch("resnet50", arch_t, pretrained_path=ckpt_p,
                                           qefs_path=None)
            model.eval()
            with torch.no_grad():
                out = model(dummy_batch)
            assert out.shape == (2, NUM_CLASSES)
            ok("build_model_with_arch (baseline backbone): forward OK")
            _gnas_ckpt_path = os.path.join(MODELS_DIR, "_test_gnas_resnet50_best.pth")
            torch.save({
                "model_state": model.state_dict(),
                "arch":        arch_t,
                "model_name":  "resnet50",
                "epoch":       1,
                "uses_qefs":   False,
                "feat_dim":    _cnn_feat_dims["resnet50"],
            }, _gnas_ckpt_path)
            ok("Synthetic GNAS checkpoint (uses_qefs=False) saved")
            del model; torch.cuda.empty_cache()
        except Exception as e:
            fail("build_model_with_arch (baseline backbone)", e)
            _gnas_ckpt_path = None


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Continual Learning: CLModel loading chain
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 7 · Continual Learning — CLModel Checkpoint Chain (06_continual_learning.py)")

# Temporarily rename test checkpoints to the real expected names
import shutil
_renames = []
if _gnas_ckpt_path and os.path.exists(_gnas_ckpt_path):
    real_gnas = os.path.join(MODELS_DIR, "gnas_resnet50_best.pth")
    if not os.path.exists(real_gnas):
        shutil.copy(_gnas_ckpt_path, real_gnas)
        _renames.append(real_gnas)

if _qefs_ckpt_path and os.path.exists(_qefs_ckpt_path):
    real_qefs = os.path.join(MODELS_DIR, "qefs_resnet50_best.pth")
    if not os.path.exists(real_qefs):
        shutil.copy(_qefs_ckpt_path, real_qefs)
        _renames.append(real_qefs)

try:
    _spec6 = importlib.util.spec_from_file_location(
        "cl_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_continual_learning.py")
    )
    _cl = importlib.util.module_from_spec(_spec6)
    _spec6.loader.exec_module(_cl)
    ok("06_continual_learning.py imported via importlib")
    CLModel = _cl.CLModel
except Exception as e:
    fail("06_continual_learning.py importlib load", e)
    CLModel = None

if CLModel is not None:
    # Test CLModel loading (GNAS → QEFS → baseline fallback)
    for scenario, desc in [
        ("gnas",     "Load from GNAS checkpoint (top priority)"),
        ("qefs",     "Load from QEFS checkpoint (fallback)"),
        ("baseline", "Load from baseline CNN (last resort)"),
    ]:
        # Temporarily hide higher-priority checkpoints to test fallback
        gnas_p = os.path.join(MODELS_DIR, "gnas_resnet50_best.pth")
        qefs_p = os.path.join(MODELS_DIR, "qefs_resnet50_best.pth")
        base_p = os.path.join(MODELS_DIR, "resnet50_best.pth")

        stash = {}
        if scenario == "qefs":
            if os.path.exists(gnas_p):
                shutil.move(gnas_p, gnas_p + ".bak"); stash[gnas_p] = gnas_p + ".bak"
        elif scenario == "baseline":
            for p in [gnas_p, qefs_p]:
                if os.path.exists(p):
                    shutil.move(p, p + ".bak"); stash[p] = p + ".bak"

        try:
            cl_model = CLModel(backbone_name="resnet50")
            cl_model.eval()
            with torch.no_grad():
                out = cl_model(dummy_batch)
            assert out.shape == (2, NUM_CLASSES), f"CLModel output {out.shape}"
            ok(f"CLModel ({desc}): feat_dim={cl_model.feat_dim}  output={tuple(out.shape)}")
        except Exception as e:
            fail(f"CLModel ({desc})", e)
        finally:
            # Restore stashed files
            for dst, src in stash.items():
                if os.path.exists(src):
                    shutil.move(src, dst)

        del cl_model
        torch.cuda.empty_cache()

    # Test replay buffer
    try:
        buf = _cl.ReplayBuffer(capacity=10)
        imgs_t  = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        labs_t  = torch.randint(0, NUM_CLASSES, (4,))
        logs_t  = torch.randn(4, NUM_CLASSES)
        buf.add(imgs_t, labs_t, logs_t)
        si, sl, sg = buf.sample(3)
        assert si.shape[0] == 3
        ok(f"ReplayBuffer: add 4, sample 3 → {tuple(si.shape)}")
    except Exception as e: fail("ReplayBuffer", e)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8 — One mini training step (real data, 1 epoch)
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 8 · Mini Training Step (1 epoch, shufflenet_v2)")

try:
    import torch.nn as nn
    from torch.utils.data import DataLoader

    mini_train = df_full.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(4, len(g)), random_state=1)
    ).reset_index(drop=True)
    mini_val = df_full.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(2, len(g)), random_state=2)
    ).reset_index(drop=True)

    tr_ds = _MultiDirDS(mini_train, dirs, get_train_transform())
    va_ds = _MultiDirDS(mini_val,   dirs, get_val_transform())
    tr_ld = DataLoader(tr_ds, batch_size=8,  shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=8,  shuffle=False, num_workers=0)

    try:
        from torch.amp import GradScaler, autocast
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast

    model = get_model("shufflenet_v2", pretrained=False).to(DEVICE)
    cw    = compute_class_weights(mini_train).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    opt   = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scl   = GradScaler(device="cuda", enabled=USE_AMP)

    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, tr_ld, opt, crit, scl)
    va_loss, va_acc, _, _ = evaluate(model, va_ld, crit)
    elapsed = time.time() - t0
    ok(f"1 epoch done in {elapsed:.1f}s  |  train_loss={tr_loss:.3f}  val_acc={va_acc:.3f}")
    del model; torch.cuda.empty_cache()
except Exception as e: fail("Mini training step", e)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9 — Script import / syntax check for all main scripts
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 9 · Syntax Check — All Pipeline Scripts")

scripts = [
    "01_data_cleaning.py",
    "02_data_balancing.py",
    "03_train_cnn_models.py",
    "04_qefs.py",
    "05_genetic_nas.py",
    "06_continual_learning.py",
    "07_comparison_plots.py",
]

for script in scripts:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if not os.path.exists(path):
        fail(f"{script} — file not found"); continue
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        compile(src, path, "exec")
        ok(f"{script} — syntax OK")
    except SyntaxError as e:
        fail(f"{script} — SyntaxError at line {e.lineno}: {e.msg}")
    except Exception as e:
        fail(f"{script}", e)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10 — Checkpoint loading for trained models
# ═══════════════════════════════════════════════════════════════════════════
section("TEST 10 · Existing Checkpoint Verification")

ckpt_status = {}
for mname in MODEL_NAMES:
    base_p  = os.path.join(MODELS_DIR, f"{mname}_best.pth")
    qefs_p  = os.path.join(MODELS_DIR, f"qefs_{mname}_best.pth")
    gnas_p  = os.path.join(MODELS_DIR, f"gnas_{mname}_best.pth")
    present = [("baseline", base_p), ("qefs", qefs_p), ("gnas", gnas_p)]
    for tag, p in present:
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                assert "model_state" in ckpt, f"'model_state' key missing in {p}"
                size_mb = os.path.getsize(p) / 1e6
                ok(f"{mname:<25} [{tag:8s}] valid checkpoint ({size_mb:.0f} MB)")
                ckpt_status[f"{mname}_{tag}"] = True
            except Exception as e:
                fail(f"{mname} [{tag}] checkpoint corrupt", e)
        else:
            print(f"  {YELLOW}–{RESET} {mname:<25} [{tag:8s}] not found (run scripts to generate)")

# CL checkpoints
for mname in MODEL_NAMES[:3]:
    cl_p = os.path.join(RESULTS_DIR, f"cl_gdumb_{mname}.json")  # example
    if os.path.exists(cl_p):
        ok(f"CL result found for {mname}")


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup test artifacts
# ═══════════════════════════════════════════════════════════════════════════
for p in [_qefs_ckpt_path if '_qefs_ckpt_path' in dir() else None,
          _gnas_ckpt_path if '_gnas_ckpt_path' in dir() else None]:
    if p and os.path.exists(p):
        os.remove(p)

for p in _renames:
    if os.path.exists(p) and "_test_" not in p:
        # These were copied from test files, only remove if they were freshly created
        pass  # Keep real ones; test copies were already removed above


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  SUMMARY{RESET}")
print(f"{BOLD}{'═'*60}{RESET}")

passed = [r for r in results_log if r[0] == "PASS"]
failed = [r for r in results_log if r[0] == "FAIL"]

print(f"  {GREEN}Passed: {len(passed)}{RESET}")
if failed:
    print(f"  {RED}Failed: {len(failed)}{RESET}")
    for _, msg in failed:
        print(f"    {RED}✗{RESET} {msg}")
    print(f"\n{RED}{BOLD}  SOME TESTS FAILED — see above for details.{RESET}")
else:
    print(f"\n{GREEN}{BOLD}  ALL TESTS PASSED ✓{RESET}")
    print(f"  Pipeline is ready. Run scripts in order:")
    print(f"    python 01_data_cleaning.py")
    print(f"    python 02_data_balancing.py")
    print(f"    python 03_train_cnn_models.py")
    print(f"    python 04_qefs.py")
    print(f"    python 05_genetic_nas.py")
    print(f"    python 06_continual_learning.py")
    print(f"    python 07_comparison_plots.py")

print()
