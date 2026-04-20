"""
config.py — Shared configuration for the AgroDetect pipeline.
All scripts import from here to ensure consistency.
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "Dataset")
TRAIN_CSV       = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES    = os.path.join(DATASET_DIR, "train_images")
TEST_IMAGES     = os.path.join(DATASET_DIR, "test_images")

CLEANED_CSV     = os.path.join(DATASET_DIR, "cleaned_train.csv")
BALANCED_CSV    = os.path.join(DATASET_DIR, "balanced_train.csv")

OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODELS_DIR      = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR        = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────
NUM_CLASSES = 10
CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGE_SIZE  = 224          # Input resolution for all models
SEED        = 42
VAL_SPLIT   = 0.15         # 15 % of training data → validation
TEST_SPLIT  = 0.10         # 10 % of training data → held-out test

# ── Training hyperparameters ───────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 8
PIN_MEMORY      = True
NUM_EPOCHS      = 60
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
MOMENTUM        = 0.9

# Scheduler
LR_SCHEDULER    = "cosine"   # "cosine" | "step" | "plateau"
T_MAX           = NUM_EPOCHS
ETA_MIN         = 1e-6

# Regularisation
DROPOUT         = 0.4
LABEL_SMOOTHING = 0.1

# Early stopping
PATIENCE        = 12
MIN_DELTA       = 1e-4

# Mixed precision
USE_AMP         = True       # FP16 via torch.cuda.amp

# ── Model names ────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "convnext_base",
    "densenet161",
    "googlenet",
    "mobilenet_v3_large",
    "resnet50",
    "shufflenet_v2",
]

# ── QEFS hyperparameters ───────────────────────────────────────────────────
QEFS_FEATURE_DIM    = 512        # dimension after quantum projection
QEFS_POP_SIZE       = 30         # HFSEA population size
QEFS_MAX_ITER       = 50         # HFSEA iterations
QEFS_ALPHA          = 0.5        # absorption coefficient (firefly)
QEFS_GAMMA          = 1.0        # light absorption coefficient
QEFS_BETA0          = 1.0        # attractiveness at r=0
QEFS_DELTA          = 0.97       # random walk step decay (swallow)
QEFS_FINETUNE_EPOCHS = 20        # quantum fine-tune epochs after HFSEA

# ── Genetic NAS hyperparameters ────────────────────────────────────────────
GNAS_POP_SIZE       = 20         # population size
GNAS_GENERATIONS    = 15         # number of generations
GNAS_CROSSOVER_RATE = 0.8
GNAS_MUTATION_RATE  = 0.15
GNAS_TOURNAMENT_K   = 3          # tournament selection size
GNAS_EVAL_EPOCHS    = 10         # epochs per candidate evaluation

# ── Continual Learning ─────────────────────────────────────────────────────
CL_N_TASKS          = 5          # split 10 classes into 5 tasks (2 classes each)
CL_BUFFER_SIZE      = 500        # replay buffer size
CL_EPOCHS_PER_TASK  = 30

# ── Cross-validation ───────────────────────────────────────────────────────
CV_FOLDS            = 5
CV_FOLD_EPOCHS      = 15       # epochs trained per fold during cross-validation

# ── SHAP ───────────────────────────────────────────────────────────────────
SHAP_BACKGROUND     = 50         # background samples for DeepSHAP
SHAP_TEST_SAMPLES   = 20

# ── Hardware ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only print once from the main process (not from each DataLoader worker)
import multiprocessing as _mp
if _mp.current_process().name == "MainProcess":
    print(f"[config] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[config] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
