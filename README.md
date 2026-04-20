# AgroDetect — Rice Disease Classification Pipeline

A full end-to-end deep learning pipeline for classifying **10 rice plant diseases** using:
- **Baseline CNNs** (6 architectures with full diagnostics)
- **QEFS** — Quantum-Inspired Evolutionary Feature Selection
- **Genetic NAS** — Genetic Algorithm Neural Architecture Search
- **Continual Learning** — 5 methods (GDumb, DER++, X-DER, A-GEM, ER-ACE)

---

## Dataset

**Source:** [Paddy Doctor — Paddy Disease Classification (Kaggle)](https://www.kaggle.com/competitions/paddy-disease-classification)

**10 classes:** `bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`, `blast`, `brown_spot`, `dead_heart`, `downy_mildew`, `hispa`, `normal`, `tungro`

**Expected folder structure after download:**

```
Agrodetect/
├── Dataset/
│   ├── train.csv
│   ├── train_images/
│   │   ├── bacterial_leaf_blight/
│   │   ├── blast/
│   │   └── ... (10 class folders)
│   └── test_images/
```

---

## Requirements

- Python 3.10+
- CUDA 12.x (NVIDIA GPU strongly recommended — tested on RTX 5080 16 GB)
- Conda or virtualenv

### Install dependencies

```bash
# Option A — pip
pip install -r requirements.txt

# Option B — Conda (recommended)
conda create -n agrodetect python=3.10 -y
conda activate agrodetect
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Full Pipeline

Run phases **in order**. Each phase saves checkpoints and results to `outputs/` so you can resume if interrupted.

### Phase 1 — Data Cleaning

Removes corrupt/duplicate images, validates file integrity, saves `Dataset/cleaned_train.csv`.

```bash
python 01_data_cleaning.py
```

### Phase 2 — Data Balancing

Balances all 10 classes to 1,700 images each via augmentation, saves `Dataset/balanced_train.csv` and augmented images to `Dataset/aug_images/`.

```bash
python 02_data_balancing.py
```

### Phase 3 — Baseline CNN Training

Trains 6 CNN architectures: **ConvNeXt-Base, DenseNet-161, GoogLeNet, MobileNetV3-Large, ResNet-50, ShuffleNetV2**.

Each model produces: train/val/test accuracy, F1-score, confusion matrix, Grad-CAM heatmaps, SHAP explanations, confidence intervals, 5-fold cross-validation, memory profile, inference time.

```bash
python 03_train_cnn_models.py
```

**Key hyperparameters** (edit `config.py` to change):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EPOCHS` | 60 | Training epochs |
| `BATCH_SIZE` | 32 | Batch size |
| `LEARNING_RATE` | 1e-3 | Initial LR (CosineAnnealing scheduler) |
| `PATIENCE` | 12 | Early stopping patience |
| `CV_FOLDS` | 5 | Cross-validation folds |

### Phase 4 — QEFS (Quantum-Inspired Evolutionary Feature Selection)

Applies a Hybrid Firefly–Swallow Evolutionary Algorithm (HFSEA) to select the most discriminative features from 512-dim quantum projections on top of each CNN backbone.

```bash
python 04_qefs.py
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QEFS_FEATURE_DIM` | 512 | Quantum projection dimension |
| `QEFS_POP_SIZE` | 30 | Firefly population size |
| `QEFS_MAX_ITER` | 50 | HFSEA iterations |
| `QEFS_FINETUNE_EPOCHS` | 20 | Quantum fine-tune epochs |

### Phase 5 — Genetic NAS

Uses a Genetic Algorithm to search for the optimal classifier head architecture on top of QEFS-enhanced backbones. Each candidate is evaluated on validation accuracy.

```bash
python 05_genetic_nas.py
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GNAS_POP_SIZE` | 20 | GA population size |
| `GNAS_GENERATIONS` | 15 | Number of GA generations |
| `GNAS_EVAL_EPOCHS` | 10 | Epochs per candidate evaluation |
| `GNAS_CROSSOVER_RATE` | 0.8 | Crossover probability |
| `GNAS_MUTATION_RATE` | 0.15 | Mutation probability |

### Phase 6 — Continual Learning

Splits 10 classes into 5 sequential tasks (2 classes each) and trains 5 continual learning methods on the GNAS-optimised backbone. Measures Average Accuracy (AA), Backward Transfer (BWT), and Forward Transfer (FWT).

```bash
python 06_continual_learning.py
```

**Key hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CL_MODEL_NAME` | `resnet50` | Backbone for all CL methods |
| `CL_N_TASKS` | 5 | Number of sequential tasks |
| `CL_EPOCHS_PER_TASK` | 30 | Epochs trained per task |
| `CL_BUFFER_SIZE` | 500 | Replay buffer size |

### Phase 7 — Comparison Plots

Generates comprehensive cross-phase comparison plots across all methods and models.

```bash
python 07_comparison_plots.py
```

---

## Pipeline Chain

```
Dataset
   │
   ▼
01_data_cleaning.py        → Dataset/cleaned_train.csv
   │
   ▼
02_data_balancing.py       → Dataset/balanced_train.csv
   │
   ▼
03_train_cnn_models.py     → outputs/models/{model}_best.pth
   │                          outputs/results/{model}.json
   ▼
04_qefs.py                 → outputs/models/qefs_{model}_best.pth
   │                          outputs/results/qefs_{model}.json
   ▼
05_genetic_nas.py          → outputs/models/gnas_{model}_best.pth
   │                          outputs/results/gnas_{model}.json
   ▼
06_continual_learning.py   → outputs/results/cl_{method}.json
   │
   ▼
07_comparison_plots.py     → outputs/plots/*.png
```

---

## Outputs

All results are saved under `outputs/`:

```
outputs/
├── models/          # .pth checkpoints (baseline, QEFS, GNAS)
├── results/         # .json metrics per model/method
├── plots/           # PNG figures (training curves, confusion matrices,
│                    #   SHAP, Grad-CAM, feature importance, CL curves, etc.)
└── logs/            # Training logs
```

---

## Validation Test

After training all phases, run the full validation suite (57 checks) to confirm everything loaded correctly:

```bash
python test_pipeline.py
```

---

## Project Structure

```
Agrodetect/
├── config.py                  # All hyperparameters and paths
├── utils.py                   # Shared dataset, training loops, metrics, EarlyStopping
├── 01_data_cleaning.py
├── 02_data_balancing.py
├── 03_train_cnn_models.py
├── 04_qefs.py
├── 05_genetic_nas.py
├── 06_continual_learning.py
├── 07_comparison_plots.py
├── quick_test_pipeline.py     # End-to-end smoke test (reduced settings)
├── test_pipeline.py           # Post-training validation suite
├── requirements.txt
├── run_pipeline.sh            # Bash script to run all phases sequentially
└── Dataset/
    ├── train.csv
    ├── train_images/
    └── test_images/
```
