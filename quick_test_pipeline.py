#!/usr/bin/env python
"""
quick_test_pipeline.py
======================
Runs all 7 Agrodetect pipeline phases end-to-end with reduced settings
so you can verify the complete pipeline executes without errors before
committing to a full multi-hour training run.

Quick-test overrides (patched in-memory — source files are NOT modified):
  • Models tested  : shufflenet_v2 only (lightest model, ~5 MB)
  • CNN training   : 2 epochs, 2-fold × 1-epoch CV, SHAP 5/3 samples
  • QEFS           : 2 fine-tune epochs, 5 particles, 3 firefly iterations
  • Genetic NAS    : 2 GA generations, pop-size 4, 2 eval-epochs/candidate
  • Continual Lrn  : 2 epochs per task, shufflenet_v2 backbone

Phases 1 & 2 (data cleaning / balancing) are automatically SKIPPED when
their output files (cleaned_train.csv, balanced_train.csv) already exist,
since re-running them would take 30+ minutes scanning images that are
already validated.  Use --force-preprocess to override.

Usage
-----
    python quick_test_pipeline.py                 # full pipeline (auto-skip 1&2 if done)
    python quick_test_pipeline.py --from 3        # resume from phase 3
    python quick_test_pipeline.py --only 5        # run only phase 5
    python quick_test_pipeline.py --force-preprocess  # force re-run phases 1 & 2
    python quick_test_pipeline.py --no-stop       # continue past failures

Expected wall-clock time:  ~10–20 min on RTX 5080 (1 model × phases 3-7)
"""

import os
import sys
import time
import subprocess
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")

def banner(title, char="─"):
    w = 66
    print(f"\n{BOLD}{char*w}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{char*w}{RESET}")

# ── Locate config paths ───────────────────────────────────────────────────────
# Import config just to read its path constants (not patching here, just reading).
sys.path.insert(0, BASE_DIR)
try:
    import config as _cfg
    _DATASET_DIR  = _cfg.DATASET_DIR
    _CLEANED_CSV  = _cfg.CLEANED_CSV
    _BALANCED_CSV = _cfg.BALANCED_CSV
except Exception:
    _DATASET_DIR  = os.path.join(BASE_DIR, "Dataset")
    _CLEANED_CSV  = os.path.join(_DATASET_DIR, "cleaned_train.csv")
    _BALANCED_CSV = os.path.join(_DATASET_DIR, "balanced_train.csv")

# ── Phase definitions ─────────────────────────────────────────────────────────
# Each entry: (display_name, script, config_patches, timeout_minutes, skip_if)
#
# skip_if: callable() -> (bool, reason_str) — if True the phase is skipped.
# config_patches are applied by monkey-patching the 'config' module in the
# subprocess BEFORE the target script executes via runpy.run_path, so all
# `from config import *` statements in the script pick up the reduced values.

def _phase1_skip():
    if os.path.exists(_CLEANED_CSV):
        return True, f"cleaned_train.csv already exists → skipping"
    return False, ""

def _phase2_skip():
    if os.path.exists(_BALANCED_CSV):
        return True, f"balanced_train.csv already exists → skipping"
    return False, ""

PHASES = [
    (
        "Phase 1 · Data Cleaning",
        "01_data_cleaning.py",
        {},
        45,           # bumped: image scan can take 20–30 min on large datasets
        _phase1_skip,
    ),
    (
        "Phase 2 · Data Balancing",
        "02_data_balancing.py",
        {},
        45,
        _phase2_skip,
    ),
    (
        "Phase 3 · CNN Training  [shufflenet_v2 only, 2 epochs, 2-fold CV]",
        "03_train_cnn_models.py",
        {
            "MODEL_NAMES":       ["shufflenet_v2"],
            "NUM_EPOCHS":        2,
            "T_MAX":             2,
            "PATIENCE":          999,
            "CV_FOLDS":          2,
            "CV_FOLD_EPOCHS":    1,
            "SHAP_BACKGROUND":   5,
            "SHAP_TEST_SAMPLES": 3,
            "NUM_WORKERS":       0,   # 0 avoids Windows shared-memory page-file drain
        },
        30,
        None,
    ),
    (
        "Phase 4 · QEFS  [shufflenet_v2 only, 2 fine-tune epochs]",
        "04_qefs.py",
        {
            "MODEL_NAMES":          ["shufflenet_v2"],
            "QEFS_FINETUNE_EPOCHS": 2,
            "QEFS_POP_SIZE":        5,
            "QEFS_MAX_ITER":        3,
            "PATIENCE":             999,
            "SHAP_BACKGROUND":      5,
            "SHAP_TEST_SAMPLES":    3,
            "NUM_WORKERS":          0,
        },
        30,
        None,
    ),
    (
        "Phase 5 · Genetic NAS  [shufflenet_v2, pop=4, 2 gens, 1 eval-epoch]",
        "05_genetic_nas.py",
        {
            "MODEL_NAMES":       ["shufflenet_v2"],
            "GNAS_POP_SIZE":     4,
            "GNAS_GENERATIONS":  2,
            "GNAS_EVAL_EPOCHS":  1,
            "NUM_EPOCHS":        2,
            "T_MAX":             2,
            "PATIENCE":          999,
            "SHAP_BACKGROUND":   5,
            "SHAP_TEST_SAMPLES": 3,
            "NUM_WORKERS":       0,
        },
        60,
        None,
    ),
    (
        "Phase 6 · Continual Learning  [shufflenet_v2 backbone, 2 epochs/task]",
        "06_continual_learning.py",
        {
            "CL_MODEL_NAME":      "shufflenet_v2",
            "CL_EPOCHS_PER_TASK": 2,
            "PATIENCE":           999,
            "NUM_WORKERS":        0,   # critical on Windows: prevents error 1455
        },
        60,
        None,
    ),
    (
        "Phase 7 · Comparison Plots",
        "07_comparison_plots.py",
        {},
        15,
        None,
    ),
]

# ── Runner ────────────────────────────────────────────────────────────────────

def run_phase(name: str, script: str, patches: dict, timeout_min: int,
              skip_fn, force: bool = False) -> str:
    """
    Execute `script` with config monkey-patched.
    Returns 'pass', 'skip', or 'fail'.
    """
    banner(name)

    # Auto-skip check
    if skip_fn and not force:
        should_skip, reason = skip_fn()
        if should_skip:
            ok(f"SKIPPED — {reason}")
            return "skip"

    patch_lines = "\n".join(
        f"config.{k} = {repr(v)}" for k, v in patches.items()
    )
    if not patch_lines:
        patch_lines = "pass  # no overrides for this phase"

    runner_code = (
        f"import sys, os\n"
        f"sys.path.insert(0, r\"{BASE_DIR}\")\n"
        f"\n"
        f"import config\n"
        f"{patch_lines}\n"
        f"\n"
        f"import runpy\n"
        f"runpy.run_path(r\"{os.path.join(BASE_DIR, script)}\", run_name='__main__')\n"
    )

    info(f"Script  : {script}")
    if patches:
        info("Overrides: " + "  ".join(f"{k}={v}" for k, v in patches.items()))
    info(f"Timeout : {timeout_min} min")
    print()

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-c", runner_code],
            cwd=BASE_DIR,
            timeout=timeout_min * 60,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            ok(f"PASSED  ({elapsed/60:.1f} min)")
            return "pass"
        else:
            err(f"FAILED  (exit code {result.returncode}, {elapsed/60:.1f} min)")
            return "fail"
    except subprocess.TimeoutExpired:
        err(f"TIMEOUT after {timeout_min} min — process killed")
        return "fail"
    except KeyboardInterrupt:
        warn("Interrupted by user")
        sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Quick end-to-end pipeline test (reduced epochs/generations)"
    )
    p.add_argument(
        "--from", dest="from_phase", type=int, default=1, metavar="N",
        help="Start from phase N (1-7). Default: 1"
    )
    p.add_argument(
        "--only", dest="only_phase", type=int, default=None, metavar="N",
        help="Run only phase N (1-7) and exit."
    )
    p.add_argument(
        "--no-stop", action="store_true",
        help="Continue running remaining phases even if one fails."
    )
    p.add_argument(
        "--force-preprocess", action="store_true",
        help="Force re-run phases 1 & 2 even if CSV outputs already exist."
    )
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"""
{BOLD}╔{'═'*64}╗
║{'  AGRODETECT  —  Quick End-to-End Pipeline Smoke Test':^64}║
║{'  shufflenet_v2 only · 2 epochs · 2 GA gens · 5 QEFS particles':^64}║
╚{'═'*64}╝{RESET}""")

    # Check preprocessing state and inform user
    p1_skip, p1_reason = _phase1_skip()
    p2_skip, p2_reason = _phase2_skip()
    if (p1_skip or p2_skip) and not args.force_preprocess:
        print(f"\n  {YELLOW}Note:{RESET} Preprocessing outputs detected:")
        if p1_skip: print(f"    • Phase 1 will be auto-skipped ({p1_reason})")
        if p2_skip: print(f"    • Phase 2 will be auto-skipped ({p2_reason})")
        print(f"    Use --force-preprocess to re-run them.\n")

    phases_to_run = PHASES
    if args.only_phase:
        idx = args.only_phase - 1
        if not (0 <= idx < len(PHASES)):
            print(f"Error: --only must be 1–{len(PHASES)}")
            sys.exit(1)
        phases_to_run = [PHASES[idx]]
    elif args.from_phase > 1:
        idx = args.from_phase - 1
        if not (0 <= idx < len(PHASES)):
            print(f"Error: --from must be 1–{len(PHASES)}")
            sys.exit(1)
        phases_to_run = PHASES[idx:]
        print(f"  Resuming from phase {args.from_phase}: {phases_to_run[0][0]}")

    results    = []   # list of (name, status)  status in {pass, fail, skip}
    wall_start = time.time()

    for name, script, patches, timeout, skip_fn in phases_to_run:
        status = run_phase(
            name, script, patches, timeout, skip_fn,
            force=args.force_preprocess,
        )
        results.append((name, status))

        if status == "fail" and not args.no_stop:
            warn("Phase failed — stopping pipeline early.")
            warn("Re-run with --no-stop to continue past failures.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    wall_time = time.time() - wall_start
    banner("SUMMARY", "═")

    n_pass  = sum(1 for _, s in results if s == "pass")
    n_skip  = sum(1 for _, s in results if s == "skip")
    n_fail  = sum(1 for _, s in results if s == "fail")
    n_ran   = len(results)

    for name, status in results:
        if status == "pass":
            sym = f"{GREEN}✓{RESET}"
            label = "passed"
        elif status == "skip":
            sym = f"{YELLOW}–{RESET}"
            label = "skipped (output exists)"
        else:
            sym = f"{RED}✗{RESET}"
            label = "FAILED"
        print(f"  {sym}  {name}  [{label}]")

    all_ran    = (n_ran == len(phases_to_run))
    all_ok     = (n_fail == 0)   # pass + skip counts as OK

    print(f"\n  {n_pass} passed · {n_skip} skipped · {n_fail} failed  "
          f"({wall_time/60:.1f} min total)")

    if all_ran and all_ok:
        print(f"\n  {BOLD}{GREEN}ALL PHASES PASSED ✓{RESET}")
        print("  Pipeline is validated — safe to run the full training:")
        print()
        print("    python 01_data_cleaning.py")
        print("    python 02_data_balancing.py")
        print("    python 03_train_cnn_models.py")
        print("    python 04_qefs.py")
        print("    python 05_genetic_nas.py")
        print("    python 06_continual_learning.py")
        print("    python 07_comparison_plots.py")
    elif not all_ran:
        failed_idx = next(
            (i + 1 for i, (_, s) in enumerate(results) if s == "fail"), n_ran + 1
        )
        print(f"\n  {YELLOW}Pipeline stopped at first failure.{RESET}")
        print("  Fix the error shown above, then re-run:")
        print(f"    python quick_test_pipeline.py --from {failed_idx}")
    else:
        print(f"\n  {RED}Some phases failed — check output above.{RESET}")

    sys.exit(0 if (all_ran and all_ok) else 1)


if __name__ == "__main__":
    main()
