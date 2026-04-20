"""
07_comparison_plots.py — Comprehensive Comparison Plots
========================================================
Loads all saved results from scripts 03–06 and generates:

  A. CNN Baseline comparisons
  B. QEFS vs Baseline per model and aggregate
  C. Genetic NAS vs Baseline per model and aggregate
  D. Continual Learning method comparisons
  E. Grand unified comparison (all methods, all models)
  F. Technique-level ablation (Baseline → QEFS → GNAS improvement)
  G. Statistical significance tests (Wilcoxon / t-test on per-fold CV)
  H. Memory & timing analysis
  I. Per-class deep-dive: best vs worst performing class per method
  J. Combined leaderboard table (image + CSV)

All plots saved to outputs/plots/
Summary CSV saved to outputs/results/leaderboard.csv

Run:
  python 07_comparison_plots.py
  (run all previous scripts first)
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import load_results, CLASS_NAMES, MODEL_NAMES, NUM_CLASSES, PLOTS_DIR, RESULTS_DIR

# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════
def load_all_baseline() -> list:
    results = []
    for name in MODEL_NAMES:
        r = load_results(name)
        if r: results.append(r)
    return results

def load_all_qefs() -> list:
    results = []
    for name in MODEL_NAMES:
        r = load_results(f"qefs_{name}")
        if r: results.append(r)
    return results

def load_all_gnas() -> list:
    results = []
    for name in MODEL_NAMES:
        r = load_results(f"gnas_{name}")
        if r: results.append(r)
    return results

def load_all_cl() -> list:
    cl_methods = ["gdumb", "derpp", "x_der", "a_gem", "er_ace"]
    results = []
    for m in cl_methods:
        r = load_results(f"cl_{m}")
        if r: results.append(r)
    return results

def safe_get(d, key, default=0):
    return d.get(key, default) if d else default


# ═══════════════════════════════════════════════════════════════════════════
# A. CNN Baseline — comprehensive comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_baseline_comprehensive(baseline: list):
    if not baseline: return
    names = [r["model_name"] for r in baseline]
    palette = sns.color_palette("tab10", len(names))

    metrics_to_plot = [
        ("test_acc",        "Test Accuracy"),
        ("val_acc",         "Validation Accuracy"),
        ("f1_macro",        "F1-Macro"),
        ("f1_weighted",     "F1-Weighted"),
        ("precision_macro", "Precision Macro"),
        ("recall_macro",    "Recall Macro"),
    ]

    # ── (a) 6-panel bar chart ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (key, title) in zip(axes.flat, metrics_to_plot):
        vals = [safe_get(r, key) for r in baseline]
        bars = ax.bar(names, vals, color=palette, edgecolor="black", linewidth=0.5)
        ax.set(title=title, ylim=(0, 1.1), ylabel="Score")
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("CNN Baseline Models — All Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_A_baseline_all_metrics.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_A_baseline_all_metrics.png")

    # ── (b) Per-class accuracy heatmap ────────────────────────────────────
    pca = np.array([[r["per_class_acc"].get(c, 0) for c in CLASS_NAMES]
                    for r in baseline])
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pca, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=CLASS_NAMES, yticklabels=names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_title("Per-class Accuracy — All Baseline Models",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=40, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_A_baseline_classacc_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_A_baseline_classacc_heatmap.png")

    # ── (c) Training time vs memory ───────────────────────────────────────
    times   = [safe_get(r, "training_time_s") / 60 for r in baseline]
    gpu_mem = [safe_get(safe_get(r, "memory", {}), "peak_gpu_mb") for r in baseline]
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(times, [safe_get(r, "test_acc") for r in baseline],
                    s=[max(g, 50) for g in gpu_mem],
                    c=range(len(names)), cmap="tab10", edgecolors="black", linewidth=0.5)
    for i, n in enumerate(names):
        ax.annotate(n, (times[i], safe_get(baseline[i], "test_acc")),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set(title="Training Time vs Test Accuracy (bubble = GPU memory)",
           xlabel="Training Time (min)", ylabel="Test Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_A_baseline_time_mem.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_A_baseline_time_mem.png")

    # ── (d) CV accuracy with error bars ───────────────────────────────────
    cv_means = [safe_get(r, "cv_acc_mean") for r in baseline]
    cv_stds  = [safe_get(r, "cv_acc_std")  for r in baseline]
    if any(v > 0 for v in cv_means):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, cv_means, yerr=cv_stds, capsize=6, color=palette, edgecolor="black")
        ax.set(title="5-Fold Cross-Validation Accuracy",
               ylabel="Accuracy ± Std", ylim=(0, 1.1))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "cmp_A_baseline_cv.png"), dpi=150)
        plt.close()
        print("  Saved: cmp_A_baseline_cv.png")

    # ── (e) Confidence interval plot ───────────────────────────────────────
    ci_lo = [safe_get(r, "ci_lower") for r in baseline]
    ci_hi = [safe_get(r, "ci_upper") for r in baseline]
    acc   = [safe_get(r, "test_acc") for r in baseline]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (n, a, lo, hi) in enumerate(zip(names, acc, ci_lo, ci_hi)):
        ax.plot([lo, hi], [i, i], "b-", linewidth=4, alpha=0.5)
        ax.plot(a, i, "ro", markersize=10)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set(title="Test Accuracy with 95% Bootstrap Confidence Interval",
           xlabel="Accuracy")
    ax.grid(True, alpha=0.3); ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_A_baseline_ci.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_A_baseline_ci.png")


# ═══════════════════════════════════════════════════════════════════════════
# B+C. Technique comparisons: Baseline vs QEFS vs GNAS
# ═══════════════════════════════════════════════════════════════════════════
def plot_technique_comparison(baseline: list, qefs: list, gnas: list):
    if not baseline: return
    names = [r["model_name"] for r in baseline]

    def get_qefs(model_name, key):
        for r in qefs:
            if r.get("model_name") == model_name: return safe_get(r, key)
        return 0.0

    def get_gnas(model_name, key):
        for r in gnas:
            if r.get("model_name") == model_name: return safe_get(r, key)
        return 0.0

    # ── (a) Grouped bar: test accuracy ────────────────────────────────────
    base_acc = [safe_get(r, "test_acc")   for r in baseline]
    qefs_acc = [get_qefs(n, "qefs_test_acc") for n in names]
    gnas_acc = [get_gnas(n, "gnas_test_acc") for n in names]

    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - w, base_acc, w, label="Baseline CNN", color="#4e79a7", edgecolor="black")
    ax.bar(x,     qefs_acc, w, label="QEFS",         color="#f28e2b", edgecolor="black")
    ax.bar(x + w, gnas_acc, w, label="GNAS",         color="#59a14f", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(title="Test Accuracy: Baseline vs QEFS vs GNAS",
           ylabel="Test Accuracy", ylim=(0, 1.12))
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    for i, (b, q, g) in enumerate(zip(base_acc, qefs_acc, gnas_acc)):
        ax.text(i - w, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)
        ax.text(i,     q + 0.005, f"{q:.3f}", ha="center", fontsize=7)
        ax.text(i + w, g + 0.005, f"{g:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_BC_technique_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_BC_technique_accuracy.png")

    # ── (b) Improvement heatmap ────────────────────────────────────────────
    qefs_delta = [q - b for q, b in zip(qefs_acc, base_acc)]
    gnas_delta = [g - b for g, b in zip(gnas_acc, base_acc)]
    delta_matrix = np.array([qefs_delta, gnas_delta])

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(delta_matrix, annot=True, fmt="+.3f", cmap="RdYlGn",
                xticklabels=names, yticklabels=["QEFS Δ", "GNAS Δ"],
                ax=ax, center=0, linewidths=0.5)
    ax.set(title="Accuracy Improvement (Δ) Over Baseline",
           xlabel="Model", ylabel="Technique")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_BC_improvement_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_BC_improvement_heatmap.png")

    # ── (c) F1-macro comparison ────────────────────────────────────────────
    base_f1 = [safe_get(r, "f1_macro")      for r in baseline]
    qefs_f1 = [get_qefs(n, "qefs_f1_macro") for n in names]
    gnas_f1 = [get_gnas(n, "gnas_f1_macro") for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(names, base_f1, "o-", label="Baseline", linewidth=2, color="#4e79a7")
    ax.plot(names, qefs_f1, "s-", label="QEFS",     linewidth=2, color="#f28e2b")
    ax.plot(names, gnas_f1, "^-", label="GNAS",     linewidth=2, color="#59a14f")
    ax.set(title="F1-Macro Score: All Techniques",
           ylabel="F1 Macro", ylim=(0, 1.05))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_BC_f1_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_BC_f1_comparison.png")

    # ── (d) Stack plot: improvement breakdown ─────────────────────────────
    qefs_gain = [max(0, d) for d in qefs_delta]
    gnas_gain = [max(0, d) for d in gnas_delta]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, base_acc, label="Baseline", color="#4e79a7", edgecolor="black")
    ax.bar(names, qefs_gain, bottom=base_acc, label="QEFS gain", color="#f28e2b", edgecolor="black")
    ax.bar(names, gnas_gain, bottom=[b+q for b,q in zip(base_acc, qefs_gain)],
           label="GNAS gain", color="#59a14f", edgecolor="black")
    ax.set(title="Stacked Accuracy Contribution", ylabel="Accuracy")
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_BC_stacked_gain.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_BC_stacked_gain.png")

    # ── (e) Best method per model ──────────────────────────────────────────
    best_methods, best_accs = [], []
    for i, name in enumerate(names):
        contenders = {
            "Baseline": base_acc[i],
            "QEFS":     qefs_acc[i],
            "GNAS":     gnas_acc[i],
        }
        best_m = max(contenders, key=contenders.get)
        best_methods.append(best_m)
        best_accs.append(contenders[best_m])

    colors_map = {"Baseline": "#4e79a7", "QEFS": "#f28e2b", "GNAS": "#59a14f"}
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, best_accs,
                  color=[colors_map[m] for m in best_methods], edgecolor="black")
    ax.set(title="Best Method per Model", ylabel="Test Accuracy", ylim=(0, 1.1))
    ax.set_xticklabels(names, rotation=30, ha="right")
    for bar, m, a in zip(bars, best_methods, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{m}\n{a:.3f}", ha="center", fontsize=8)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=m) for m, c in colors_map.items()])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_BC_best_per_model.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_BC_best_per_model.png")


# ═══════════════════════════════════════════════════════════════════════════
# D. Continual Learning comparisons
# ═══════════════════════════════════════════════════════════════════════════
def plot_cl_comprehensive(cl: list):
    if not cl: return
    methods = [r["method"] for r in cl]
    palette = sns.color_palette("Set2", len(methods))

    # ── Task accuracy progressions ────────────────────────────────────────
    fig, axes = plt.subplots(1, CL_N_TASKS, figsize=(20, 5), sharey=True)
    for t_idx, ax in enumerate(axes):
        for i, r in enumerate(cl):
            mat = np.array(r["acc_matrix"])
            if t_idx < mat.shape[1]:
                vals = mat[:, t_idx]
                # Only show diagonal + below (we haven't trained this task yet above)
                plot_vals = [vals[j] if j >= t_idx else np.nan for j in range(CL_N_TASKS)]
                ax.plot(range(CL_N_TASKS), plot_vals, "o-",
                        color=palette[i], label=r["method"], linewidth=2)
        ax.set(title=f"Task {t_idx} Accuracy Over Time",
               xlabel="Trained on Task #", ylabel="Accuracy")
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
        if t_idx == 0: ax.legend(fontsize=8)
    plt.suptitle("Task Accuracy Over Sequential Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_D_cl_task_progression.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_D_cl_task_progression.png")

    # ── BWT vs AA scatter ─────────────────────────────────────────────────
    AA  = [safe_get(r, "AA")  for r in cl]
    BWT = [safe_get(r, "BWT") for r in cl]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (m, a, b) in enumerate(zip(methods, AA, BWT)):
        ax.scatter(b, a, s=200, color=palette[i], zorder=5, edgecolors="black")
        ax.annotate(m, (b, a), textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set(title="AA vs BWT (higher AA + BWT closer to 0 is better)",
           xlabel="Backward Transfer (BWT)", ylabel="Average Accuracy (AA)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_D_cl_aa_vs_bwt.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_D_cl_aa_vs_bwt.png")

    # ── Final accuracy vs forgetting (−BWT) ───────────────────────────────
    forgetting = [-safe_get(r, "BWT") for r in cl]
    final_acc  = [safe_get(r, "final_test_acc") for r in cl]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (m, a, f) in enumerate(zip(methods, final_acc, forgetting)):
        ax.scatter(f, a, s=200, color=palette[i], zorder=5, edgecolors="black")
        ax.annotate(m, (f, a), textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.set(title="Final Test Accuracy vs Catastrophic Forgetting",
           xlabel="Forgetting (−BWT, lower is better)",
           ylabel="Final Test Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_D_cl_forgetting_vs_acc.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_D_cl_forgetting_vs_acc.png")


# ═══════════════════════════════════════════════════════════════════════════
# E. Grand unified comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_grand_comparison(baseline: list, qefs: list, gnas: list, cl: list):
    entries = []

    for r in baseline:
        entries.append({
            "label":   r["model_name"],
            "group":   "Baseline CNN",
            "acc":     safe_get(r, "test_acc"),
            "f1":      safe_get(r, "f1_macro"),
            "train_s": safe_get(r, "training_time_s"),
        })
    for r in qefs:
        entries.append({
            "label":   r["model_name"] + " (QEFS)",
            "group":   "QEFS",
            "acc":     safe_get(r, "qefs_test_acc"),
            "f1":      safe_get(r, "qefs_f1_macro"),
            "train_s": safe_get(r, "training_time_s"),
        })
    for r in gnas:
        entries.append({
            "label":   r["model_name"] + " (GNAS)",
            "group":   "Genetic NAS",
            "acc":     safe_get(r, "gnas_test_acc"),
            "f1":      safe_get(r, "gnas_f1_macro"),
            "train_s": safe_get(r, "training_time_s"),
        })
    for r in cl:
        entries.append({
            "label":   r["method"],
            "group":   "Continual Learning",
            "acc":     safe_get(r, "final_test_acc"),
            "f1":      safe_get(r, "final_f1_macro"),
            "train_s": safe_get(r, "total_training_time_s"),
        })

    if not entries: return
    df = pd.DataFrame(entries).sort_values("acc", ascending=False)

    # ── Horizontal sorted accuracy bar ────────────────────────────────────
    group_colors = {
        "Baseline CNN":       "#4e79a7",
        "QEFS":               "#f28e2b",
        "Genetic NAS":        "#59a14f",
        "Continual Learning": "#e15759",
    }
    colors = [group_colors.get(g, "gray") for g in df["group"]]

    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))
    bars = ax.barh(df["label"], df["acc"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set(title="Grand Unified Accuracy Comparison — All Methods & Models",
           xlabel="Test Accuracy", xlim=(0, 1.1))
    for bar, v in zip(bars, df["acc"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in group_colors.items()],
              loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_E_grand_unified.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_E_grand_unified.png")

    # ── Box plot per group ────────────────────────────────────────────────
    groups_order = ["Baseline CNN", "QEFS", "Genetic NAS", "Continual Learning"]
    data_by_group = {g: df[df["group"] == g]["acc"].tolist() for g in groups_order}
    data_by_group = {g: v for g, v in data_by_group.items() if v}

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(data_by_group))
    bp = ax.boxplot(list(data_by_group.values()),
                    labels=list(data_by_group.keys()),
                    patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], [group_colors[g] for g in data_by_group]):
        patch.set_facecolor(color)
    ax.set(title="Accuracy Distribution by Method Group",
           ylabel="Test Accuracy", ylim=(0, 1.1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_E_group_boxplot.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_E_group_boxplot.png")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# F. Ablation: Baseline → QEFS → GNAS
# ═══════════════════════════════════════════════════════════════════════════
def plot_ablation(baseline: list, qefs: list, gnas: list):
    names = [r["model_name"] for r in baseline]
    stages = ["Baseline", "QEFS", "GNAS"]

    def get_acc(res_list, model_name, key):
        for r in res_list:
            if r.get("model_name") == model_name:
                return safe_get(r, key)
        return 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    palette = sns.color_palette("viridis_r", 3)
    for ax, name in zip(axes.flat, names):
        b = get_acc(baseline, name, "test_acc")
        q = get_acc(qefs,     name, "qefs_test_acc")
        g = get_acc(gnas,     name, "gnas_test_acc")
        vals = [b, q, g]
        bars = ax.bar(stages, vals, color=palette, edgecolor="black")
        ax.set(title=name, ylim=(0, 1.1), ylabel="Test Accuracy")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Arrows showing change
        for i in range(2):
            delta = vals[i+1] - vals[i]
            ax.annotate("", xy=(i+1, vals[i+1]), xytext=(i, vals[i]),
                        arrowprops=dict(arrowstyle="->",
                                        color="green" if delta >= 0 else "red",
                                        lw=1.5))
    plt.suptitle("Ablation: Baseline → QEFS → GNAS per Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_F_ablation.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_F_ablation.png")


# ═══════════════════════════════════════════════════════════════════════════
# G. Statistical significance
# ═══════════════════════════════════════════════════════════════════════════
def plot_statistical_tests(baseline: list, qefs: list, gnas: list):
    """
    Wilcoxon signed-rank test comparing per-fold CV accuracy of baseline
    vs QEFS/GNAS. Uses cv_fold_accs stored in results.
    """
    rows = []
    for r in baseline:
        name = r["model_name"]
        b_folds = r.get("cv_fold_accs", [])
        if not b_folds: continue

        for res_list, method in [(qefs, "QEFS"), (gnas, "GNAS")]:
            other = next((x for x in res_list if x.get("model_name") == name), None)
            if not other: continue
            o_folds = other.get("cv_fold_accs", [])
            if not o_folds: continue
            n = min(len(b_folds), len(o_folds))
            if n < 2: continue
            try:
                stat, pval = stats.wilcoxon(b_folds[:n], o_folds[:n])
            except Exception:
                pval = 1.0
            rows.append({
                "Model":   name,
                "Method":  method,
                "p-value": pval,
                "Significant": "✓" if pval < 0.05 else "✗",
            })

    if not rows: return
    df = pd.DataFrame(rows)
    print("\n  Statistical Significance Tests (Wilcoxon signed-rank):")
    print(df.to_string(index=False))

    # Heatmap of p-values
    pivot = df.pivot(index="Model", columns="Method", values="p-value").fillna(1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                ax=ax, vmin=0, vmax=0.1, linewidths=0.5)
    ax.set_title("Wilcoxon p-values (< 0.05 = significant improvement)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_G_statistical_tests.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_G_statistical_tests.png")


# ═══════════════════════════════════════════════════════════════════════════
# H. Memory & timing
# ═══════════════════════════════════════════════════════════════════════════
def plot_timing_memory(baseline: list, qefs: list, gnas: list, cl: list):
    all_entries = []
    for r in baseline:
        all_entries.append({
            "label":   r["model_name"] + "\n(Base)",
            "group":   "Baseline",
            "inf_ms":  safe_get(r, "inference_time_per_img_s") * 1000,
            "train_m": safe_get(r, "training_time_s") / 60,
            "gpu_mb":  safe_get(safe_get(r, "memory", {}), "peak_gpu_mb"),
        })
    for r in qefs:
        all_entries.append({
            "label":   r["model_name"] + "\n(QEFS)",
            "group":   "QEFS",
            "inf_ms":  safe_get(r, "inference_time_per_img_s") * 1000,
            "train_m": safe_get(r, "training_time_s") / 60,
            "gpu_mb":  0,
        })
    for r in gnas:
        all_entries.append({
            "label":   r["model_name"] + "\n(GNAS)",
            "group":   "GNAS",
            "inf_ms":  safe_get(r, "inference_time_per_img_s") * 1000,
            "train_m": safe_get(r, "training_time_s") / 60,
            "gpu_mb":  0,
        })
    for r in cl:
        all_entries.append({
            "label":   r["method"],
            "group":   "CL",
            "inf_ms":  safe_get(r, "inference_time_per_img_s") * 1000,
            "train_m": safe_get(r, "total_training_time_s") / 60,
            "gpu_mb":  0,
        })

    if not all_entries: return
    df = pd.DataFrame(all_entries)
    group_colors = {"Baseline": "#4e79a7", "QEFS": "#f28e2b",
                    "GNAS": "#59a14f", "CL": "#e15759"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, col, title in zip(axes,
                              ["inf_ms", "train_m"],
                              ["Inference Time per Image (ms)",
                               "Total Training Time (minutes)"]):
        colors = [group_colors.get(g, "gray") for g in df["group"]]
        bars = ax.barh(df["label"], df[col], color=colors, edgecolor="black", linewidth=0.3)
        ax.set(title=title, xlabel=col.split("_")[0].capitalize())
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Inference & Training Time — All Methods", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_H_timing.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_H_timing.png")


# ═══════════════════════════════════════════════════════════════════════════
# I. Per-class deep-dive
# ═══════════════════════════════════════════════════════════════════════════
def plot_class_deepdive(baseline: list, qefs: list, gnas: list):
    if not baseline: return
    names = [r["model_name"] for r in baseline]

    # Build per-class acc matrix for each group
    def get_pca(res_list, acc_key="per_class_acc"):
        return np.array([[r.get(acc_key, {}).get(c, 0) for c in CLASS_NAMES]
                         for r in res_list])

    base_pca = get_pca(baseline)
    qefs_pca = get_pca(qefs) if qefs else None
    gnas_pca = get_pca(gnas) if gnas else None

    # Best / worst class per model
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    best_idx  = base_pca.argmax(axis=1)
    worst_idx = base_pca.argmin(axis=1)

    axes[0].barh(names, base_pca.max(axis=1), color="#4CAF50", edgecolor="black")
    for i, bi in enumerate(best_idx):
        axes[0].text(base_pca[i, bi] + 0.005, i,
                     CLASS_NAMES[bi], va="center", fontsize=8)
    axes[0].set(title="Best Class Accuracy per Model", xlabel="Accuracy")

    axes[1].barh(names, base_pca.min(axis=1), color="#F44336", edgecolor="black")
    for i, wi in enumerate(worst_idx):
        axes[1].text(base_pca[i, wi] + 0.005, i,
                     CLASS_NAMES[wi], va="center", fontsize=8)
    axes[1].set(title="Worst Class Accuracy per Model", xlabel="Accuracy")

    plt.suptitle("Per-Class Best/Worst Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_I_class_deepdive.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_I_class_deepdive.png")

    # Average per-class accuracy across all models / techniques
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(NUM_CLASSES)
    ax.bar(x - 0.2, base_pca.mean(axis=0), 0.2, label="Baseline", color="#4e79a7", edgecolor="black")
    if qefs_pca is not None and len(qefs_pca) > 0:
        ax.bar(x,     qefs_pca.mean(axis=0), 0.2, label="QEFS",    color="#f28e2b", edgecolor="black")
    if gnas_pca is not None and len(gnas_pca) > 0:
        ax.bar(x + 0.2, gnas_pca.mean(axis=0), 0.2, label="GNAS",  color="#59a14f", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set(title="Average Per-class Accuracy (across all models)",
           ylabel="Accuracy", ylim=(0, 1.1))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_I_avg_class_acc.png"), dpi=150)
    plt.close()
    print("  Saved: cmp_I_avg_class_acc.png")


# ═══════════════════════════════════════════════════════════════════════════
# J. Leaderboard
# ═══════════════════════════════════════════════════════════════════════════
def generate_leaderboard(baseline: list, qefs: list, gnas: list, cl: list):
    rows = []

    for r in baseline:
        rows.append({
            "Rank": 0, "Method": "Baseline CNN", "Model": r["model_name"],
            "Test Acc": safe_get(r, "test_acc"),
            "Val Acc":  safe_get(r, "val_acc"),
            "F1 Macro": safe_get(r, "f1_macro"),
            "F1 Wt":    safe_get(r, "f1_weighted"),
            "Precision": safe_get(r, "precision_macro"),
            "Recall":   safe_get(r, "recall_macro"),
            "Train Time (s)": safe_get(r, "training_time_s"),
            "Inf Time (ms)":  safe_get(r, "inference_time_per_img_s") * 1000,
            "CV Acc": safe_get(r, "cv_acc_mean"),
            "CI Low": safe_get(r, "ci_lower"),
            "CI High": safe_get(r, "ci_upper"),
        })
    for r in qefs:
        rows.append({
            "Rank": 0, "Method": "QEFS", "Model": r["model_name"],
            "Test Acc": safe_get(r, "qefs_test_acc"),
            "Val Acc":  safe_get(r, "qcnn_test_acc"),
            "F1 Macro": safe_get(r, "qefs_f1_macro"),
            "F1 Wt":    safe_get(r, "f1_weighted"),
            "Precision": safe_get(r, "precision_macro"),
            "Recall":   safe_get(r, "recall_macro"),
            "Train Time (s)": safe_get(r, "training_time_s"),
            "Inf Time (ms)":  safe_get(r, "inference_time_per_img_s") * 1000,
            "CV Acc": 0, "CI Low": safe_get(r, "ci_lower"), "CI High": safe_get(r, "ci_upper"),
        })
    for r in gnas:
        rows.append({
            "Rank": 0, "Method": "Genetic NAS", "Model": r["model_name"],
            "Test Acc": safe_get(r, "gnas_test_acc"),
            "Val Acc":  safe_get(r, "gnas_val_acc"),
            "F1 Macro": safe_get(r, "gnas_f1_macro"),
            "F1 Wt":    safe_get(r, "f1_weighted"),
            "Precision": safe_get(r, "precision_macro"),
            "Recall":   safe_get(r, "recall_macro"),
            "Train Time (s)": safe_get(r, "training_time_s"),
            "Inf Time (ms)":  safe_get(r, "inference_time_per_img_s") * 1000,
            "CV Acc": 0, "CI Low": safe_get(r, "ci_lower"), "CI High": safe_get(r, "ci_upper"),
        })
    for r in cl:
        rows.append({
            "Rank": 0, "Method": r["method"], "Model": "ResNet50 (CL)",
            "Test Acc": safe_get(r, "final_test_acc"),
            "Val Acc":  0,
            "F1 Macro": safe_get(r, "final_f1_macro"),
            "F1 Wt":    safe_get(r, "f1_weighted"),
            "Precision": safe_get(r, "precision_macro"),
            "Recall":   safe_get(r, "recall_macro"),
            "Train Time (s)": safe_get(r, "total_training_time_s"),
            "Inf Time (ms)":  safe_get(r, "inference_time_per_img_s") * 1000,
            "CV Acc": 0,
            "CI Low": safe_get(r, "ci_lower"), "CI High": safe_get(r, "ci_upper"),
        })

    if not rows: return
    df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "leaderboard.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Leaderboard CSV → {csv_path}")

    # Plot leaderboard table image
    fig, ax = plt.subplots(figsize=(22, max(6, len(df) * 0.45 + 2)))
    ax.axis("off")
    cols_show = ["Rank", "Method", "Model", "Test Acc", "F1 Macro",
                 "Precision", "Recall", "CV Acc", "Train Time (s)", "Inf Time (ms)"]
    cell_text = df[cols_show].round(4).astype(str).values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=cols_show,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(cols_show))))

    # Colour top-3 rows
    for col in range(len(cols_show)):
        for row_i in range(min(3, len(df))):
            cell = table[row_i + 1, col]
            cell.set_facecolor(["#FFD700", "#C0C0C0", "#CD7F32"][row_i])

    plt.title("AgroDetect — Full Leaderboard", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cmp_J_leaderboard.png"), dpi=120,
                bbox_inches="tight")
    plt.close()
    print("  Saved: cmp_J_leaderboard.png")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AgroDetect — Comprehensive Comparison Plots")
    print("=" * 60)

    baseline = load_all_baseline()
    qefs     = load_all_qefs()
    gnas     = load_all_gnas()
    cl       = load_all_cl()

    print(f"\n  Loaded: {len(baseline)} baseline, {len(qefs)} QEFS, "
          f"{len(gnas)} GNAS, {len(cl)} CL results")

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})

    print("\n[A] Baseline comparisons …")
    plot_baseline_comprehensive(baseline)

    print("\n[B+C] Technique comparisons …")
    plot_technique_comparison(baseline, qefs, gnas)

    print("\n[D] Continual learning …")
    plot_cl_comprehensive(cl)

    print("\n[E] Grand unified …")
    grand_df = plot_grand_comparison(baseline, qefs, gnas, cl)

    print("\n[F] Ablation …")
    plot_ablation(baseline, qefs, gnas)

    print("\n[G] Statistical tests …")
    plot_statistical_tests(baseline, qefs, gnas)

    print("\n[H] Timing & memory …")
    plot_timing_memory(baseline, qefs, gnas, cl)

    print("\n[I] Per-class deep-dive …")
    plot_class_deepdive(baseline, qefs, gnas)

    print("\n[J] Leaderboard …")
    lb = generate_leaderboard(baseline, qefs, gnas, cl)
    if lb is not None:
        print(lb[["Rank", "Method", "Model", "Test Acc", "F1 Macro"]].to_string(index=False))

    print(f"\n✓ All comparison plots saved to {PLOTS_DIR}\n")
