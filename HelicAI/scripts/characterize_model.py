#!/usr/bin/env python3
"""
characterize_model.py — Post-training characterization report for a trained
E2E-H / E2E-64 ECD model.

Consumes artifacts ALREADY SAVED by train_pipeline.py -- no retraining, no
TensorFlow dependency here at all:
    histories/<label>_history.json
    models/<label>/WL_MODEL.npy
    models/<label>/test/Y_test_true.npy, Y_test_pred.npy
    models/<label>/<label>_test_molecule_metrics.csv

Produces, per model label:
  1. Loss curves (MSE and MAE, train vs validation, best epoch marked)
  2. Precision overview: predicted-vs-true density plot + distributions of
     per-molecule cosine similarity / R2 / RMSE (cosine shown first)
  3. Random ECD spectra (true vs predicted) for a handful of test molecules
  4. Cosine-similarity-quartile representative spectra (best / typical / poor),
     with R2 / RMSE / MAE also called out on each panel
  5. Error analysis by substituent, ranked by mean cosine similarity
     (RMSE / R2 / MAE also computed and reported for each substituent)
  6. Error vs. number of substituents (nsust) as a complexity check

Run as a script or paste directly into a Colab cell.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------------------------------- #
# Colab-friendly defaults (see train_pipeline.py notes on parse_known_args)  #
# --------------------------------------------------------------------------- #

DEFAULT_OUTPUT_DIR = "/content/ml_experiments"
DEFAULT_INPUT_CSV = "/content/ECD_total_Gauss.csv"
DEFAULT_LIBRARY = "/content/substituent_library.json"
DEFAULT_LABELS = ["E2E_H", "E2E_64"]

HAMMETT_COLS = [f"Pos_{i}" for i in range(1, 17)]
VDW_COLS = [f"VdW_{i}" for i in range(1, 17)]
H_VDW = 7.24  # unsubstituted-position placeholder, see train_pipeline.py

# --------------------------------------------------------------------------- #
# Publication-style axis labels, shared by every spectrum / ECD-intensity     #
# plot so they stay identical across the whole report.                       #
# --------------------------------------------------------------------------- #

XLABEL_WL = r"$\lambda$ / nm"
YLABEL_ECD = r"$R\cdot10^{40}$ / esu cm erg G$^{-1}$"
LEGEND_FONTSIZE = 18  # larger legend text for figures meant to go straight into a paper


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_run(output_dir: str, label: str) -> Optional[dict]:
    model_dir = os.path.join(output_dir, "models", label)
    history_path = os.path.join(output_dir, "histories", f"{label}_history.json")
    metrics_csv = os.path.join(model_dir, f"{label}_test_molecule_metrics.csv")
    y_true_path = os.path.join(model_dir, "test", "Y_test_true.npy")
    y_pred_path = os.path.join(model_dir, "test", "Y_test_pred.npy")
    wl_path = os.path.join(model_dir, "WL_MODEL.npy")

    # history.json is NOT required: it only feeds the per-epoch loss-curve
    # plot. A "packaged" export of a model (predictions + metrics CSV, no
    # training log) can still run every other part of the report.
    required = [metrics_csv, y_true_path, y_pred_path, wl_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"[{label}] skipped — missing artifacts:\n  " + "\n  ".join(missing))
        return None

    history = None
    best_epoch = None
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        best_epoch = int(np.argmin(history["val_loss"]) + 1)
    else:
        print(f"[{label}] no histories/{label}_history.json found — the loss-curve plot will be "
              "skipped, everything else runs normally.")

    train_config = None
    train_config_path = os.path.join(model_dir, f"{label}_training_config.json")
    if os.path.exists(train_config_path):
        with open(train_config_path) as f:
            train_config = json.load(f)

    cv_summary = None
    cv_summary_path = os.path.join(output_dir, "cross_validation", f"{label}_cv_summary.json")
    if os.path.exists(cv_summary_path):
        with open(cv_summary_path) as f:
            cv_summary = json.load(f)

    return {
        "label": label,
        "model_dir": model_dir,
        "history": history,
        "train_config": train_config,
        "cv_summary": cv_summary,
        "best_epoch": best_epoch,
        "wl_model": np.load(wl_path),
        "y_true": np.load(y_true_path),
        "y_pred": np.load(y_pred_path),
        "molecule_metrics": pd.read_csv(metrics_csv),
    }


def load_substituent_library(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"Substituent library not found at {path} — substituent error analysis will be skipped.")
        return None
    with open(path) as f:
        return json.load(f)


def hammett_to_name(value: float, library: dict, tol: float = 1e-3) -> str:
    names = library["names"]
    key = f"{value:g}"
    if key in names:
        return names[key]
    best_key = min(names, key=lambda k: abs(float(k) - value))
    if abs(float(best_key) - value) < tol:
        return names[best_key]
    return f"unknown({value:g})"


def attach_substituents(molecule_metrics: pd.DataFrame, input_csv: str, library: dict) -> pd.DataFrame:
    """Adds a 'substituents' column (sorted, de-duplicated substituent names
    present in each molecule) by looking up dataset_index in the original CSV."""
    df = pd.read_csv(input_csv, sep=";", usecols=["Archivo"] + HAMMETT_COLS + VDW_COLS)

    substituent_lists = []
    n_unmatched = 0
    for idx in molecule_metrics["dataset_index"]:
        row = df.iloc[idx]
        names = set()
        for i in range(1, 17):
            if row[f"VdW_{i}"] != H_VDW:
                name = hammett_to_name(row[f"Pos_{i}"], library)
                if name.startswith("unknown"):
                    n_unmatched += 1
                names.add(name)
        substituent_lists.append(sorted(names))

    if n_unmatched:
        print(f"WARNING: {n_unmatched} substituted position(s) did not match any known "
              "substituent in the library (kept as 'unknown(...)').")

    out = molecule_metrics.copy()
    out["substituents"] = substituent_lists
    return out


# --------------------------------------------------------------------------- #
# 1. Loss curves                                                              #
# --------------------------------------------------------------------------- #

def plot_loss_curves(run: dict) -> None:
    h, label, best_epoch = run["history"], run["label"], run["best_epoch"]

    if h is None:
        print(f"[{label}] no training history available — skipping loss-curve plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(h["loss"], label="Training")
    axes[0].plot(h["val_loss"], label="Validation")
    axes[0].axvline(best_epoch - 1, ls="--", color="grey", label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE loss")
    axes[0].set_title(f"{label} — loss"); axes[0].legend()

    axes[1].plot(h["mae"], label="Training")
    axes[1].plot(h["val_mae"], label="Validation")
    axes[1].axvline(best_epoch - 1, ls="--", color="grey", label=f"Best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
    axes[1].set_title(f"{label} — MAE"); axes[1].legend()

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1.5. Cross-validation summary (mean +/- std across folds)                   #
# --------------------------------------------------------------------------- #

def plot_cv_summary(run: dict) -> Optional[pd.DataFrame]:
    """Bar-with-error-bar view of the k-fold CV metrics (mean +/- std across
    folds) saved by train_pipeline.py, plus the single held-out TEST value
    for comparison -- if the two disagree a lot, that's worth investigating
    (e.g. the test set happened to be unusually easy/hard)."""
    cv_summary, label = run.get("cv_summary"), run["label"]
    if not cv_summary:
        print(f"[{label}] no cross_validation/{label}_cv_summary.json found -- this run was trained "
              "without --k-folds cross-validation, or with an older version of train_pipeline.py. "
              "Skipping the CV uncertainty plot.")
        return None

    metrics = cv_summary["metrics"] if "metrics" in cv_summary else cv_summary
    k_folds = cv_summary.get("k_folds", len(next(iter(metrics.values()))["values"]))

    test_metric_map = {
        "r2": run["molecule_metrics"]["R2"].mean(),
        "rmse": run["molecule_metrics"]["RMSE"].mean(),
        "mae": run["molecule_metrics"]["MAE"].mean(),
        "cosine": run["molecule_metrics"]["cosine_similarity"].mean(),
    }

    names = ["cosine", "r2", "rmse", "mae"]
    titles = ["Cosine similarity", "R²", "RMSE", "MAE"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, name, title in zip(axes, names, titles):
        mean, std = metrics[name]["mean"], metrics[name]["std"]
        values = metrics[name]["values"]

        ax.bar([0], [mean], yerr=[std], capsize=8, color="#4472C4", alpha=0.85,
               width=0.5, error_kw={"linewidth": 2.5, "ecolor": "#1A1A1A"})
        ax.scatter([0] * len(values), values, color="#1A1A1A", s=40, zorder=5, label="Individual folds")
        ax.scatter([0.55], [test_metric_map[name]], color="#C0272D", s=140, marker="D",
                   zorder=5, label="Held-out test")

        ax.set_xlim(-0.5, 1.0)
        ax.set_xticks([])
        ax.set_title(f"{title}\nCV: {mean:.4f} ± {std:.4f}", fontsize=15, fontweight="bold")
        ax.tick_params(axis="y", labelsize=13)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, axis="y")
        if name == "cosine":
            ax.legend(fontsize=11, loc="lower right")

    fig.suptitle(f"{label} — {k_folds}-fold cross-validation (mean ± std across folds) vs. held-out test",
                 fontsize=19, fontweight="bold")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"[{label}] {k_folds}-fold CV summary:")
    rows = []
    for name, title in zip(names, titles):
        m = metrics[name]
        rows.append({"metric": title, "cv_mean": m["mean"], "cv_std": m["std"],
                      "held_out_test": test_metric_map[name]})
        print(f"    {title:>18}: CV = {m['mean']:.4f} ± {m['std']:.4f}   |   test = {test_metric_map[name]:.4f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Precision overview                                                       #
# --------------------------------------------------------------------------- #

def plot_precision_overview(run: dict) -> None:
    y_true, y_pred = run["y_true"], run["y_pred"]
    mm, label = run["molecule_metrics"], run["label"]

    global_r2 = r2_score(y_true.ravel(), y_pred.ravel())
    global_rmse = np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    axes[0].hexbin(y_true.ravel(), y_pred.ravel(), gridsize=60, mincnt=1, cmap="viridis")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel(f"True {YLABEL_ECD}"); axes[0].set_ylabel(f"Predicted {YLABEL_ECD}")
    axes[0].set_title(f"Pointwise: R²={global_r2:.4f}, RMSE={global_rmse:.3f}")

    axes[1].hist(mm["cosine_similarity"], bins=30, color="#55A868")
    axes[1].axvline(mm["cosine_similarity"].median(), color="k", ls="--",
                     label=f"Median={mm['cosine_similarity'].median():.3f}")
    axes[1].set_xlabel("Per-molecule cosine similarity"); axes[1].set_ylabel("Count"); axes[1].legend()
    axes[1].set_title("Per-molecule cosine distribution")

    axes[2].hist(mm["R2"], bins=30, color="#4C72B0")
    axes[2].axvline(mm["R2"].median(), color="k", ls="--", label=f"Median={mm['R2'].median():.3f}")
    axes[2].set_xlabel("Per-molecule R²"); axes[2].set_ylabel("Count"); axes[2].legend()
    axes[2].set_title("Per-molecule R² distribution")

    axes[3].hist(mm["RMSE"], bins=30, color="#DD8452")
    axes[3].axvline(mm["RMSE"].median(), color="k", ls="--", label=f"Median={mm['RMSE'].median():.3f}")
    axes[3].set_xlabel("Per-molecule RMSE"); axes[3].set_ylabel("Count"); axes[3].legend()
    axes[3].set_title("Per-molecule RMSE distribution")

    fig.suptitle(f"{label} — test-set precision overview", fontsize=13)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"[{label}] test set: n={len(mm)} molecules | "
          f"mean cosine={mm['cosine_similarity'].mean():.4f} | "
          f"mean R²={mm['R2'].mean():.4f} | mean RMSE={mm['RMSE'].mean():.4f}")


# --------------------------------------------------------------------------- #
# 3. Random spectra                                                           #
# --------------------------------------------------------------------------- #

def plot_random_spectra(run: dict, n: int = 6, seed: int = 42,
                         ylim: Optional[tuple] = (-400.0, 400.0)) -> None:
    y_true, y_pred, wl = run["y_true"], run["y_pred"], run["wl_model"]
    mm, label = run["molecule_metrics"].reset_index(drop=True), run["label"]

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(mm), size=min(n, len(mm)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.2 * nrows))
    axes = np.asarray(axes).ravel()

    detail_lines = []

    for i, (ax, pos) in enumerate(zip(axes, picks), start=1):
        row = mm.iloc[pos]
        ax.plot(wl, y_true[pos], label="Real", linewidth=3, color="#1A1A1A")
        ax.plot(wl, y_pred[pos], label="Predicted", linewidth=3, color="#C0272D", linestyle="--")
        ax.axhline(0, color="#AAAAAA", linewidth=1, alpha=0.7)
        subs = row.get("substituents", None)
        sub_str = ", ".join(subs) if isinstance(subs, list) and subs else str(row.get("Archivo", ""))
        ax.set_title(f"#{i}", fontsize=18, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(XLABEL_WL, fontsize=14); ax.set_ylabel(YLABEL_ECD, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

        detail_lines.append(
            f"  #{i}: {row.get('Archivo', '')}"
            f"{'  [' + sub_str + ']' if sub_str else ''} | "
            f"cos={row['cosine_similarity']:.4f}  "
            f"R²={row['R2']:.4f}  "
            f"RMSE={row['RMSE']:.4f}"
        )

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(f"{label} — random test-set spectra", fontsize=19, fontweight="bold")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"\n[{label}] Random test-set spectra:")
    for line in detail_lines:
        print(line)


# --------------------------------------------------------------------------- #
# 4. Metric-based representative spectra (cosine similarity by default)       #
# --------------------------------------------------------------------------- #
#
# The test set is divided into quartiles according to a selectable metric.
#
# Metrics:
#   RMSE  -> lower is better
#   MAE   -> lower is better
#   R2    -> higher is better
#   cosine_similarity -> higher is better
#
# Q1 always corresponds to the best-performing 25% and Q4 to the
# worst-performing 25%, regardless of the direction of the metric.
#
# Within each quartile, the representative molecule is the one closest
# to the median value of that metric. This avoids showing an extreme
# example and gives a representative spectrum for each performance regime.
# --------------------------------------------------------------------------- #

METRIC_INFO = {
    "RMSE": {
        "column": "RMSE",
        "direction": "lower",
        "label": "RMSE",
        "unit": ""
    },
    "MAE": {
        "column": "MAE",
        "direction": "lower",
        "label": "MAE",
        "unit": ""
    },
    "R2": {
        "column": "R2",
        "direction": "higher",
        "label": "R²",
        "unit": ""
    },
    "cosine_similarity": {
        "column": "cosine_similarity",
        "direction": "higher",
        "label": "Cosine similarity",
        "unit": ""
    },
}


def plot_quartile_spectra(
    run: dict,
    metric: str = "cosine_similarity",
    ylim: Optional[tuple] = (-400.0, 400.0),
) -> None:
    """
    Plot representative true-vs-predicted spectra for the four quartiles
    of a selected test-set metric.

    Parameters
    ----------
    run : dict
        Loaded model run.
    metric : str
        One of:
            "RMSE"
            "MAE"
            "R2"
            "cosine_similarity"
    ylim : tuple or None
        Fixed y-axis range applied to all four panels, so quartiles are
        directly comparable at a glance (publication-figure style). Pass
        None to let each panel auto-scale independently instead.

    Q1 = best 25%
    Q2 = 25-50%
    Q3 = 50-75%
    Q4 = worst 25%

    The representative molecule in each quartile is the molecule closest
    to that quartile's median metric value.
    """

    if metric not in METRIC_INFO:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from: {list(METRIC_INFO.keys())}"
        )

    y_true = run["y_true"]
    y_pred = run["y_pred"]
    wl = run["wl_model"]
    mm = run["molecule_metrics"].reset_index(drop=True)
    label = run["label"]

    info = METRIC_INFO[metric]
    column = info["column"]
    direction = info["direction"]
    metric_label = info["label"]

    if column not in mm.columns:
        print(
            f"[{label}] metric '{column}' not found in molecule metrics. "
            "Skipping quartile analysis."
        )
        return

    # ------------------------------------------------------------------ #
    # Remove invalid metric values                                       #
    # ------------------------------------------------------------------ #

    valid = mm[column].notna() & np.isfinite(mm[column])

    if not valid.all():
        print(
            f"[{label}] warning: {(~valid).sum()} molecules have invalid "
            f"{column} values and will be excluded from quartile analysis."
        )

    mm_valid = mm.loc[valid].copy()

    # ------------------------------------------------------------------ #
    # Sort so that Q1 is ALWAYS the best quartile                       #
    # ------------------------------------------------------------------ #

    ascending = direction == "lower"

    mm_sorted = mm_valid.sort_values(
        column,
        ascending=ascending
    ).reset_index()

    n = len(mm_sorted)

    if n < 4:
        print(f"[{label}] too few molecules for quartile analysis.")
        return

    # Split into four approximately equal groups.
    quartiles = np.array_split(mm_sorted, 4)

    quartile_names = [
        "Q1 — best 25%",
        "Q2",
        "Q3",
        "Q4 — worst 25%",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    detail_lines = []

    for ax, qname, subset in zip(
        axes,
        quartile_names,
        quartiles
    ):

        # -------------------------------------------------------------- #
        # Representative = closest to quartile median                   #
        # -------------------------------------------------------------- #

        target = subset[column].median()

        local_pos = (
            subset[column] - target
        ).abs().idxmin()

        row = subset.loc[local_pos]

        # Original position in y_true / y_pred
        pos = int(row["index"])

        # -------------------------------------------------------------- #
        # Spectrum                                                       #
        # -------------------------------------------------------------- #

        ax.plot(
            wl,
            y_true[pos],
            label="Real",
            linewidth=3,
            color="#1A1A1A"
        )

        ax.plot(
            wl,
            y_pred[pos],
            label="Predicted",
            linewidth=3,
            color="#C0272D",
            linestyle="--"
        )

        ax.axhline(
            0,
            color="#AAAAAA",
            linewidth=1,
            alpha=0.7
        )

        ax.set_xlim(
            float(wl.min()),
            float(wl.max())
        )

        # -------------------------------------------------------------- #
        # Substituent information                                        #
        # -------------------------------------------------------------- #

        subs = row.get("substituents", None)

        if isinstance(subs, list) and subs:
            sub_str = ", ".join(subs)
        else:
            sub_str = str(row.get("Archivo", ""))

        # -------------------------------------------------------------- #
        # Metrics                                                        #
        # -------------------------------------------------------------- #

        ax.set_title(
            qname,
            fontsize=18,
            fontweight="bold"
        )

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xlabel(
            XLABEL_WL,
            fontsize=15
        )

        ax.set_ylabel(
            YLABEL_ECD,
            fontsize=15
        )

        ax.tick_params(
            axis="both",
            labelsize=14
        )

        ax.legend(
            fontsize=LEGEND_FONTSIZE
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            alpha=0.2
        )

        detail_lines.append(
            f"  {qname}: "
            f"{row.get('Archivo', '')}"
            f"{'  [' + sub_str + ']' if sub_str else ''} | "
            f"cos={row['cosine_similarity']:.4f}  "
            f"R²={row['R2']:.4f}  "
            f"RMSE={row['RMSE']:.4f}  "
            f"MAE={row['MAE']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Overall title                                                      #
    # ------------------------------------------------------------------ #

    fig.suptitle(
        f"{label} — representative test spectra by {metric_label}\n"
        f"Q1 = best 25% | Q4 = worst 25%",
        fontsize=20,
        fontweight="bold"
    )

    fig.tight_layout()

    plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Print representatives                                              #
    # ------------------------------------------------------------------ #

    print(
        f"\n[{label}] Representative spectra ranked by {metric_label}"
    )

    for line in detail_lines:
        print(line)

# --------------------------------------------------------------------------- #
# 5. Error analysis by substituent                                            #
# --------------------------------------------------------------------------- #

def error_by_substituent(run: dict, save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    mm, label = run["molecule_metrics"], run["label"]
    if "substituents" not in mm.columns:
        print(f"[{label}] no substituent library provided — skipping error-by-substituent analysis.")
        return None

    # One row per (molecule, substituent) pair. A molecule contributes to
    # every substituent it contains (deduplicated), so a bi-substituted
    # molecule's error counts toward both of its substituents' statistics.
    exploded = mm.explode("substituents").rename(columns={"substituents": "substituent"})
    exploded = exploded.dropna(subset=["substituent"])

    summary = (
        exploded.groupby("substituent")
        .agg(n_molecules=("dataset_index", "nunique"),
             mean_R2=("R2", "mean"), median_R2=("R2", "median"),
             mean_RMSE=("RMSE", "mean"), median_RMSE=("RMSE", "median"), std_RMSE=("RMSE", "std"),
             mean_MAE=("MAE", "mean"), median_MAE=("MAE", "median"),
             mean_cosine=("cosine_similarity", "mean"), median_cosine=("cosine_similarity", "median"),
             std_cosine=("cosine_similarity", "std"))
        .sort_values("mean_cosine", ascending=False)
    )

    order = summary.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    cosine_groups = [exploded.loc[exploded["substituent"] == s, "cosine_similarity"].values for s in order]
    axes[0].boxplot(cosine_groups, labels=order, showmeans=True)
    axes[0].set_ylabel("Per-molecule cosine similarity"); axes[0].set_title(f"{label} — cosine by substituent")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].grid(alpha=0.2, axis="y")

    rmse_groups = [exploded.loc[exploded["substituent"] == s, "RMSE"].values for s in order]
    axes[1].boxplot(rmse_groups, labels=order, showmeans=True)
    axes[1].set_ylabel("Per-molecule RMSE"); axes[1].set_title(f"{label} — RMSE by substituent")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].grid(alpha=0.2, axis="y")

    fig.suptitle("Error attribution: a molecule's error is counted once per distinct "
                 "substituent it contains", fontsize=10, y=1.02)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"\n[{label}] error by substituent — summary table (sorted best -> worst by mean cosine similarity):")
    print(summary.round(4).to_string())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{label}_error_by_substituent.csv")
        summary.round(6).to_csv(path)
        print(f"Saved: {path}")

    return summary


# --------------------------------------------------------------------------- #
# 6. Error vs. number of substituents                                         #
# --------------------------------------------------------------------------- #

def error_vs_nsust(run: dict) -> None:
    mm, label = run["molecule_metrics"], run["label"]
    if "nsust" not in mm.columns:
        return

    order = sorted(mm["nsust"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].boxplot([mm.loc[mm["nsust"] == n, "cosine_similarity"].values for n in order],
                     labels=order, showmeans=True)
    axes[0].set_xlabel("Number of substituents (nsust)"); axes[0].set_ylabel("Per-molecule cosine similarity")
    axes[0].set_title(f"{label} — cosine vs. molecular complexity")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].boxplot([mm.loc[mm["nsust"] == n, "RMSE"].values for n in order], labels=order, showmeans=True)
    axes[1].set_xlabel("Number of substituents (nsust)"); axes[1].set_ylabel("Per-molecule RMSE")
    axes[1].set_title(f"{label} — RMSE vs. molecular complexity")
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Error vs. position of the lowest-energy (longest-wavelength) band        #
# --------------------------------------------------------------------------- #
#
# The lowest-energy electronic transition shows up as the reddest (largest
# wavelength) significant feature of the ECD spectrum. Molecules differ a
# lot in where that falls -- close to the 600 nm edge of the training
# window, or well inside it around 400-500 nm. This checks whether the
# model is systematically worse for molecules whose defining low-energy
# band sits near the edge (where it has seen less data, and where the true
# band could even be partly cut off by the 250-600 nm window).

DEFAULT_BAND_CENTERS = [300, 350, 400, 450, 500, 550]


def band_bin_edges(centers: list, bin_width: float = 50.0) -> tuple:
    """Bin edges for bins CENTERED on the given values (e.g. center=300,
    width=50 -> bin (275, 325]), plus the center->label mapping."""
    half = bin_width / 2
    centers = sorted(centers)
    edges = [centers[0] - half] + [c + half for c in centers]
    return edges, [str(c) for c in centers]


def find_reddest_band(spectrum: np.ndarray, wl: np.ndarray, threshold_frac: float = 0.10) -> tuple:
    """Wavelength of the reddest (largest-wavelength) significant peak or
    trough of a spectrum, plus whether that feature sits right at the 600 nm
    edge (a sign the true band may extend beyond the modeled window)."""
    max_abs = np.max(np.abs(spectrum))
    if max_abs < 1e-9:
        return np.nan, False

    d = np.diff(spectrum)
    interior = np.where(np.diff(np.sign(d)) != 0)[0] + 1
    candidates = np.unique(np.concatenate([interior, [0, len(spectrum) - 1]]))
    significant = candidates[np.abs(spectrum[candidates]) > threshold_frac * max_abs]
    if len(significant) == 0:
        significant = np.array([int(np.argmax(np.abs(spectrum)))])

    reddest_idx = int(significant.max())
    edge_clipped = reddest_idx == len(spectrum) - 1
    return float(wl[reddest_idx]), edge_clipped


def attach_band_position(run: dict, threshold_frac: float = 0.10) -> pd.DataFrame:
    """Adds 'reddest_band_nm' and 'band_at_edge' columns to run['molecule_metrics']
    (in place, cached: safe to call more than once) and returns it."""
    mm = run["molecule_metrics"]
    if "reddest_band_nm" in mm.columns:
        return mm

    y_true, wl = run["y_true"], run["wl_model"]
    mm = mm.reset_index(drop=True)
    band_wl, edge_clipped = zip(*(find_reddest_band(y_true[i], wl, threshold_frac) for i in range(len(mm))))
    mm["reddest_band_nm"] = band_wl
    mm["band_at_edge"] = edge_clipped
    run["molecule_metrics"] = mm
    return mm


def error_vs_band_position(run: dict, threshold_frac: float = 0.10, save_dir: Optional[str] = None,
                            bin_centers: list = DEFAULT_BAND_CENTERS, bin_width: float = 50.0) -> Optional[pd.DataFrame]:
    label = run["label"]
    mm = attach_band_position(run, threshold_frac)

    # scatter: continuous relationship, with Spearman correlation
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(mm["reddest_band_nm"], mm["cosine_similarity"], alpha=0.4, s=15, color="#55A868")
    rho_cos, p_cos = spearmanr(mm["reddest_band_nm"], mm["cosine_similarity"])
    axes[0].set_xlabel("Wavelength of reddest significant band (nm)")
    axes[0].set_ylabel("Per-molecule cosine similarity")
    axes[0].set_title(f"{label} — cosine vs. band position (Spearman ρ={rho_cos:.3f}, p={p_cos:.1e})")

    axes[1].scatter(mm["reddest_band_nm"], mm["RMSE"], alpha=0.4, s=15)
    rho_rmse, p_rmse = spearmanr(mm["reddest_band_nm"], mm["RMSE"])
    axes[1].set_xlabel("Wavelength of reddest significant band (nm)")
    axes[1].set_ylabel("Per-molecule RMSE")
    axes[1].set_title(f"{label} — RMSE vs. band position (Spearman ρ={rho_rmse:.3f}, p={p_rmse:.1e})")

    fig.tight_layout()
    plt.show()
    plt.close(fig)

    # binned view: bins CENTERED on bin_centers (e.g. 300, 350, ..., 550),
    # not on arbitrary range edges -- molecules with a band outside
    # [min(centers)-width/2, max(centers)+width/2] fall in no bin and are
    # excluded from this binned view (they're still in the scatter above).
    edges, labels = band_bin_edges(bin_centers, bin_width)
    mm["band_bin"] = pd.cut(mm["reddest_band_nm"], bins=edges, labels=labels, include_lowest=True)

    summary = (
        mm.groupby("band_bin", observed=True)
        .agg(n_molecules=("dataset_index", "nunique"),
             pct_edge_clipped=("band_at_edge", "mean"),
             mean_R2=("R2", "mean"), median_R2=("R2", "median"),
             mean_RMSE=("RMSE", "mean"), median_RMSE=("RMSE", "median"),
             mean_MAE=("MAE", "mean"), median_MAE=("MAE", "median"),
             mean_cosine=("cosine_similarity", "mean"), median_cosine=("cosine_similarity", "median"))
    )
    summary["pct_edge_clipped"] = (summary["pct_edge_clipped"] * 100).round(1)

    order = summary.index.astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cosine_groups = [mm.loc[mm["band_bin"].astype(str) == b, "cosine_similarity"].values for b in order]
    axes[0].boxplot(cosine_groups, labels=order, showmeans=True)
    axes[0].set_xlabel("Reddest significant band (nm)"); axes[0].set_ylabel("Per-molecule cosine similarity")
    axes[0].set_title(f"{label} — cosine by band-position bin")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.2, axis="y")

    rmse_groups = [mm.loc[mm["band_bin"].astype(str) == b, "RMSE"].values for b in order]
    axes[1].boxplot(rmse_groups, labels=order, showmeans=True)
    axes[1].set_xlabel("Reddest significant band (nm)"); axes[1].set_ylabel("Per-molecule RMSE")
    axes[1].set_title(f"{label} — RMSE by band-position bin")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.2, axis="y")

    fig.suptitle(f"{label} — does the model do worse for molecules whose lowest-energy band "
                 "sits near the edge of the window?", fontsize=10, y=1.03)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"\n[{label}] error by position of the reddest significant band "
          f"(threshold={threshold_frac:.0%} of that molecule's own peak amplitude):")
    print(summary.round(4).to_string())
    print("(pct_edge_clipped: % of molecules in that bin whose reddest feature sits exactly at "
          "600 nm -- their true lowest-energy band may extend beyond the modeled window, so take "
          "that bin's numbers with extra caution.)")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{label}_error_by_band_position.csv")
        summary.to_csv(path)
        print(f"Saved: {path}")

    return summary


def plot_band_position_spectra(run: dict, threshold_frac: float = 0.10,
                                bin_centers: list = DEFAULT_BAND_CENTERS, bin_width: float = 50.0,
                                ylim: Optional[tuple] = (-400.0, 400.0)) -> None:
    """One representative true-vs-predicted spectrum per band-position bin
    (the molecule closest to that bin's median cosine similarity), so you
    can actually SEE whether a band near 550 nm is predicted as well as one
    near 300 nm, not just read it off a boxplot."""
    y_true, y_pred, wl, label = run["y_true"], run["y_pred"], run["wl_model"], run["label"]
    mm = attach_band_position(run, threshold_frac).reset_index(drop=True)

    edges, labels = band_bin_edges(bin_centers, bin_width)
    mm["band_bin"] = pd.cut(mm["reddest_band_nm"], bins=edges, labels=labels, include_lowest=True)

    bins_present = [b for b in mm["band_bin"].cat.categories if (mm["band_bin"] == b).any()]
    if not bins_present:
        print(f"[{label}] no molecules with a detectable band — skipping band-position spectra.")
        return

    ncols = 3
    nrows = int(np.ceil(len(bins_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.2 * nrows))
    axes = np.asarray(axes).ravel()

    detail_lines = []

    for ax, b in zip(axes, bins_present):
        subset = mm[mm["band_bin"] == b]
        target = subset["cosine_similarity"].median()
        pos = int((subset["cosine_similarity"] - target).abs().idxmin())
        row = mm.iloc[pos]

        ax.plot(wl, y_true[pos], label="Real", linewidth=3, color="#1A1A1A")
        ax.plot(wl, y_pred[pos], label="Predicted", linewidth=3, color="#C0272D", linestyle="--")
        ax.axhline(0, color="#AAAAAA", linewidth=1, alpha=0.7)
        ax.axvline(row["reddest_band_nm"], color="#4472C4", ls=":", linewidth=2,
                   label=f"Reddest band ({row['reddest_band_nm']:.0f} nm)")
        subs = row.get("substituents", None)
        sub_str = ", ".join(subs) if isinstance(subs, list) and subs else str(row.get("Archivo", ""))
        edge_flag = " ⚠" if row["band_at_edge"] else ""
        ax.set_title(f"Band ≈{b} nm (n={len(subset)}){edge_flag}", fontsize=16, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(XLABEL_WL, fontsize=14); ax.set_ylabel(YLABEL_ECD, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

        detail_lines.append(
            f"  Band ≈{b} nm: {row.get('Archivo', '')}"
            f"{'  [' + sub_str + ']' if sub_str else ''}"
            f"{'  (at window edge)' if row['band_at_edge'] else ''} | "
            f"cos={row['cosine_similarity']:.4f}  "
            f"R²={row['R2']:.4f}  "
            f"RMSE={row['RMSE']:.4f}"
        )

    for ax in axes[len(bins_present):]:
        ax.axis("off")

    fig.suptitle(f"{label} — representative spectrum per band-position bin "
                 "(molecule closest to that bin's median cosine similarity)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"\n[{label}] Representative spectra by band position:")
    for line in detail_lines:
        print(line)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def characterize(output_dir: str, input_csv: Optional[str], library_path: Optional[str],
                  labels: list, n_random: int, seed: int, save_tables_dir: Optional[str] = None) -> dict:
    library = load_substituent_library(library_path) if library_path else None
    runs = {}

    for label in labels:
        print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
        run = load_run(output_dir, label)
        if run is None:
            continue

        if library is not None and input_csv and os.path.exists(input_csv):
            run["molecule_metrics"] = attach_substituents(run["molecule_metrics"], input_csv, library)

        plot_loss_curves(run)
        plot_cv_summary(run)
        plot_precision_overview(run)
        plot_random_spectra(run, n=n_random, seed=seed)
        plot_quartile_spectra(run)
        error_by_substituent(run, save_dir=save_tables_dir)
        error_vs_nsust(run)
        error_vs_band_position(run, save_dir=save_tables_dir)
        plot_band_position_spectra(run)

        runs[label] = run

    if len(runs) > 1:
        print(f"\n{'=' * 90}\nSUMMARY ACROSS MODELS\n{'=' * 90}")
        comparison = pd.DataFrame([
            {"Model": lbl,
             "Mean_cosine": r["molecule_metrics"]["cosine_similarity"].mean(),
             "Median_cosine": r["molecule_metrics"]["cosine_similarity"].median(),
             "Mean_R2": r["molecule_metrics"]["R2"].mean(),
             "Mean_RMSE": r["molecule_metrics"]["RMSE"].mean()}
            for lbl, r in runs.items()
        ])
        print(comparison.round(4).to_string(index=False))

    return runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-training characterization report for E2E ECD models.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV,
                    help="Needed only for the error-by-substituent analysis.")
    p.add_argument("--library", default=DEFAULT_LIBRARY,
                    help="substituent_library.json. Needed only for the error-by-substituent analysis.")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, choices=["E2E_H", "E2E_64"])
    p.add_argument("--n-random", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-tables-dir", default=None,
                    help="If given, save the substituent and band-position summary tables as CSV here.")
    args, _unknown = p.parse_known_args()  # tolerate Colab/Jupyter's injected argv
    return args


if __name__ == "__main__":
    args = parse_args()
    characterize(args.output_dir, args.input_csv, args.library, args.labels, args.n_random, args.seed,
                 save_tables_dir=args.save_tables_dir)
