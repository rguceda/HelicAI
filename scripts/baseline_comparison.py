#!/usr/bin/env python3
"""
baseline_comparison.py — Does the E2E-H / E2E-64 neural network actually
beat trivial baselines, on the EXACT SAME leakage-free split (CV folds +
held-out test) that train_pipeline.py already built and saved?

Two baselines, both with zero "learned representation":

  1. 1-NEAREST-NEIGHBOR: predict a query molecule's spectrum as the real,
     DFT-computed spectrum of its closest match in the training pool
     (Euclidean distance in standardized descriptor space -- the same
     space/metric already used for the applicability domain elsewhere in
     this project). No training beyond storing the training set.
  2. LINEAR REGRESSION (Ridge, alpha=1.0): a plain multi-output linear map
     straight from the descriptors to the 100-point spectrum. No hidden
     layers, no nonlinearity, no interaction terms.

Both are evaluated with the exact same metrics (R2, RMSE, MAE, cosine) on
the exact same CV folds and held-out test set as the neural network, by
reading train_pipeline.py's saved split indices directly -- so "does the
extra complexity earn its keep" is a fair, apples-to-apples question.

Usage
-----
    python baseline_comparison.py \\
        --input-csv ECD_total_Gauss.csv \\
        --experiment-dir ml_experiments \\
        --output-dir baseline_results \\
        --labels E2E_H E2E_64
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from train_pipeline import (
    HAMMETT_COLS, DESCRIPTOR_COLS, calculate_metrics, interpolate_to_model_grid,
    load_dataset, reconstruct_ecd_spectra, validate_columns, validate_finite,
)

TITLE_FS, LABEL_FS, TICK_FS = 17, 14, 13
METHOD_COLORS = {"1-Nearest-Neighbor": "#4472C4", "Linear (Ridge)": "#C0272D",
                  "Neural network (E2E)": "#2E7D32"}
METHOD_ORDER = ["1-Nearest-Neighbor", "Linear (Ridge)", "Neural network (E2E)"]


# --------------------------------------------------------------------------- #
# Load the split train_pipeline.py already built (same folds, same test)     #
# --------------------------------------------------------------------------- #

def load_splits(splits_dir: str) -> tuple:
    trainval_idx = np.load(os.path.join(splits_dir, "trainval_indices.npy"))
    test_idx = np.load(os.path.join(splits_dir, "test_indices.npy"))
    folds = []
    i = 1
    while os.path.exists(os.path.join(splits_dir, f"fold{i}_train_indices.npy")):
        ftr = np.load(os.path.join(splits_dir, f"fold{i}_train_indices.npy"))
        fte = np.load(os.path.join(splits_dir, f"fold{i}_test_indices.npy"))
        folds.append((ftr, fte))
        i += 1
    if not folds:
        raise FileNotFoundError(f"No fold split files found in {splits_dir}. Run train_pipeline.py first "
                                 "-- this script reuses its exact split, it doesn't build its own.")
    return trainval_idx, test_idx, folds


# --------------------------------------------------------------------------- #
# The two baselines                                                           #
# --------------------------------------------------------------------------- #

def nn_baseline_fit_predict(x_train_s: np.ndarray, y_train: np.ndarray, x_query_s: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(x_train_s)
    _, idx = nn.kneighbors(x_query_s)
    return y_train[idx[:, 0]]


def linear_baseline_fit_predict(x_train_s: np.ndarray, y_train: np.ndarray, x_query_s: np.ndarray,
                                 alpha: float = 1.0) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(x_train_s, y_train)
    return model.predict(x_query_s)


def evaluate_baselines_cv(x: np.ndarray, y_ecd: np.ndarray, folds: list, wl_model: np.ndarray) -> dict:
    records = {"nn": [], "linear": []}
    for fold_i, (ftr_idx, fte_idx) in enumerate(folds, start=1):
        x_train, x_test = x[ftr_idx], x[fte_idx]
        y_train, y_test = y_ecd[ftr_idx], y_ecd[fte_idx]
        sx = StandardScaler().fit(x_train)
        x_train_s, x_test_s = sx.transform(x_train), sx.transform(x_test)

        y_nn = nn_baseline_fit_predict(x_train_s, y_train, x_test_s)
        m_nn = calculate_metrics(y_test, y_nn, wl_model)
        records["nn"].append({"fold": fold_i, "r2": m_nn["molecule"]["mean_r2"],
                               "rmse": m_nn["molecule"]["mean_rmse"], "mae": m_nn["molecule"]["mean_mae"],
                               "cosine": m_nn["molecule"]["mean_cosine"]})

        y_lin = linear_baseline_fit_predict(x_train_s, y_train, x_test_s)
        m_lin = calculate_metrics(y_test, y_lin, wl_model)
        records["linear"].append({"fold": fold_i, "r2": m_lin["molecule"]["mean_r2"],
                                   "rmse": m_lin["molecule"]["mean_rmse"], "mae": m_lin["molecule"]["mean_mae"],
                                   "cosine": m_lin["molecule"]["mean_cosine"]})
        print(f"  fold {fold_i}: 1-NN R2={m_nn['molecule']['mean_r2']:.4f}  "
              f"Linear R2={m_lin['molecule']['mean_r2']:.4f}")
    return {name: pd.DataFrame(recs) for name, recs in records.items()}


def evaluate_baselines_test(x: np.ndarray, y_ecd: np.ndarray, trainval_idx: np.ndarray,
                             test_idx: np.ndarray, wl_model: np.ndarray) -> dict:
    x_train, x_test = x[trainval_idx], x[test_idx]
    y_train, y_test = y_ecd[trainval_idx], y_ecd[test_idx]
    sx = StandardScaler().fit(x_train)
    x_train_s, x_test_s = sx.transform(x_train), sx.transform(x_test)

    y_nn = nn_baseline_fit_predict(x_train_s, y_train, x_test_s)
    y_lin = linear_baseline_fit_predict(x_train_s, y_train, x_test_s)
    return {"nn": calculate_metrics(y_test, y_nn, wl_model), "linear": calculate_metrics(y_test, y_lin, wl_model)}


def summarize(fold_df: pd.DataFrame) -> dict:
    return {metric: {"mean": float(fold_df[metric].mean()), "std": float(fold_df[metric].std(ddof=1))}
            for metric in ("r2", "rmse", "mae", "cosine")}


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #

def plot_comparison(summary: pd.DataFrame, output_dir: str) -> None:
    metrics = [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"), ("cosine", "Cosine similarity")]

    for label in summary["Label"].unique():
        sub = summary[summary["Label"] == label]
        methods_present = [m for m in METHOD_ORDER if m in sub["Method"].values]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.8))
        for ax, (metric, title) in zip(axes, metrics):
            means = [sub[sub["Method"] == m][f"CV_{metric}_mean"].iloc[0] for m in methods_present]
            stds = [sub[sub["Method"] == m][f"CV_{metric}_std"].iloc[0] for m in methods_present]
            colors = [METHOD_COLORS[m] for m in methods_present]
            x_pos = np.arange(len(methods_present))

            ax.bar(x_pos, means, yerr=stds, capsize=8, color=colors, alpha=0.85, width=0.6,
                   error_kw={"linewidth": 2.5, "ecolor": "#1A1A1A"})
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods_present, rotation=20, ha="right", fontsize=12)
            ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, axis="y")

        fig.suptitle(f"{label} — neural network vs. trivial baselines (CV mean ± std, same folds)",
                     fontsize=20, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(output_dir, f"{label}_baseline_comparison.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run_baseline_comparison(input_csv: str, experiment_dir: str, output_dir: str,
                             labels: tuple = ("E2E_H", "E2E_64")) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(input_csv)
    validate_columns(df)
    _, nm_matrix, rot_matrix = validate_finite(df)
    wl_grid, y_high_res = reconstruct_ecd_spectra(nm_matrix, rot_matrix)
    wl_model, y_ecd = interpolate_to_model_grid(wl_grid, y_high_res)

    splits_dir = os.path.join(experiment_dir, "splits")
    trainval_idx, test_idx, folds = load_splits(splits_dir)
    print(f"Loaded split from {splits_dir}: trainval={len(trainval_idx)}  test={len(test_idx)}  "
          f"folds={[len(fte) for _, fte in folds]}")

    inputs = {
        "E2E_H": df[HAMMETT_COLS].to_numpy(dtype=np.float32),
        "E2E_64": df[DESCRIPTOR_COLS].to_numpy(dtype=np.float32),
    }

    rows = []
    for label in labels:
        print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
        x = inputs[label]

        print("Cross-validation (same folds as the neural network)...")
        cv_results = evaluate_baselines_cv(x, y_ecd, folds, wl_model)
        for name, fold_df in cv_results.items():
            fold_df.to_csv(os.path.join(output_dir, f"{label}_{name}_cv_fold_metrics.csv"), index=False)

        print("Held-out test evaluation (same test set as the neural network)...")
        test_results = evaluate_baselines_test(x, y_ecd, trainval_idx, test_idx, wl_model)

        method_names = {"nn": "1-Nearest-Neighbor", "linear": "Linear (Ridge)"}
        for name, fold_df in cv_results.items():
            s = summarize(fold_df)
            tm = test_results[name]["molecule"]
            rows.append({
                "Label": label, "Method": method_names[name],
                **{f"CV_{m}_mean": s[m]["mean"] for m in s}, **{f"CV_{m}_std": s[m]["std"] for m in s},
                "Test_r2": tm["mean_r2"], "Test_rmse": tm["mean_rmse"],
                "Test_mae": tm["mean_mae"], "Test_cosine": tm["mean_cosine"],
            })

        # pull in the neural network's OWN already-computed CV + test numbers
        # (never re-trains it -- just reads what train_pipeline.py already saved)
        nn_cv_path = os.path.join(experiment_dir, "cross_validation", f"{label}_cv_summary.json")
        nn_metrics_path = os.path.join(experiment_dir, "models", label, f"{label}_metrics.json")
        if os.path.exists(nn_cv_path) and os.path.exists(nn_metrics_path):
            with open(nn_cv_path) as f:
                nn_cv = json.load(f)["metrics"]
            with open(nn_metrics_path) as f:
                nn_test = json.load(f)["test"]["molecule"]
            rows.append({
                "Label": label, "Method": "Neural network (E2E)",
                **{f"CV_{m}_mean": nn_cv[m]["mean"] for m in nn_cv}, **{f"CV_{m}_std": nn_cv[m]["std"] for m in nn_cv},
                "Test_r2": nn_test["mean_r2"], "Test_rmse": nn_test["mean_rmse"],
                "Test_mae": nn_test["mean_mae"], "Test_cosine": nn_test["mean_cosine"],
            })
        else:
            print(f"[{label}] neural network CV/test results not found under {experiment_dir} -- "
                  "showing baselines only for this label. Run train_pipeline.py first for the full comparison.")

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "baseline_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 90 + "\nSUMMARY\n" + "=" * 90)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {summary_path}")

    plot_comparison(summary, output_dir)
    return summary


DEFAULT_INPUT_CSV = "/content/ECD_total_Gauss.csv"
DEFAULT_EXPERIMENT_DIR = "/content/ml_experiments"
DEFAULT_OUTPUT_DIR = "/content/baseline_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare the E2E neural networks against trivial baselines "
                                             "on the exact same split.")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    p.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR,
                    help="The --output-dir you gave train_pipeline.py (reads its saved splits/ and "
                         "cross_validation/ automatically).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--labels", nargs="+", default=["E2E_H", "E2E_64"], choices=["E2E_H", "E2E_64"])
    args, _unknown = p.parse_known_args()  # tolerate Colab/Jupyter's injected argv
    return args


if __name__ == "__main__":
    args = parse_args()
    run_baseline_comparison(args.input_csv, args.experiment_dir, args.output_dir, tuple(args.labels))
