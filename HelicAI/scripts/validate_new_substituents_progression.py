#!/usr/bin/env python3
"""
validate_new_substituents_progression.py — Full generalization report for
new (never-seen-in-training) substituents, across a series of CSVs with
increasing substitution complexity (1, 2, 3, ... substituents per
molecule).

For each CSV: reconstructs the real (DFT) ECD spectrum, builds the 64
descriptors from the new-substituent dictionary, predicts with the trained
model, and computes R2 / RMSE / MAE / cosine similarity per compound.

Produces:
  1. Metric-progression plot: R2, RMSE, MAE, cosine vs. number of
     substituents (boxplots, one box per file/level).
  2. RMSE-quartile representative spectra (Q1=best 25% ... Q4=worst 25%),
     computed across ALL compounds from every file together, real vs
     predicted overlay.
  3. Four randomly chosen compounds PER substitution level, real vs
     predicted overlay (one figure per level).

Reuses validate_new_substituents.py's reconstruction/descriptor-building
code, so both scripts always treat "real" and "predicted" the same way.

Usage
-----
    python validate_new_substituents_progression.py \\
        --csv ECD_Nuevos_1sust.csv ECD_Nuevos_2sust.csv ECD_Nuevos_3sust.csv \\
              ECD_Nuevos_4sust.csv ECD_Nuevos_5sust.csv ECD_Nuevos_6sust.csv \\
        --new-library new_substituent_library.json \\
        --model-dir E2E_64_package --label E2E_64 \\
        --output-dir validation_progression
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors

from predict_ecd import load_model_package
from validate_new_substituents import (
    HAMMETT_COLS, NM_COLS, R_COLS, WL_MIN, WL_MAX,
    apply_hammett_remap, build_64_descriptor_vector, load_new_library,
    reconstruct_real_spectrum, substituent_names,
)

# Font sizes / axis labels / line style, matching the project's established
# figure style (same constants as the HelicAI vs. xTB-sTDA comparison plot).
TITLE_FS = 18
LABEL_FS = 20
TICK_FS = 18
LEGEND_FS = 16
SUPTITLE_FS = 23
LINEWIDTH = 2.5
LAMBDA_LABEL = r"$\lambda$ / nm"
R_LABEL = r"$R\cdot10^{40}$ / esu cm erg G$^{-1}$"


def cosine_similarity(t: np.ndarray, p: np.ndarray, wl: np.ndarray) -> float:
    num = np.trapezoid(t * p, wl)
    den = np.sqrt(max(np.trapezoid(t * t, wl) * np.trapezoid(p * p, wl), 1e-30))
    return float(num / den)


def infer_nsust_level(csv_path: str, fallback_nsust_col: Optional[int] = None) -> int:
    m = re.search(r"(\d+)\s*sust", os.path.basename(csv_path), re.IGNORECASE)
    if m:
        return int(m.group(1))
    if fallback_nsust_col is not None:
        return int(fallback_nsust_col)
    raise ValueError(f"Could not infer the substitution level from filename '{csv_path}' "
                      "(expected something like '...2sust...') and no nsust column fallback given.")


# --------------------------------------------------------------------------- #
# Per-file evaluation                                                         #
# --------------------------------------------------------------------------- #

def evaluate_file(csv_path: str, library: dict, package, hammett_remap: Optional[dict] = None) -> tuple:
    df = pd.read_csv(csv_path, sep=";")
    missing = [c for c in ["Archivo"] + HAMMETT_COLS + NM_COLS + R_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    real_list, x_rows, records = [], [], []
    wl_model = None
    for _, row in df.iterrows():
        pos_values = apply_hammett_remap(row[HAMMETT_COLS].to_numpy(dtype=float), hammett_remap)
        nm_row = row[NM_COLS].to_numpy(dtype=float)
        r_row = row[R_COLS].to_numpy(dtype=float)
        wl_model, real_spectrum = reconstruct_real_spectrum(nm_row, r_row)
        x_rows.append(build_64_descriptor_vector(pos_values, library))
        real_list.append(real_spectrum)
        records.append({"Archivo": row["Archivo"], "substituents": substituent_names(pos_values, library),
                         "nsust_col": row.get("nsust", np.nan)})

    x = np.array(x_rows, dtype=np.float32)
    real = np.array(real_list, dtype=np.float32)
    x_scaled = package.scaler_x.transform(x)
    pred_scaled = package.model.predict(x_scaled, batch_size=128, verbose=0)
    pred = package.scaler_y.inverse_transform(pred_scaled)

    ad = package.applicability_domain
    ad_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ad["training_scaled_X"])
    ad_dist, _ = ad_nn.kneighbors(x_scaled)
    ad_threshold = float(ad["threshold"])

    for i, rec in enumerate(records):
        t, p = real[i], pred[i]
        rec.update({
            "R2": float(r2_score(t, p)),
            "RMSE": float(np.sqrt(mean_squared_error(t, p))),
            "MAE": float(mean_absolute_error(t, p)),
            "cosine_similarity": cosine_similarity(t, p, wl_model),
            "ad_distance": float(ad_dist[i, 0]),
            "ad_threshold": ad_threshold,
            "outside_applicability_domain": bool(ad_dist[i, 0] > ad_threshold),
        })

    return pd.DataFrame(records), real, pred, wl_model


# --------------------------------------------------------------------------- #
# Block 1: metric progression across substitution levels                      #
# --------------------------------------------------------------------------- #

def plot_metric_progression(combined: pd.DataFrame, output_dir: str, label: str) -> None:
    levels = sorted(combined["nsust_level"].unique())
    metrics = [("R2", "R²"), ("RMSE", "RMSE"), ("MAE", "MAE"), ("cosine_similarity", "Cosine similarity")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        groups = [combined.loc[combined["nsust_level"] == lv, col].values for lv in levels]
        bp = ax.boxplot(groups, labels=levels, showmeans=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#F5D5D3")
            patch.set_edgecolor("#C0272D")
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color("#C0272D")
        ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
        ax.set_ylabel(title, fontsize=LABEL_FS)
        ax.set_title(f"{title} vs. substitution level", fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"{label} — generalization to new substituents: metric progression",
                 fontsize=SUPTITLE_FS, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_metric_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")

    summary = combined.groupby("nsust_level").agg(
        n=("Archivo", "count"),
        mean_R2=("R2", "mean"), median_R2=("R2", "median"),
        mean_RMSE=("RMSE", "mean"), median_RMSE=("RMSE", "median"),
        mean_MAE=("MAE", "mean"), median_MAE=("MAE", "median"),
        mean_cosine=("cosine_similarity", "mean"), median_cosine=("cosine_similarity", "median"),
        pct_outside_AD=("outside_applicability_domain", "mean"),
    )
    summary["pct_outside_AD"] = (summary["pct_outside_AD"] * 100).round(1)
    print("\nMetric summary by substitution level:")
    print(summary.round(4).to_string())
    summary.to_csv(os.path.join(output_dir, f"{label}_metric_progression_summary.csv"))


# --------------------------------------------------------------------------- #
# Block 2: RMSE-quartile representative spectra, across ALL compounds         #
# --------------------------------------------------------------------------- #

def plot_quartile_spectra(combined: pd.DataFrame, combined_real: np.ndarray, combined_pred: np.ndarray,
                           wl: np.ndarray, output_dir: str, label: str) -> None:
    mm = combined.reset_index(drop=True)
    q25, q50, q75 = mm["cosine_similarity"].quantile([0.25, 0.50, 0.75])
    bins = {
        "Q1 (best 25% cosine)": mm[mm["cosine_similarity"] >= q75],
        "Q2": mm[(mm["cosine_similarity"] >= q50) & (mm["cosine_similarity"] < q75)],
        "Q3": mm[(mm["cosine_similarity"] >= q25) & (mm["cosine_similarity"] < q50)],
        "Q4 (worst 25% cosine)": mm[mm["cosine_similarity"] < q25],
    }

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    axes = axes.ravel()
    detail_lines = []
    for i, (ax, (qname, subset)) in enumerate(zip(axes, bins.items()), start=1):
        if subset.empty:
            ax.axis("off")
            continue
        target = subset["cosine_similarity"].median()
        pos = int((subset["cosine_similarity"] - target).abs().idxmin())
        row = mm.iloc[pos]

        ax.plot(wl, combined_real[pos], linewidth=LINEWIDTH, color="#2E2E2E", label="Real")
        ax.plot(wl, combined_pred[pos], linewidth=LINEWIDTH, color="#C0272D", linestyle="--", label="Predicted")
        ax.axhline(0, color="#888888", linewidth=1, alpha=0.6)
        ax.set_xlim(WL_MIN, WL_MAX)
        ad_flag = "  ⚠ outside AD" if row["outside_applicability_domain"] else ""
        ax.set_title(f"{qname}{ad_flag}\ncosine={row['cosine_similarity']:.3f}  R²={row['R2']:.3f}  RMSE={row['RMSE']:.3f}",
                     fontsize=TITLE_FS, fontweight="bold")
        ax.set_xlabel(LAMBDA_LABEL, fontsize=LABEL_FS)
        ax.set_ylabel(R_LABEL, fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS, framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        detail_lines.append(f"  {qname} -- {row['Archivo']} (nsust={row['nsust_level']}): {row['substituents']}")

    fig.suptitle(f"{label} — cosine-similarity-quartile representatives across ALL new-substituent compounds",
                 fontsize=SUPTITLE_FS, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_new_substituents_quartiles.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
    print("Where exactly each quartile representative is substituted:")
    for line in detail_lines:
        print(line)


# --------------------------------------------------------------------------- #
# Block 3: 4 random compounds PER substitution level                          #
# --------------------------------------------------------------------------- #

def plot_random_per_level(spectra_store: dict, output_dir: str, label: str,
                           n: int = 4, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    for level, (real, pred, wl, results_df) in sorted(spectra_store.items()):
        n_here = min(n, len(results_df))
        picks = rng.choice(len(results_df), size=n_here, replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(17, 12))
        axes = axes.ravel()
        detail_lines = []
        for i, (ax, pos) in enumerate(zip(axes, picks), start=1):
            row = results_df.iloc[pos]
            ax.plot(wl, real[pos], linewidth=LINEWIDTH, color="#2E2E2E", label="Real")
            ax.plot(wl, pred[pos], linewidth=LINEWIDTH, color="#C0272D", linestyle="--", label="Predicted")
            ax.axhline(0, color="#888888", linewidth=1, alpha=0.6)
            ax.set_xlim(WL_MIN, WL_MAX)
            ad_flag = "  ⚠ outside AD" if row["outside_applicability_domain"] else ""
            ax.set_title(f"Compound {i}{ad_flag}\ncosine={row['cosine_similarity']:.3f}  "
                         f"R²={row['R2']:.3f}  RMSE={row['RMSE']:.3f}", fontsize=TITLE_FS, fontweight="bold")
            ax.set_xlabel(LAMBDA_LABEL, fontsize=LABEL_FS)
            ax.set_ylabel(R_LABEL, fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.legend(fontsize=LEGEND_FS, framealpha=0.85)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2)
            detail_lines.append(f"  Compound {i} -- {row['Archivo']}: {row['substituents']}")

        for ax in axes[n_here:]:
            ax.axis("off")

        fig.suptitle(f"{label} — {n_here} random compounds with {level} substituent(s)",
                     fontsize=SUPTITLE_FS, fontweight="bold", y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{label}_random_nsust{level}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved: {path}")
        print(f"Where exactly each random nsust={level} compound is substituted:")
        for line in detail_lines:
            print(line)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run_full_validation(csv_paths: list, new_library_path: str, model_dir: str, label: str,
                         output_dir: str, hammett_remap: Optional[dict] = None,
                         n_random_per_level: int = 4, seed: int = 42) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    library = load_new_library(new_library_path)
    package = load_model_package(model_dir, label)

    all_results, all_real, all_pred = [], [], []
    spectra_store = {}
    wl = None

    for csv_path in csv_paths:
        results_df, real, pred, wl = evaluate_file(csv_path, library, package, hammett_remap)
        fallback = results_df["nsust_col"].iloc[0] if results_df["nsust_col"].notna().all() else None
        level = infer_nsust_level(csv_path, fallback_nsust_col=fallback)
        results_df["nsust_level"] = level
        results_df["source_file"] = os.path.basename(csv_path)

        print(f"{os.path.basename(csv_path)} (nsust={level}): {len(results_df)} compounds -- "
              f"mean R²={results_df['R2'].mean():.3f}, mean RMSE={results_df['RMSE'].mean():.3f}, "
              f"mean cosine={results_df['cosine_similarity'].mean():.3f}, "
              f"{int(results_df['outside_applicability_domain'].sum())} outside AD")

        all_results.append(results_df)
        all_real.append(real)
        all_pred.append(pred)
        spectra_store[level] = (real, pred, wl, results_df.reset_index(drop=True))

    combined = pd.concat(all_results, ignore_index=True)
    combined_real = np.vstack(all_real)
    combined_pred = np.vstack(all_pred)

    combined.to_csv(os.path.join(output_dir, f"{label}_new_substituents_full_results.csv"), index=False)
    print(f"\nTotal compounds evaluated: {len(combined)}")

    plot_metric_progression(combined, output_dir, label)
    plot_quartile_spectra(combined, combined_real, combined_pred, wl, output_dir, label)
    plot_random_per_level(spectra_store, output_dir, label, n=n_random_per_level, seed=seed)

    return combined, spectra_store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full new-substituent validation across substitution levels.")
    p.add_argument("--csv", nargs="+", required=True, help="All the ECD_Nuevos_Nsust.csv files.")
    p.add_argument("--new-library", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", default="E2E_64")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hammett-remap", default=None, help='JSON dict, e.g. \'{"-0.37": -0.24}\'.')
    p.add_argument("--n-random-per-level", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args, _unknown = p.parse_known_args()  # tolerate Colab/Jupyter's injected argv
    return args


if __name__ == "__main__":
    args = parse_args()
    remap = json.loads(args.hammett_remap) if args.hammett_remap else None
    run_full_validation(args.csv, args.new_library, args.model_dir, args.label, args.output_dir,
                         hammett_remap=remap, n_random_per_level=args.n_random_per_level, seed=args.seed)
