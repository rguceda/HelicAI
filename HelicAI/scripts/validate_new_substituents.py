#!/usr/bin/env python3
"""
validate_new_substituents.py — Real generalization test: predict the ECD
spectrum of compounds bearing substituents the model has NEVER seen (not in
the original 16-substituent vocabulary), and compare against their real,
DFT-computed spectra.

This is a stronger test than leave-one-substituent-out: LOSO holds out a
known substituent from training; here the substituent's chemistry (its
VdW/R+/R- values) was never in the training data's *vocabulary* at all.

Input files
-----------
1. A CSV of new compounds, same layout as ECD_total_Gauss.csv but WITHOUT
   VdW_i/Rplus_i/Rminus_i columns (those aren't known a priori for a new
   substituent -- that's the whole point):
       Archivo;nsust;Pos_1;...;Pos_16;R_1;nm_1;...;R_100;nm_100
   Pos_i is already the Hammett sigma of whatever sits at position i (0 for
   unsubstituted).
2. A small JSON dictionary (same schema as substituent_library.json) giving
   vdw/r_plus/r_minus for each new substituent's Hammett key, e.g.:
       {"descriptors": {"vdw": {"0.54": 42.74}, "r_plus": {...}, "r_minus": {...}},
        "names": {"0.54": "CF3"}, "defaults": {...}}
   IMPORTANT: a Hammett sigma is not a unique substituent ID -- two
   different substituents can share the same value. This script uses ONLY
   the new dictionary for these compounds; it never falls back to the
   original substituent_library.json.

Usage
-----
    python validate_new_substituents.py \\
        --new-compounds-csv new_compounds.csv \\
        --new-library new_substituent_library.json \\
        --model-dir E2E_64_package --label E2E_64 \\
        --output-dir validation_new_subs
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from predict_ecd import load_model_package

HC_EV_NM = 1239.841984
SIGMA_EV = 0.30
WL_MIN, WL_MAX = 250.0, 600.0
N_GRID = 2000
N_MODEL_POINTS = 100
N_POSITIONS = 16

HAMMETT_COLS = [f"Pos_{i}" for i in range(1, N_POSITIONS + 1)]
NM_COLS = [f"nm_{i}" for i in range(1, 101)]
R_COLS = [f"R_{i}" for i in range(1, 101)]


# --------------------------------------------------------------------------- #
# Real-spectrum reconstruction (identical method/constants to               #
# train_pipeline.py, so predicted and real are on the exact same footing)    #
# --------------------------------------------------------------------------- #

def reconstruct_real_spectrum(nm_row: np.ndarray, r_row: np.ndarray) -> tuple:
    e_grid = np.linspace(HC_EV_NM / WL_MAX, HC_EV_NM / WL_MIN, N_GRID)
    wl_grid = HC_EV_NM / e_grid
    order = np.argsort(wl_grid)
    wl_grid, e_grid = wl_grid[order], e_grid[order]

    valid = np.isfinite(nm_row) & np.isfinite(r_row) & (nm_row > 0)
    energies = HC_EV_NM / nm_row[valid]
    spectrum = np.zeros(N_GRID)
    for energy, strength in zip(energies, r_row[valid]):
        spectrum += strength * np.exp(-((e_grid - energy) / SIGMA_EV) ** 2)

    wl_model = np.linspace(WL_MIN, WL_MAX, N_MODEL_POINTS)
    return wl_model, np.interp(wl_model, wl_grid, spectrum)


# --------------------------------------------------------------------------- #
# Descriptor construction using ONLY the new-substituent dictionary          #
# --------------------------------------------------------------------------- #

def load_new_library(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def hammett_to_descriptors(value: float, library: dict, tol: float = 1e-3) -> tuple:
    """(vdw, r_plus, r_minus) for a given Hammett sigma, looked up in the
    NEW dictionary only. Raises a clear error if the value isn't there --
    silently falling back to a default would quietly corrupt the test."""
    d = library["defaults"]
    if abs(value) < 1e-9:
        return d["vdw"], d["r_plus"], d["r_minus"]

    vdw_dict = library["descriptors"]["vdw"]
    key = f"{value:g}"
    if key not in vdw_dict:
        best = min(vdw_dict, key=lambda k: abs(float(k) - value))
        if abs(float(best) - value) < tol:
            key = best
        else:
            raise ValueError(f"Hammett value {value} not found in the new substituent "
                              f"dictionary (closest: {best}). Known keys: {list(vdw_dict)}")
    return (vdw_dict[key], library["descriptors"]["r_plus"][key], library["descriptors"]["r_minus"][key])


def apply_hammett_remap(pos_values: np.ndarray, hammett_remap: dict, tol: float = 1e-6) -> np.ndarray:
    """Replaces any Pos_i value matching a key in hammett_remap (e.g.
    {-0.37: -0.24}, for correcting a mislabeled/updated Hammett sigma)
    with its mapped value, BEFORE descriptor lookup and BEFORE it's used
    as the Hammett feature itself -- so the corrected value is what the
    model actually sees, not just what the dictionary is keyed by."""
    if not hammett_remap:
        return pos_values
    out = pos_values.copy()
    for old, new in hammett_remap.items():
        out[np.abs(out - old) < tol] = new
    return out


def build_64_descriptor_vector(pos_values: np.ndarray, library: dict) -> np.ndarray:
    """pos_values: the 16 Pos_i Hammett values for one molecule (0 = H).
    Returns the (64,) vector in [Hammett]*16 + [VdW]*16 + [R+]*16 + [R-]*16 order."""
    hammett = np.asarray(pos_values, dtype=float)
    vdw = np.empty(N_POSITIONS)
    r_plus = np.empty(N_POSITIONS)
    r_minus = np.empty(N_POSITIONS)
    for i, v in enumerate(hammett):
        vdw[i], r_plus[i], r_minus[i] = hammett_to_descriptors(v, library)
    return np.concatenate([hammett, vdw, r_plus, r_minus])


def substituent_names(pos_values: np.ndarray, library: dict) -> str:
    names = library.get("names", {})
    parts = []
    for i, v in enumerate(pos_values, start=1):
        if abs(v) > 1e-9:
            key = f"{v:g}"
            name = names.get(key, key)
            parts.append(f"Pos{i}={name}")
    return ", ".join(parts) if parts else "(unsubstituted)"


# --------------------------------------------------------------------------- #
# Main validation                                                             #
# --------------------------------------------------------------------------- #

def validate(new_compounds_csv: str, new_library_path: str, model_dir: str, label: str,
             output_dir: str, hammett_remap: dict = None) -> pd.DataFrame:
    library = load_new_library(new_library_path)
    package = load_model_package(model_dir, label)

    df = pd.read_csv(new_compounds_csv, sep=";")
    missing = [c for c in ["Archivo"] + HAMMETT_COLS + NM_COLS + R_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    real_spectra, x_rows, labels_desc = [], [], []
    for _, row in df.iterrows():
        pos_values = apply_hammett_remap(row[HAMMETT_COLS].to_numpy(dtype=float), hammett_remap)
        nm_row = row[NM_COLS].to_numpy(dtype=float)
        r_row = row[R_COLS].to_numpy(dtype=float)

        wl_model, real_spectrum = reconstruct_real_spectrum(nm_row, r_row)
        x_rows.append(build_64_descriptor_vector(pos_values, library))
        real_spectra.append(real_spectrum)
        labels_desc.append(substituent_names(pos_values, library))

    x = np.array(x_rows, dtype=np.float32)
    real_spectra = np.array(real_spectra, dtype=np.float32)

    x_scaled = package.scaler_x.transform(x)
    pred_scaled = package.model.predict(x_scaled, batch_size=64, verbose=0)
    pred_spectra = package.scaler_y.inverse_transform(pred_scaled)

    ad = package.applicability_domain
    from sklearn.neighbors import NearestNeighbors
    ad_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ad["training_scaled_X"])
    ad_distances, _ = ad_nn.kneighbors(x_scaled)
    ad_threshold = float(ad["threshold"])

    def cosine_sim(t, p, wl):
        num = np.trapezoid(t * p, wl)
        den = np.sqrt(max(np.trapezoid(t * t, wl) * np.trapezoid(p * p, wl), 1e-30))
        return num / den

    records = []
    for i in range(len(df)):
        t, p = real_spectra[i], pred_spectra[i]
        records.append({
            "Archivo": df.iloc[i]["Archivo"],
            "substituents": labels_desc[i],
            "R2": float(r2_score(t, p)),
            "RMSE": float(np.sqrt(mean_squared_error(t, p))),
            "MAE": float(mean_absolute_error(t, p)),
            "cosine_similarity": float(cosine_sim(t, p, wl_model)),
            "ad_distance": float(ad_distances[i, 0]),
            "ad_threshold": ad_threshold,
            "outside_applicability_domain": bool(ad_distances[i, 0] > ad_threshold),
        })
    results = pd.DataFrame(records)

    print("\nGeneralization to never-before-seen substituents:")
    print(results.round(4).to_string(index=False))
    print(f"\nMean R²={results['R2'].mean():.4f}  Mean RMSE={results['RMSE'].mean():.4f}  "
          f"Mean cosine={results['cosine_similarity'].mean():.4f}")
    n_outside = int(results["outside_applicability_domain"].sum())
    if n_outside:
        print(f"WARNING: {n_outside}/{len(results)} compounds fall outside the model's applicability "
              "domain -- expect those predictions to be less reliable, by construction.")

    # overlay plots: real (solid) vs predicted (dashed), same style as the rest of the project
    ncols = 3
    nrows = int(np.ceil(len(df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= len(df):
            ax.axis("off")
            continue
        ax.plot(wl_model, real_spectra[i], linewidth=3, color="#2E2E2E", label="Real (DFT)")
        ax.plot(wl_model, pred_spectra[i], linewidth=3, color="#C0272D", linestyle="--", label="Predicted")
        ax.axhline(0, color="#888888", linewidth=1, alpha=0.6)
        ax.set_xlim(WL_MIN, WL_MAX)
        r = results.iloc[i]
        ad_flag = "  ⚠ outside AD" if r["outside_applicability_domain"] else ""
        ax.set_title(f"{df.iloc[i]['Archivo']}{ad_flag}\nR²={r['R2']:.3f}  RMSE={r['RMSE']:.3f}  "
                     f"cos={r['cosine_similarity']:.3f}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=13)
        ax.set_ylabel("ECD intensity", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

    fig.suptitle(f"{label} — generalization to never-seen substituents (real vs predicted)",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{label}_new_substituents_overlay.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    results_path = os.path.join(output_dir, f"{label}_new_substituents_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {fig_path}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the model against never-seen substituents.")
    p.add_argument("--new-compounds-csv", required=True)
    p.add_argument("--new-library", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", default="E2E_64")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hammett-remap", default=None,
                    help='JSON dict to correct mislabeled Hammett values before lookup, '
                         'e.g. \'{"-0.37": -0.24}\'.')
    args, _unknown = p.parse_known_args()  # tolerate Colab/Jupyter's injected argv
    return args


if __name__ == "__main__":
    args = parse_args()
    remap = json.loads(args.hammett_remap) if args.hammett_remap else None
    validate(args.new_compounds_csv, args.new_library, args.model_dir, args.label,
             args.output_dir, hammett_remap=remap)
