#!/usr/bin/env python3
"""
band_area_scaling_study.py — How does the strongest achievable ECD band
area (positive and negative) scale with the number of substituents,
from 1 up to all 16 positions?

METHODOLOGY (documented here precisely so results are reproducible and
defensible under peer review)
--------------------------------------------------------------------------
For every candidate molecule, the trained model predicts a 100-point ECD
spectrum R(lambda) on the standard wavelength grid lambda in [250, 600] nm
(the same grid used throughout training).

  POSITIVE BAND AREA  = integral of max(R(lambda), 0) d(lambda)
  NEGATIVE BAND AREA  = integral of min(R(lambda), 0) d(lambda)   (<= 0 by construction)

Both integrals use the trapezoidal rule (numpy.trapezoid) directly on the
model's native 100-point grid -- no additional smoothing or resampling.
Units are [model output intensity] x nm; no independent physical
calibration is applied. NEGATIVE BAND AREA is a signed (non-positive)
quantity in the saved results; plots show its MAGNITUDE (|value|) for
readability, and every axis/label says so explicitly.

Candidate generation per substitution level k (1 to 16):
  - k = 1, k = 2, k = 3 (EXHAUSTIVE_MAX_LEVEL): EXHAUSTIVE enumeration of
    every C2-symmetry-deduplicated molecule (see screen_substituents.py).
    Their statistics below are therefore EXACT over the full population
    at that level.
  - k above EXHAUSTIVE_MAX_LEVEL: a RANDOM SAMPLE of n_per_level unique
    (symmetry-deduplicated) molecules. Their statistics are estimates with
    finite-sample uncertainty that shrinks as n_per_level grows -- this
    script is meant to be run first with a small n_per_level (e.g. 100)
    to validate the pipeline, then re-run at full scale (e.g. 10,000,000).

At every level we report, across whatever set (exhaustive or sampled) was
evaluated there:
  - the MEAN area (positive; |negative|)
  - the MEAN area of the TOP 10% molecules at that level, ranked by that
    same area (this "top decile" line answers "how good can it get",
    the plain mean answers "how good is it on average")

Usage
-----
    python band_area_scaling_study.py \\
        --library substituent_library.json \\
        --model-dir E2E_64_package --label E2E_64 \\
        --output-dir band_area_study \\
        --n-per-level 100          # start small, then re-run with 1000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from predict_ecd import load_model_package
from screen_substituents import (
    N_POSITIONS, build_descriptor_matrix, build_vocabulary, is_canonical,
    iter_assignments, load_library, mirror_assignment,
)

TOP_FRACTION = 0.01
EXHAUSTIVE_MAX_LEVEL = 3  # k=1, k=2, k=3 are always exhaustive; k>=4 are sampled

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 16, 22, 20, 18


# --------------------------------------------------------------------------- #
# Candidate generation                                                         #
# --------------------------------------------------------------------------- #

def random_candidate(vocabulary: list, k: int, rng: np.random.Generator) -> tuple:
    positions = tuple(sorted(rng.choice(np.arange(1, N_POSITIONS + 1), size=k, replace=False).tolist()))
    substituents = tuple(rng.choice(vocabulary, size=k, replace=True).tolist())
    return positions, substituents


def generate_level(level: int, vocabulary: list, n_per_level: int, seed: int) -> tuple:
    """Returns (candidates, is_exhaustive)."""
    if level <= EXHAUSTIVE_MAX_LEVEL:
        candidates = list(iter_assignments(vocabulary, level, dedupe_symmetry=True))
        return candidates, True

    rng = np.random.default_rng(seed + level)  # different, reproducible stream per level
    seen, out = set(), []
    max_attempts = n_per_level * 30  # safety valve if the space is smaller than requested
    attempts = 0
    while len(out) < n_per_level and attempts < max_attempts:
        cand = random_candidate(vocabulary, level, rng)
        canonical = cand if is_canonical(*cand) else mirror_assignment(*cand)
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
        attempts += 1
    if len(out) < n_per_level:
        print(f"  [level {level}] only found {len(out)} unique molecules in {attempts} attempts "
              f"(requested {n_per_level}) -- the space at this level may be smaller than requested, "
              "or duplicates are being hit often; treat this level as exhaustive-ish.")
    return out, False


# --------------------------------------------------------------------------- #
# Batch evaluation (chunked so this stays memory-safe from 100 to 1,000,000)  #
# --------------------------------------------------------------------------- #

def find_largest_band_areas(Y: np.ndarray, wl: np.ndarray) -> tuple:
    """For each row of Y (n, 100), the area of the single largest
    contiguous positive-valued run and the single largest contiguous
    negative-valued run (magnitude), bounded by sign changes on the grid.

    Fully vectorized (no per-row Python loop, so this stays cheap even at
    n=1,000,000): on a uniform grid, the trapezoidal area of a band with a
    zero-crossing flank on each side equals dx * sum(interior values) --
    the flanking zero contributes nothing regardless of its trapezoid
    weight. A band that instead touches the true domain edge (250 or
    600 nm, no flanking zero available) gets that edge point HALF weight,
    exactly matching plain np.trapezoid()'s own convention -- this is what
    makes a single-band spectrum's "largest band area" come out identical
    to its total positive/negative integral, verified in testing."""
    dx = wl[1] - wl[0]
    n, m = Y.shape

    def per_sign(arr: np.ndarray) -> np.ndarray:
        is_on = arr > 0
        prev_on = np.zeros_like(is_on)
        prev_on[:, 1:] = is_on[:, :-1]
        run_start = is_on & ~prev_on
        group_id = np.cumsum(run_start, axis=1)
        max_groups = int(group_id.max()) + 1 if group_id.size else 1
        row_offset = np.arange(n)[:, None] * (max_groups + 1)
        flat_key = (row_offset + group_id).ravel()
        sums_flat = np.bincount(flat_key, weights=arr.ravel(),
                                 minlength=n * (max_groups + 1)).astype(np.float64)

        left_key = row_offset[:, 0] + group_id[:, 0]
        np.subtract.at(sums_flat, left_key, 0.5 * arr[:, 0])
        right_key = row_offset[:, 0] + group_id[:, -1]
        np.subtract.at(sums_flat, right_key, 0.5 * arr[:, -1])

        sums = sums_flat.reshape(n, max_groups + 1) * dx
        return sums.max(axis=1)

    positive_band = per_sign(np.clip(Y, 0, None))
    negative_band_abs = per_sign(np.clip(-Y, 0, None))
    return positive_band, negative_band_abs


def evaluate_level(candidates: list, library: dict, package, chunk_size: int = 20000,
                    check_ad: bool = True, batch_size: int = 512) -> pd.DataFrame:
    wl = package.wavelengths
    ad_nn = None
    ad_threshold = None
    if check_ad:
        ad = package.applicability_domain
        ad_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ad["training_scaled_X"])
        ad_threshold = float(ad["threshold"])

    n = len(candidates)
    positive_area = np.empty(n, dtype=np.float64)
    negative_area = np.empty(n, dtype=np.float64)
    largest_positive_band = np.empty(n, dtype=np.float64)
    largest_negative_band_abs = np.empty(n, dtype=np.float64)
    outside_ad = np.zeros(n, dtype=bool)

    for start in range(0, n, chunk_size):
        chunk = candidates[start:start + chunk_size]
        x = build_descriptor_matrix(chunk, library)
        x_scaled = package.scaler_x.transform(x)
        y_scaled = package.model.predict(x_scaled, batch_size=batch_size, verbose=0)
        y = package.scaler_y.inverse_transform(y_scaled)

        positive_area[start:start + len(chunk)] = np.trapezoid(np.clip(y, 0, None), wl, axis=1)
        negative_area[start:start + len(chunk)] = np.trapezoid(np.clip(y, None, 0), wl, axis=1)
        lp, ln = find_largest_band_areas(y, wl)
        largest_positive_band[start:start + len(chunk)] = lp
        largest_negative_band_abs[start:start + len(chunk)] = ln

        if check_ad:
            dist, _ = ad_nn.kneighbors(x_scaled)
            outside_ad[start:start + len(chunk)] = dist[:, 0] > ad_threshold

    return pd.DataFrame({
        "positive_area": positive_area,
        "negative_area": negative_area,           # signed, <= 0
        "largest_positive_band_area": largest_positive_band,
        "largest_negative_band_area_abs": largest_negative_band_abs,
        "negative_area_abs": np.abs(negative_area),
        "outside_applicability_domain": outside_ad,
    })


def format_candidate_readable(candidate: tuple, library: dict) -> str:
    positions, substituents = candidate
    names = library.get("names", {})
    return ", ".join(f"Pos{p}={names.get(s, s)}" for p, s in zip(positions, substituents))


def extract_top10(level: int, candidates: list, per_molecule: pd.DataFrame, area_col: str,
                   library: dict, band: str) -> pd.DataFrame:
    """Top 10 candidates at this level by area_col (descending). Only the
    10 winners are looked up/formatted -- cheap even when per_molecule has
    1,000,000 rows, since we never format all of them, just sort + slice."""
    values = per_molecule[area_col].to_numpy()
    top_idx = np.argsort(values)[::-1][:10]

    records = []
    for rank, idx in enumerate(top_idx, start=1):
        cand = candidates[idx]
        records.append({
            "level": level,
            "band": band,
            "rank": rank,
            "area": float(values[idx]),
            "positions": ";".join(str(p) for p in cand[0]),
            "substituents": ";".join(cand[1]),
            "readable": format_candidate_readable(cand, library),
            "outside_applicability_domain": bool(per_molecule.iloc[idx]["outside_applicability_domain"]),
        })
    return pd.DataFrame(records)


def summarize_level(level: int, per_molecule: pd.DataFrame, is_exhaustive: bool,
                     top_fraction: float = TOP_FRACTION) -> dict:
    n = len(per_molecule)
    n_top = max(1, int(np.ceil(n * top_fraction)))

    def mean_std_and_top(col):
        vals = per_molecule[col].to_numpy()
        top_vals = np.sort(vals)[::-1][:n_top]
        # ddof=1 (sample std) when there are >1 values, falls back to 0 for a single one
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        top_std = float(top_vals.std(ddof=1)) if len(top_vals) > 1 else 0.0
        return float(vals.mean()), std, float(top_vals.mean()), top_std

    pos_mean, pos_std, pos_top, pos_top_std = mean_std_and_top("positive_area")
    neg_mean, neg_std, neg_top, neg_top_std = mean_std_and_top("negative_area_abs")
    lpb_mean, lpb_std, lpb_top, lpb_top_std = mean_std_and_top("largest_positive_band_area")
    lnb_mean, lnb_std, lnb_top, lnb_top_std = mean_std_and_top("largest_negative_band_area_abs")

    return {
        "level": level,
        "n_evaluated": n,
        "exhaustive": is_exhaustive,
        "pct_outside_AD": float(100 * per_molecule["outside_applicability_domain"].mean()),
        "positive_area_mean": pos_mean,
        "positive_area_std": pos_std,
        "positive_area_top1pct_mean": pos_top,
        "positive_area_top1pct_std": pos_top_std,
        "negative_area_abs_mean": neg_mean,
        "negative_area_abs_std": neg_std,
        "negative_area_abs_top1pct_mean": neg_top,
        "negative_area_abs_top1pct_std": neg_top_std,
        "largest_positive_band_area_mean": lpb_mean,
        "largest_positive_band_area_std": lpb_std,
        "largest_positive_band_area_top1pct_mean": lpb_top,
        "largest_positive_band_area_top1pct_std": lpb_top_std,
        "largest_negative_band_area_abs_mean": lnb_mean,
        "largest_negative_band_area_abs_std": lnb_std,
        "largest_negative_band_area_abs_top1pct_mean": lnb_top,
        "largest_negative_band_area_abs_top1pct_std": lnb_top_std,
    }


# --------------------------------------------------------------------------- #
# Plots (English, per spec)                                                   #
# --------------------------------------------------------------------------- #

def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def plot_positive(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.errorbar(summary["level"], summary["positive_area_mean"], yerr=summary["positive_area_std"],
                marker="o", linewidth=2.5, capsize=5, elinewidth=1.5, color="#2E7D32",
                ecolor="#2E7D32", alpha=0.95, label="Mean (all evaluated) \u00b1 std")
    ax.errorbar(summary["level"], summary["positive_area_top1pct_mean"],
            yerr=summary["positive_area_top1pct_std"], marker="o", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#1B5E20", ecolor="#1B5E20",
            alpha=0.95, label="Mean of top 1% \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Positive ECD band area vs. substitution level",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS)
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_positive_band_area_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_negative(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.errorbar(summary["level"], summary["negative_area_abs_mean"], yerr=summary["negative_area_abs_std"],
                marker="o", linewidth=2.5, capsize=5, elinewidth=1.5, color="#C0272D",
                ecolor="#C0272D", alpha=0.95, label="Mean (all evaluated) \u00b1 std")
    ax.errorbar(summary["level"], summary["negative_area_abs_top1pct_mean"],
            yerr=summary["negative_area_abs_top1pct_std"], marker="o", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#7B0000", ecolor="#7B0000",
            alpha=0.95, label="Mean of top 1% \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Negative ECD band area vs. substitution level",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS)
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_negative_band_area_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.errorbar(summary["level"], summary["positive_area_mean"], yerr=summary["positive_area_std"],
                marker="o", linewidth=2.5, capsize=5, elinewidth=1.5,
                color="#2E7D32", ecolor="#2E7D32", alpha=0.95, label="Positive — mean \u00b1 std")
    ax.errorbar(summary["level"], summary["positive_area_top1pct_mean"],
            yerr=summary["positive_area_top1pct_std"], marker="o", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#1B5E20", ecolor="#1B5E20",
            alpha=0.95, label="Positive — top 1% mean \u00b1 std")
    ax.errorbar(summary["level"], summary["negative_area_abs_mean"], yerr=summary["negative_area_abs_std"],
                marker="s", linewidth=2.5, capsize=5, elinewidth=1.5,
                color="#C0272D", ecolor="#C0272D", alpha=0.95, label="Negative (|area|) — mean \u00b1 std")
    ax.errorbar(summary["level"], summary["negative_area_abs_top1pct_mean"],
            yerr=summary["negative_area_abs_top1pct_std"], marker="s", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#7B0000", ecolor="#7B0000",
            alpha=0.95, label="Negative (|area|) — top 1% mean \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Positive vs. negative ECD band area progression",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS, loc="best")
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_combined_band_area_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined_means_only(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.errorbar(summary["level"], summary["positive_area_mean"], yerr=summary["positive_area_std"],
                marker="o", linewidth=2.5, capsize=5, elinewidth=1.5,
                color="#2E7D32", ecolor="#2E7D32", alpha=0.95, label="Positive — mean \u00b1 std")
    ax.errorbar(summary["level"], summary["negative_area_abs_mean"], yerr=summary["negative_area_abs_std"],
                marker="s", linewidth=2.5, capsize=5, elinewidth=1.5,
                color="#C0272D", ecolor="#C0272D", alpha=0.95, label="Negative (|area|) — mean \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Positive vs. negative ECD band area progression (mean only)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS, loc="best")
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_combined_band_area_progression_means_only.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


# --- Parallel study: the SINGLE LARGEST contiguous band, not the total ---

def plot_largest_positive_band(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.errorbar(summary["level"], summary["largest_positive_band_area_mean"],
                yerr=summary["largest_positive_band_area_std"], marker="o", linewidth=2.5,
                capsize=5, elinewidth=1.5, color="#2E7D32", ecolor="#2E7D32", alpha=0.95,
                label="Mean (all evaluated) \u00b1 std")
    ax.errorbar(summary["level"], summary["largest_positive_band_area_top1pct_mean"],
            yerr=summary["largest_positive_band_area_top1pct_std"], marker="o", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#1B5E20", ecolor="#1B5E20",
            alpha=0.95, label="Mean of top 1% \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Largest single positive ECD band vs. substitution level",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS)
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_largest_positive_band_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_largest_negative_band(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.errorbar(summary["level"], summary["largest_negative_band_area_abs_mean"],
                yerr=summary["largest_negative_band_area_abs_std"], marker="o", linewidth=2.5,
                capsize=5, elinewidth=1.5, color="#C0272D", ecolor="#C0272D", alpha=0.95,
                label="Mean (all evaluated) \u00b1 std")
    ax.errorbar(summary["level"], summary["largest_negative_band_area_abs_top1pct_mean"],
            yerr=summary["largest_negative_band_area_abs_top1pct_std"], marker="o", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#7B0000", ecolor="#7B0000",
            alpha=0.95, label="Mean of top 1% \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Largest single negative ECD band vs. substitution level",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS)
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_largest_negative_band_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined_largest_band(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.errorbar(summary["level"], summary["largest_positive_band_area_mean"],
                yerr=summary["largest_positive_band_area_std"], marker="o", linewidth=2.5,
                capsize=5, elinewidth=1.5, color="#2E7D32", ecolor="#2E7D32", alpha=0.95,
                label="Positive band — mean \u00b1 std")
    ax.errorbar(summary["level"], summary["largest_positive_band_area_top1pct_mean"],
            yerr=summary["largest_positive_band_area_top1pct_std"], marker="o", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#1B5E20", ecolor="#1B5E20",
            alpha=0.95, label="Positive band — top 1% mean \u00b1 std")
    ax.errorbar(summary["level"], summary["largest_negative_band_area_abs_mean"],
                yerr=summary["largest_negative_band_area_abs_std"], marker="s", linewidth=2.5,
                capsize=5, elinewidth=1.5, color="#C0272D", ecolor="#C0272D", alpha=0.95,
                label="Negative band — mean \u00b1 std")
    ax.errorbar(summary["level"], summary["largest_negative_band_area_abs_top1pct_mean"],
            yerr=summary["largest_negative_band_area_abs_top1pct_std"], marker="s", linewidth=2.5,
            linestyle="--", capsize=5, elinewidth=1.5, color="#7B0000", ecolor="#7B0000",
            alpha=0.95, label="Negative band — top 1% mean \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Largest single positive vs. negative ECD band progression",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS, loc="best")
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_combined_largest_band_progression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined_largest_band_means_only(summary: pd.DataFrame, output_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.errorbar(summary["level"], summary["largest_positive_band_area_mean"],
                yerr=summary["largest_positive_band_area_std"], marker="o", linewidth=2.5,
                capsize=5, elinewidth=1.5, color="#2E7D32", ecolor="#2E7D32", alpha=0.95,
                label="Largest positive band — mean \u00b1 std")
    ax.errorbar(summary["level"], summary["largest_negative_band_area_abs_mean"],
                yerr=summary["largest_negative_band_area_abs_std"], marker="s", linewidth=2.5,
                capsize=5, elinewidth=1.5, color="#C0272D", ecolor="#C0272D", alpha=0.95,
                label="Largest negative band (|area|) — mean \u00b1 std")
    ax.set_xlabel("Number of substituents", fontsize=LABEL_FS)
    ax.set_ylabel("Integral", fontsize=LABEL_FS)
    ax.set_title(f"{label} — Largest single positive vs. negative ECD band progression (mean only)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS, loc="best")
    ax.set_xticks(summary["level"])
    _style_axis(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{label}_combined_largest_band_progression_means_only.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

METHODOLOGY_TEXT = """BAND AREA SCALING STUDY -- METHODOLOGY

TOTAL band area
  Positive band area   = integral of max(R(lambda), 0) d(lambda), lambda in [250, 600] nm,
                          trapezoidal rule on the model's native 100-point grid.
  Negative band area    = integral of min(R(lambda), 0) d(lambda), same grid/range/rule.
                          Signed (<= 0) in the saved CSV; plots show its magnitude |.|.
  This sums ALL positive (or negative) regions of the spectrum together, however many
  separate lobes there are.

LARGEST SINGLE band area
  For each spectrum, its positive (negative) regions are split into contiguous bands,
  each bounded by a zero-crossing (or the edge of the 250-600 nm window). Each band's
  area is computed with the same trapezoidal rule as above. The LARGEST such band's
  area is reported -- i.e. the single strongest, best-defined band, as opposed to the
  sum of every small lobe. For a spectrum with only one band of a given sign, this is
  identical to that sign's TOTAL band area by construction (verified in testing).

Units = [model output intensity] x nm; no independent physical calibration.

Candidate generation per substitution level k:
  k = 1, k = 2, k = 3  -> EXHAUSTIVE enumeration, C2-symmetry-deduplicated (exact statistics).
  k = 4 .. 16          -> a RANDOM SAMPLE of n_per_level unique, symmetry-deduplicated molecules
                          (estimated statistics; finite-sample uncertainty shrinks as n_per_level grows).

Reported per level, for each of the four area definitions above: the mean across every
molecule evaluated, and the mean of the top 1% (by that same area) at that level. The
top 10 individual best molecules (by area) at each level are also listed and saved.
"""


def replot_from_summary(summary_csv_path: str, output_dir: str, label: str) -> pd.DataFrame:
    """Regenerates all 8 progression plots from an already-saved
    {label}_band_area_summary.csv -- no re-prediction, no re-running
    run_study(). Use this to restyle, re-export, or just re-view the
    plots after the (potentially long) evaluation step has already run
    once. Does NOT regenerate the top-10 listings or per-molecule detail
    CSVs, since those aren't stored in the summary -- only the 8 PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    summary_df = pd.read_csv(summary_csv_path)

    std_cols = ["positive_area_std", "negative_area_abs_std",
                "largest_positive_band_area_std", "largest_negative_band_area_abs_std",
                "positive_area_top1pct_std", "negative_area_abs_top1pct_std",
                "largest_positive_band_area_top1pct_std", "largest_negative_band_area_abs_top1pct_std"]
    missing = [c for c in std_cols if c not in summary_df.columns]
    if missing:
        print(f"This summary CSV predates the error-bar update (missing: {missing}) -- "
              "filling with 0 so the plots still render, just without error bars for those lines. "
              "Re-run run_study() to get real std values.")
        for c in missing:
            summary_df[c] = 0.0

    plot_positive(summary_df, output_dir, label)
    plot_negative(summary_df, output_dir, label)
    plot_combined(summary_df, output_dir, label)
    plot_combined_means_only(summary_df, output_dir, label)
    plot_largest_positive_band(summary_df, output_dir, label)
    plot_largest_negative_band(summary_df, output_dir, label)
    plot_combined_largest_band(summary_df, output_dir, label)
    plot_combined_largest_band_means_only(summary_df, output_dir, label)

    return summary_df


def run_study(library_path: str, model_dir: str, label: str, output_dir: str,
              n_per_level: int = 100, levels: Optional[list] = None, seed: int = 42,
              chunk_size: int = 20000, batch_size: int = 512, check_ad: bool = True,
              save_full_detail_max_rows: int = 5000) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "METHODOLOGY.txt"), "w") as f:
        f.write(METHODOLOGY_TEXT)
    print(METHODOLOGY_TEXT)

    library = load_library(library_path)
    vocabulary = build_vocabulary(library)
    package = load_model_package(model_dir, label)

    levels = levels or list(range(1, N_POSITIONS + 1))
    summaries = []
    top10_records = []
    t0 = time.time()

    for level in levels:
        t_level = time.time()
        candidates, is_exhaustive = generate_level(level, vocabulary, n_per_level, seed)
        per_molecule = evaluate_level(candidates, library, package, chunk_size, check_ad, batch_size)
        summary = summarize_level(level, per_molecule, is_exhaustive)
        summaries.append(summary)

        tag = "EXHAUSTIVE" if is_exhaustive else f"SAMPLE(n={n_per_level})"
        print(f"level {level:2d} [{tag:>16}]: n={summary['n_evaluated']:>9,}  "
              f"pos_mean={summary['positive_area_mean']:.2f}  pos_top1%={summary['positive_area_top1pct_mean']:.2f}  "
              f"|neg|_mean={summary['negative_area_abs_mean']:.2f}  |neg|_top1%={summary['negative_area_abs_top1pct_mean']:.2f}  "
              f"({time.time() - t_level:.1f}s)")

        top10_pos = extract_top10(level, candidates, per_molecule, "positive_area", library, "positive_total")
        top10_neg = extract_top10(level, candidates, per_molecule, "negative_area_abs", library, "negative_total")
        top10_lpb = extract_top10(level, candidates, per_molecule, "largest_positive_band_area", library, "positive_largest_band")
        top10_lnb = extract_top10(level, candidates, per_molecule, "largest_negative_band_area_abs", library, "negative_largest_band")
        top10_records += [top10_pos, top10_neg, top10_lpb, top10_lnb]

        print(f"  Top 10 by TOTAL POSITIVE area at level {level}:")
        for _, r in top10_pos.iterrows():
            ad_flag = "  \u26a0 outside AD" if r["outside_applicability_domain"] else ""
            print(f"    {r['rank']:>2}. area={r['area']:>12.2f}  {r['readable']}{ad_flag}")
        print(f"  Top 10 by TOTAL NEGATIVE (|area|) at level {level}:")
        for _, r in top10_neg.iterrows():
            ad_flag = "  \u26a0 outside AD" if r["outside_applicability_domain"] else ""
            print(f"    {r['rank']:>2}. |area|={r['area']:>12.2f}  {r['readable']}{ad_flag}")
        print(f"  Top 10 by LARGEST SINGLE POSITIVE band at level {level}:")
        for _, r in top10_lpb.iterrows():
            ad_flag = "  \u26a0 outside AD" if r["outside_applicability_domain"] else ""
            print(f"    {r['rank']:>2}. area={r['area']:>12.2f}  {r['readable']}{ad_flag}")
        print(f"  Top 10 by LARGEST SINGLE NEGATIVE band (|area|) at level {level}:")
        for _, r in top10_lnb.iterrows():
            ad_flag = "  \u26a0 outside AD" if r["outside_applicability_domain"] else ""
            print(f"    {r['rank']:>2}. |area|={r['area']:>12.2f}  {r['readable']}{ad_flag}")

        if len(per_molecule) <= save_full_detail_max_rows:
            detail_path = os.path.join(output_dir, f"{label}_level{level:02d}_full_detail.csv")
            meta = pd.DataFrame([{"positions": ";".join(map(str, p)), "substituents": ";".join(s)}
                                  for p, s in candidates])
            pd.concat([meta, per_molecule], axis=1).to_csv(detail_path, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(output_dir, f"{label}_band_area_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    top10_df = pd.concat(top10_records, ignore_index=True)
    top10_path = os.path.join(output_dir, f"{label}_top10_per_level.csv")
    top10_df.to_csv(top10_path, index=False)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print(f"Saved: {summary_path}")
    print(f"Saved: {top10_path}")
    print("\nFull summary table:")
    print(summary_df.round(3).to_string(index=False))

    plot_positive(summary_df, output_dir, label)
    plot_negative(summary_df, output_dir, label)
    plot_combined(summary_df, output_dir, label)
    plot_combined_means_only(summary_df, output_dir, label)
    plot_largest_positive_band(summary_df, output_dir, label)
    plot_largest_negative_band(summary_df, output_dir, label)
    plot_combined_largest_band(summary_df, output_dir, label)
    plot_combined_largest_band_means_only(summary_df, output_dir, label)

    return summary_df, top10_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Band area scaling study across substitution levels 1-16.")
    p.add_argument("--library", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", default="E2E_64")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-per-level", type=int, default=100,
                    help="Random-sample size for k=3..16. Start small (100), then scale up to 1000000.")
    p.add_argument("--levels", type=int, nargs="+", default=None, help="Subset of levels to run, e.g. 1 2 3.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=20000,
                    help="How many candidate MOLECULES are generated, scored and held in memory at "
                         "once per level, before moving to the next chunk. This is a Python-side/RAM "
                         "control, not a GPU control -- lower it only if you run out of memory, not "
                         "for speed (a tiny value like 10 makes this SLOWER, since almost every "
                         "molecule gets its own separate loop iteration).")
    p.add_argument("--batch-size", type=int, default=512,
                    help="How many molecules go into a single model.predict() call on the GPU/CPU. "
                         "This IS the one that affects raw prediction speed -- larger (1024, 4096) is "
                         "usually faster on a GPU with enough memory, smaller if you hit an "
                         "out-of-memory error from Keras/TensorFlow itself.")
    p.add_argument("--no-ad-check", action="store_true", help="Skip the applicability-domain check (faster).")
    args, _unknown = p.parse_known_args()  # tolerate Colab/Jupyter's injected argv
    return args


if __name__ == "__main__":
    args = parse_args()
    run_study(args.library, args.model_dir, args.label, args.output_dir,
               n_per_level=args.n_per_level, levels=args.levels, seed=args.seed,
               chunk_size=args.chunk_size, batch_size=args.batch_size, check_ad=not args.no_ad_check)
