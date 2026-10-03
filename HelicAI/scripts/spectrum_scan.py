#!/usr/bin/env python3
"""
spectrum_scan.py — Predict and save the COMPLETE 100-point ECD spectrum
(not just a summary number) for every candidate molecule at one or more
substitution levels -- e.g. "give me the full predicted spectrum of every
trisubstituted molecule" (level=3).

This is the sibling of band_area_scaling_study.py: same candidate
generation (exhaustive for k <= EXHAUSTIVE_MAX_LEVEL, random sample above
that), same band-detection code, so the numbers match exactly if you
cross-check -- the difference is this script KEEPS every predicted
spectrum instead of discarding it after computing summary statistics.

You get two files per level, always both:
  1. A CSV with one row per candidate: its positions/substituents, and
     four ready-to-use numbers per molecule (not just the top 10 or an
     aggregate mean like band_area_scaling_study.py) --
       - positive_area / negative_area   (TOTAL area of that sign)
       - largest_positive_band / largest_negative_band  (single biggest
         contiguous band of that sign -- see band_area_scaling_study.py's
         own docstring for exactly how this is defined)
     Use this file to filter, sort, or plot a histogram over every single
     candidate at that level -- e.g. "show me the distribution of largest
     positive band across all 1,146,880 trisubstituted molecules".
  2. A compressed .npz with the actual (N, 100) array of predicted spectra,
     row-aligned with the CSV (row i of the CSV <-> spectra[i] in the
     .npz), plus the wavelength grid. Use load_spectrum() or
     load_spectra_matching() below to pull out the ones you want after
     filtering the CSV.

A NOTE ON SIZE, so you can set n_per_level yourself with your eyes open:
each spectrum is 100 float32 numbers = 400 bytes. At level 3 (exhaustive,
1,146,880 molecules) that's ~440 MB for the .npz -- fine. At level 6 with
n_per_level=10,000,000 it would be ~4 GB for THAT LEVEL ALONE. This script
prints the exact estimated size before running each level so you can
Ctrl-C if it's more than you want.

Usage
-----
    python spectrum_scan.py \\
        --library substituent_library.json \\
        --model-dir E2E_64_package --label E2E_64 \\
        --output-dir spectrum_scan_results \\
        --levels 3 \\
        --n-per-level 100000 \\
        --chunk-size 20000 --batch-size 512

As a library, to get every trisubstituted molecule's full spectrum:
    import spectrum_scan as ss
    df, spectra, wl = ss.scan_level(
        level=3, library_path="substituent_library.json",
        model_dir="E2E_64_package", label="E2E_64",
    )
    # df.iloc[i] describes the molecule whose spectrum is spectra[i]
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from band_area_scaling_study import (
    EXHAUSTIVE_MAX_LEVEL, find_largest_band_areas, generate_level,
)
from predict_ecd import load_model_package
from screen_substituents import build_descriptor_matrix, build_vocabulary, load_library

BYTES_PER_SPECTRUM = 100 * 4  # float32


def format_candidate_readable(candidate: tuple, library: dict) -> str:
    positions, substituents = candidate
    names = library.get("names", {})
    return ", ".join(f"Pos{p}={names.get(s, s)}" for p, s in zip(positions, substituents))


# --------------------------------------------------------------------------- #
# Core scan                                                                    #
# --------------------------------------------------------------------------- #

def scan_level(level: int, library_path: str, model_dir: str, label: str,
               n_per_level: int = 100, seed: int = 42, chunk_size: int = 20000,
               batch_size: int = 512) -> tuple:
    """Returns (dataframe, spectra, wavelengths). dataframe.iloc[i]
    describes the molecule whose spectrum is spectra[i]."""
    library = load_library(library_path)
    vocabulary = build_vocabulary(library)
    package = load_model_package(model_dir, label)
    wl = package.wavelengths

    candidates, is_exhaustive = generate_level(level, vocabulary, n_per_level, seed)
    n = len(candidates)
    size_mb = n * BYTES_PER_SPECTRUM / 1e6
    tag = "EXHAUSTIVE" if is_exhaustive else f"SAMPLE(n={n_per_level})"
    print(f"level {level} [{tag}]: {n:,} molecules -> spectra array will be ~{size_mb:,.1f} MB")

    spectra = np.empty((n, 100), dtype=np.float32)
    positive_area = np.empty(n, dtype=np.float64)
    negative_area = np.empty(n, dtype=np.float64)
    largest_positive_band = np.empty(n, dtype=np.float64)
    largest_negative_band_abs = np.empty(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        chunk = candidates[start:start + chunk_size]
        x = build_descriptor_matrix(chunk, library)
        x_scaled = package.scaler_x.transform(x)
        y_scaled = package.model.predict(x_scaled, batch_size=batch_size, verbose=0)
        y = package.scaler_y.inverse_transform(y_scaled)

        end = start + len(chunk)
        spectra[start:end] = y
        positive_area[start:end] = np.trapezoid(np.clip(y, 0, None), wl, axis=1)
        negative_area[start:end] = np.trapezoid(np.clip(y, None, 0), wl, axis=1)
        lp, ln = find_largest_band_areas(y, wl)
        largest_positive_band[start:end] = lp
        largest_negative_band_abs[start:end] = ln
        print(f"  {end:,}/{n:,} predicted")

    df = pd.DataFrame({
        "positions": [";".join(str(p) for p in c[0]) for c in candidates],
        "substituents": [";".join(c[1]) for c in candidates],
        "readable": [format_candidate_readable(c, library) for c in candidates],
        "positive_area": positive_area,
        "negative_area": negative_area,
        "largest_positive_band": largest_positive_band,
        "largest_negative_band_abs": largest_negative_band_abs,
    })
    return df, spectra, wl


def save_level(level: int, df: pd.DataFrame, spectra: np.ndarray, wl: np.ndarray,
               output_dir: str, label: str) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{label}_level{level:02d}_summary.csv")
    npz_path = os.path.join(output_dir, f"{label}_level{level:02d}_spectra.npz")

    df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, spectra=spectra, wavelengths=wl)

    print(f"Saved: {csv_path}  ({len(df):,} rows)")
    print(f"Saved: {npz_path}  ({spectra.nbytes / 1e6:,.1f} MB uncompressed, less on disk once compressed)")
    return csv_path, npz_path


def run_scan(library_path: str, model_dir: str, label: str, output_dir: str, levels: list,
            n_per_level: int = 100, seed: int = 42, chunk_size: int = 20000,
            batch_size: int = 512) -> dict:
    results = {}
    for level in levels:
        print(f"\n{'=' * 90}\nLEVEL {level}\n{'=' * 90}")
        df, spectra, wl = scan_level(level, library_path, model_dir, label, n_per_level,
                                     seed, chunk_size, batch_size)
        csv_path, npz_path = save_level(level, df, spectra, wl, output_dir, label)
        results[level] = {"csv": csv_path, "npz": npz_path, "n": len(df)}
    return results


# --------------------------------------------------------------------------- #
# Convenience loaders, for AFTER you've filtered the CSV                      #
# --------------------------------------------------------------------------- #

def load_spectrum(npz_path: str, row_index: int) -> tuple:
    """The single spectrum at a given row -- the same row index as in the
    companion CSV. Returns (wavelengths, spectrum)."""
    with np.load(npz_path) as data:
        return data["wavelengths"], data["spectra"][row_index]


def load_spectra_matching(csv_path: str, npz_path: str, query: str) -> tuple:
    """Filter the CSV with a pandas query string (e.g. "largest_positive_band > 50000"
    or "readable.str.contains('NO2')", the latter needs engine='python' handled
    internally), then pull out exactly those rows' spectra from the .npz.
    Returns (filtered_dataframe, matching_spectra, wavelengths)."""
    df = pd.read_csv(csv_path)
    try:
        filtered = df.query(query)
    except Exception:
        filtered = df.query(query, engine="python")
    with np.load(npz_path) as data:
        wl = data["wavelengths"]
        spectra = data["spectra"][filtered.index.to_numpy()]
    return filtered.reset_index(drop=True), spectra, wl


DEFAULT_LIBRARY = "/content/substituent_library.json"
DEFAULT_MODEL_DIR = "/content/E2E_64_package"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/ecd_project/spectrum_scan_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict and save the full ECD spectrum of every "
                                             "candidate molecule at one or more substitution levels.")
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--label", default="E2E_64")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--levels", type=int, nargs="+", required=True, help="e.g. --levels 3, for all "
                    "trisubstituted molecules.")
    p.add_argument("--n-per-level", type=int, default=100,
                    help="Random-sample size for levels above EXHAUSTIVE_MAX_LEVEL "
                         f"(currently {EXHAUSTIVE_MAX_LEVEL}). Ignored -- exhaustive is used instead "
                         "-- for levels at or below that.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=20000,
                    help="Molecules generated/held in memory at once per level (RAM control).")
    p.add_argument("--batch-size", type=int, default=512,
                    help="Molecules per model.predict() call (GPU/CPU speed control).")
    args, _unknown = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_scan(args.library, args.model_dir, args.label, args.output_dir, args.levels,
             n_per_level=args.n_per_level, seed=args.seed,
             chunk_size=args.chunk_size, batch_size=args.batch_size)
