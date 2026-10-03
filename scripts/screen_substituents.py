#!/usr/bin/env python3
"""
screen_substituents.py — Combinatorial ECD screening for [6]helicenes.

Given a substituent library (Hammett sigma -> {vdw, r_plus, r_minus}) and a
trained E2E-64 (or E2E-H) model package, this generates every heliceno with
exactly `k` substituted positions (out of 16, the rest left as H) and
predicts its ECD spectrum in a single batched pass — not a Python loop
calling model.predict() once per molecule, which would be far slower.

Requires predict_ecd.py in the same directory (reuses load_model_package).

Combinatorics
-------------
For k substituted positions and a vocabulary of n valid substituents:
    n_molecules = C(16, k) * n**k
Position identity matters (position 3 is not interchangeable with position
9), so (substituent A at position i, substituent B at position j) and
(B at i, A at j) are counted as two distinct molecules. Same substituent
repeated at multiple positions (e.g. a symmetric di-bromo heliceno) is
included automatically.

This grows fast: k=2, n=16 -> 30,720 molecules (seconds). k=3, n=16 ->
2,293,760 molecules (minutes, and ~900 MB for the raw spectra matrix in
float32). Use --dry-run first to see the count before committing.

Example
-------
    python screen_substituents.py \\
        --model-dir E2E_64_package \\
        --label E2E_64 \\
        --library substituent_library.json \\
        --n-substituents 2 \\
        --output-dir screening_results \\
        --sort-by max_abs_intensity \\
        --top-n 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from predict_ecd import load_model_package, ModelPackage

HAMMETT_COLS = [f"Pos_{i}" for i in range(1, 17)]
VDW_COLS = [f"VdW_{i}" for i in range(1, 17)]
RPLUS_COLS = [f"Rplus_{i}" for i in range(1, 17)]
RMINUS_COLS = [f"Rminus_{i}" for i in range(1, 17)]
ALL_64_COLS = HAMMETT_COLS + VDW_COLS + RPLUS_COLS + RMINUS_COLS
N_POSITIONS = 16


# --------------------------------------------------------------------------- #
# C2 symmetry deduplication (position i <-> 17-i, see train_pipeline.py's    #
# build_symmetry_mirror for the empirical verification: pairs related this   #
# way are the SAME molecule, median ECD cosine 1.00000 across 359 real       #
# examples). Roughly half of a naive combinatorial screen would otherwise be #
# literal duplicates re-labeled under different position numbers.            #
# --------------------------------------------------------------------------- #

def mirror_assignment(position_combo: tuple, substituent_combo: tuple) -> tuple:
    mirrored_positions = (N_POSITIONS + 1 - p for p in position_combo)
    paired = sorted(zip(mirrored_positions, substituent_combo))
    return tuple(p for p, _ in paired), tuple(s for _, s in paired)


def is_canonical(position_combo: tuple, substituent_combo: tuple) -> bool:
    """True if this assignment is its own mirror or the lexicographically
    smaller of the two -- keeping exactly one representative per physical
    molecule. (For k=1 this alone restricts positions to 1-8, matching
    exactly what was found in the real dataset's mono-substituted set.)"""
    mirrored = mirror_assignment(position_combo, substituent_combo)
    return (position_combo, substituent_combo) <= mirrored


# --------------------------------------------------------------------------- #
# Substituent library                                                         #
# --------------------------------------------------------------------------- #

def load_library(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_vocabulary(library: dict, include_partial: bool = False) -> list:
    """Return the list of substituent keys (Hammett sigma strings) usable for
    screening. By default, only keys present in vdw AND r_plus AND r_minus
    are kept, since a key present in only vdw cannot produce a full
    4-descriptor vector without guessing r_plus/r_minus."""
    vdw_keys = set(library["descriptors"]["vdw"])
    rplus_keys = set(library["descriptors"]["r_plus"])
    rminus_keys = set(library["descriptors"]["r_minus"])

    full_keys = sorted(vdw_keys & rplus_keys & rminus_keys, key=float)
    partial_keys = sorted(vdw_keys - (rplus_keys & rminus_keys), key=float)

    if partial_keys:
        print(f"NOTE: {len(partial_keys)} substituent(s) have VdW but no (complete) "
              f"R+/R- entry: {partial_keys}")
        if include_partial:
            print("  --include-partial is set: these will be used with r_plus=r_minus="
                  f"{library['defaults']['r_plus']} (the default). Verify this is chemically "
                  "correct before trusting predictions for these substituents.")
        else:
            print("  Excluded from the vocabulary. Re-run with --include-partial to force-include them.")

    vocabulary = full_keys + (partial_keys if include_partial else [])
    if not vocabulary:
        raise ValueError("No usable substituents found in the library.")
    return vocabulary


def substituent_vector(key: Optional[str], library: dict) -> tuple:
    """(hammett, vdw, r_plus, r_minus) for one substituent key, or the H
    default if key is None."""
    d = library["defaults"]
    if key is None:
        return d["hammett"], d["vdw"], d["r_plus"], d["r_minus"]
    desc = library["descriptors"]
    hammett = float(key)
    vdw = desc["vdw"][key]
    r_plus = desc["r_plus"].get(key, d["r_plus"])
    r_minus = desc["r_minus"].get(key, d["r_minus"])
    return hammett, vdw, r_plus, r_minus


# --------------------------------------------------------------------------- #
# Combination generation (chunked, so memory stays bounded for large k)       #
# --------------------------------------------------------------------------- #

def count_combinations(n_positions: int, k: int, vocab_size: int) -> int:
    from math import comb
    return comb(n_positions, k) * (vocab_size ** k)


def iter_assignments(vocabulary: list, k: int, positions: range = range(1, N_POSITIONS + 1),
                      dedupe_symmetry: bool = True):
    """Yield (position_tuple, substituent_tuple) for every molecule with
    exactly k substituted positions. If dedupe_symmetry, skip the ~half of
    combinations that are the same physical molecule as one already yielded
    under the helicene's own C2 symmetry (position i <-> 17-i)."""
    for position_combo in itertools.combinations(positions, k):
        for substituent_combo in itertools.product(vocabulary, repeat=k):
            if dedupe_symmetry and not is_canonical(position_combo, substituent_combo):
                continue
            yield position_combo, substituent_combo


def build_descriptor_matrix(assignments_chunk: list, library: dict) -> np.ndarray:
    """assignments_chunk: list of (position_tuple, substituent_tuple).
    Returns an (n, 64) matrix in ALL_64_COLS order."""
    n = len(assignments_chunk)
    hammett = np.full((n, N_POSITIONS), library["defaults"]["hammett"], dtype=np.float32)
    vdw = np.full((n, N_POSITIONS), library["defaults"]["vdw"], dtype=np.float32)
    r_plus = np.full((n, N_POSITIONS), library["defaults"]["r_plus"], dtype=np.float32)
    r_minus = np.full((n, N_POSITIONS), library["defaults"]["r_minus"], dtype=np.float32)

    for row, (position_combo, substituent_combo) in enumerate(assignments_chunk):
        for position, key in zip(position_combo, substituent_combo):
            h, v, rp, rm = substituent_vector(key, library)
            col = position - 1
            hammett[row, col] = h
            vdw[row, col] = v
            r_plus[row, col] = rp
            r_minus[row, col] = rm

    return np.hstack([hammett, vdw, r_plus, r_minus])


# --------------------------------------------------------------------------- #
# Spectrum summary statistics (generic, chemistry-agnostic)                   #
# --------------------------------------------------------------------------- #

def summarize_spectra(spectra: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
    max_idx = np.argmax(np.abs(spectra), axis=1)
    max_abs_intensity = np.abs(spectra)[np.arange(len(spectra)), max_idx]
    wl_at_max = wavelengths[max_idx]
    integrated_abs_area = np.trapezoid(np.abs(spectra), wavelengths, axis=1)
    sign_changes = np.sum(np.diff(np.sign(spectra), axis=1) != 0, axis=1)
    return pd.DataFrame({
        "max_abs_intensity": max_abs_intensity,
        "wavelength_at_max_nm": wl_at_max,
        "integrated_abs_area": integrated_abs_area,
        "n_sign_changes": sign_changes,
    })


# --------------------------------------------------------------------------- #
# Main screening loop                                                         #
# --------------------------------------------------------------------------- #

def screen(model_dir: str, label: str, library_path: str, n_substituents: int,
           output_dir: str, include_partial: bool = False, chunk_size: int = 5000,
           limit: Optional[int] = None, dry_run: bool = False, dedupe_symmetry: bool = True) -> Optional[pd.DataFrame]:

    library = load_library(library_path)
    vocabulary = build_vocabulary(library, include_partial=include_partial)
    n_naive = count_combinations(N_POSITIONS, n_substituents, len(vocabulary))

    print(f"\nVocabulary size: {len(vocabulary)} substituents")
    print(f"Positions: {N_POSITIONS}, substituted per molecule: {n_substituents}")
    if dedupe_symmetry:
        print(f"Naive combinatorial count: {n_naive:,} — with C2-symmetry deduplication "
              f"(position i <-> 17-i), expect roughly half that many distinct molecules "
              "(exact count only known after generating them; a molecule that is its own "
              "mirror is kept once, not eliminated).")
    else:
        print(f"Total molecules to evaluate: {n_naive:,} (symmetry deduplication OFF — "
              "roughly half of these are the same physical molecule under a different "
              "position numbering; see train_pipeline.py's build_symmetry_mirror).")
    if limit:
        print(f"Capped to {limit:,} by --limit.")

    if dry_run:
        print("Dry run — stopping before loading the model or predicting.")
        return None

    package = load_model_package(model_dir, label)
    if package.scaler_x.n_features_in_ != len(ALL_64_COLS):
        raise ValueError(
            f"Model '{label}' expects {package.scaler_x.n_features_in_} descriptors; "
            f"this screening script builds {len(ALL_64_COLS)}-descriptor vectors "
            "(use the E2E_64 package for combinatorial screening)."
        )

    ad = package.applicability_domain
    ad_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ad["training_scaled_X"])
    ad_threshold = float(ad["threshold"])

    os.makedirs(output_dir, exist_ok=True)

    all_assignments = iter_assignments(vocabulary, n_substituents, dedupe_symmetry=dedupe_symmetry)
    n_estimate = (n_naive + 1) // 2 if dedupe_symmetry else n_naive
    n_estimate = min(n_estimate, limit) if limit else n_estimate

    spectra_chunks, meta_rows = [], []
    processed = 0
    t0 = time.time()

    while limit is None or processed < limit:
        take = chunk_size if limit is None else min(chunk_size, limit - processed)
        chunk = list(itertools.islice(all_assignments, take))
        if not chunk:
            break  # generator exhausted (this is the normal stopping condition)

        x = build_descriptor_matrix(chunk, library)
        x_scaled = package.scaler_x.transform(x)
        y_scaled = package.model.predict(x_scaled, batch_size=256, verbose=0)
        y = package.scaler_y.inverse_transform(y_scaled)

        distances, _ = ad_nn.kneighbors(x_scaled)
        distances = distances[:, 0]

        names = library.get("names", {})
        spectra_chunks.append(y.astype(np.float32))
        for (position_combo, substituent_combo), dist in zip(chunk, distances):
            meta_rows.append({
                "positions": ";".join(str(p) for p in position_combo),
                "substituents": ";".join(names.get(k, k) for k in substituent_combo),
                "substituent_hammett_keys": ";".join(substituent_combo),
                "ad_distance": float(dist),
                "outside_applicability_domain": bool(dist > ad_threshold),
            })

        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  {processed:,} (~{100 * processed / max(n_estimate, 1):.1f}% of estimate) "
              f"— {rate:.0f} molecules/s", end="\r")

    print()
    spectra = np.vstack(spectra_chunks)
    meta = pd.DataFrame(meta_rows)
    summary = summarize_spectra(spectra, package.wavelengths)
    results = pd.concat([meta, summary], axis=1)

    spectra_path = os.path.join(output_dir, f"{label}_screening_spectra.npy")
    wl_path = os.path.join(output_dir, f"{label}_screening_wavelengths.npy")
    results_path = os.path.join(output_dir, f"{label}_screening_results.csv")
    np.save(spectra_path, spectra)
    np.save(wl_path, package.wavelengths)
    results.to_csv(results_path, index=False)

    n_outside = int(results["outside_applicability_domain"].sum())
    print(f"\nDone. {len(results):,} molecules evaluated in {time.time() - t0:.1f}s.")
    print(f"{n_outside:,} ({100 * n_outside / len(results):.1f}%) fall outside the "
          f"applicability domain (distance > {ad_threshold:.3f}) — treat those with caution.")
    print(f"Results table: {results_path}")
    print(f"Spectra matrix: {spectra_path}  (row i matches results.csv row i)")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combinatorial ECD screening for helicenes.")
    p.add_argument("--model-dir", required=True, help="Unzipped E2E_64 model package directory.")
    p.add_argument("--label", default="E2E_64", choices=["E2E_64", "E2E_H"])
    p.add_argument("--library", required=True, help="Path to substituent_library.json.")
    p.add_argument("--n-substituents", type=int, default=2, help="Number of substituted positions (k).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--include-partial", action="store_true",
                    help="Include substituents that only have VdW defined (r_plus/r_minus default to 0).")
    p.add_argument("--dedupe-symmetry", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip molecules that are the same physical compound as one already "
                         "generated under the helicene's C2 symmetry (position i <-> 17-i). "
                         "On by default; use --no-dedupe-symmetry to screen the raw, "
                         "unreduced combinatorial space instead.")
    p.add_argument("--chunk-size", type=int, default=5000, help="Molecules per batched prediction call.")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many molecules (for testing).")
    p.add_argument("--dry-run", action="store_true", help="Only print the combinatorial count and exit.")
    p.add_argument("--sort-by", default="max_abs_intensity",
                    choices=["max_abs_intensity", "integrated_abs_area", "n_sign_changes", "ad_distance"])
    p.add_argument("--top-n", type=int, default=20, help="Print the top N results sorted by --sort-by.")
    args, _unknown = p.parse_known_args()  # tolerate Colab/Jupyter's injected argv
    return args


if __name__ == "__main__":
    args = parse_args()
    results = screen(
        model_dir=args.model_dir, label=args.label, library_path=args.library,
        n_substituents=args.n_substituents, output_dir=args.output_dir,
        include_partial=args.include_partial, chunk_size=args.chunk_size,
        limit=args.limit, dry_run=args.dry_run, dedupe_symmetry=args.dedupe_symmetry,
    )
    if results is not None:
        print(f"\nTop {args.top_n} by {args.sort_by}:")
        print(results.sort_values(args.sort_by, ascending=False).head(args.top_n).to_string(index=False))
