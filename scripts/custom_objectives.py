#!/usr/bin/env python3
"""
custom_objectives.py — Band-based design objectives for design_candidates.py
(random_search / hill_climb_search / genetic_search).

These are the objectives used for the inverse-design campaigns of the
HelicAI paper (Case IV; Supporting Information, Section S6). They are built
on the area of the SINGLE LARGEST contiguous band of each sign, the same
quantity used in the high-throughput screening (Case II, Figure 4):

  A+max  = area of the largest contiguous POSITIVE band
  |A-max| = area of the largest contiguous NEGATIVE band (as a positive number)

Why not the built-in "positive_integral"? That one is the total area of the
whole positive part of the spectrum (every positive lobe added together).
If a molecule has two separate positive bands, "positive_integral" sums both;
"positive_band" keeps only the bigger one.

Usage
-----
    import design_candidates as dc
    from custom_objectives import band_objective, annotate_band_metrics

    best = dc.genetic_search(
        model_dir="models/E2E_64", label="E2E_64",
        library_path="data/substituent_library.json",
        objective=band_objective("negative_band", "min"),   # a function, not a string
        ad_penalty_weight=1.0, k_min=1, k_max=6,
        population_size=1000, n_generations=50,
    )
    best = annotate_band_metrics(best)   # band areas in physical units

design_candidates.py also accepts the short string form
"<metric>:<direction>", e.g. objective="negative_band:min", and annotates
the band areas automatically.

metric
------
    "positive_band"  -> area of the single largest contiguous POSITIVE band (A+max)
    "negative_band"  -> area of the single largest contiguous NEGATIVE band (|A-max|)
    "peak_height"    -> height of the tallest peak or valley, whichever sign
    "positive_peak"  -> height of the tallest POSITIVE peak only
    "negative_peak"  -> depth of the deepest NEGATIVE valley only (positive number)

direction
---------
    "max" -> search for the LARGEST value of that quantity
    "min" -> search for the SMALLEST value (e.g. "almost no negative band").
             Implemented as maximizing 1 / (1 + value): positive, and
             monotonically decreasing with the value, which is what the
             applicability-domain penalty of design_candidates.py requires.
             The 'score' column is then this transformed value -- the
             physical band areas are in 'positive_band_area' /
             'negative_band_area' (see annotate_band_metrics).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

BAND_METRICS = ("positive_band", "negative_band", "peak_height", "positive_peak", "negative_peak")
DIRECTIONS = ("max", "min")


def _largest_band_area(spectrum: np.ndarray, wavelengths: np.ndarray, sign: str) -> float:
    """Area of the single largest contiguous band of the given sign.

    sign='positive' -> contiguous stretch where spectrum > 0.
    sign='negative' -> contiguous stretch where spectrum < 0 (returned as a
    positive number). Returns 0.0 if the spectrum has no region of that sign.

    Trapezoidal rule (np.trapezoid, as in design_candidates.spectrum_metrics).
    Each band is integrated including the flanking grid point on either side,
    which contributes zero after clipping (that is where the sign flips), so a
    single-band spectrum gives exactly the same area as the total integral of
    that sign. A band touching the edge of the 250-600 nm window contributes
    its edge point with half weight, as in the standard trapezoidal rule.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    part = np.clip(spectrum, 0, None) if sign == "positive" else -np.clip(spectrum, None, 0)
    is_on = part > 0
    if not is_on.any():
        return 0.0
    n = len(part)
    group_id = np.cumsum(is_on & ~np.r_[False, is_on[:-1]])
    group_id = np.where(is_on, group_id, -1)
    best = 0.0
    for g in np.unique(group_id):
        if g < 0:
            continue
        idx = np.where(group_id == g)[0]
        lo = max(int(idx[0]) - 1, 0)
        hi = min(int(idx[-1]) + 1, n - 1)
        sl = slice(lo, hi + 1)
        best = max(best, float(np.trapezoid(part[sl], wavelengths[sl])))
    return best


def _base_metric_fns() -> dict:
    def positive_band(row):
        return _largest_band_area(row["spectrum"], row["wavelengths"], "positive")

    def negative_band(row):
        return _largest_band_area(row["spectrum"], row["wavelengths"], "negative")

    def peak_height(row):
        return float(np.max(np.abs(row["spectrum"])))

    def positive_peak(row):
        return float(max(np.max(row["spectrum"]), 0.0))

    def negative_peak(row):
        return float(max(-np.min(row["spectrum"]), 0.0))

    return {"positive_band": positive_band, "negative_band": negative_band,
            "peak_height": peak_height, "positive_peak": positive_peak,
            "negative_peak": negative_peak}


def band_objective(metric: str = "positive_band", direction: str = "max") -> Callable:
    """Objective callable(row) -> float, to pass as `objective=...` to
    random_search / hill_climb_search / genetic_search.

    The returned value is always >= 0 (for 'max' the raw magnitude, for 'min'
    1 / (1 + magnitude)); both are bigger-is-better for the search and both
    are compatible with ad_penalty_weight, which assumes non-negative scores.
    """
    base_fns = _base_metric_fns()
    if metric not in base_fns:
        raise ValueError(f"metric must be one of {list(base_fns)}, got {metric!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
    base_fn = base_fns[metric]

    if direction == "max":
        def objective(row):
            return base_fn(row)
    else:
        def objective(row):
            return 1.0 / (1.0 + base_fn(row))

    objective.label = f"{metric}:{direction}"     # readable name for printouts
    objective.__name__ = f"{metric}_{direction}"
    return objective


def annotate_band_metrics(df):
    """Adds 'positive_band_area' (A+max) and 'negative_band_area' (|A-max|)
    columns, in R·10^40 / esu cm erg G^-1 × nm, to a results DataFrame
    returned by random_search / hill_climb_search / genetic_search.

    Needs the 'wavelengths' / 'spectrum' columns, i.e. call it on the
    DataFrame returned in memory, not on a CSV loaded back from disk.
    Always computed over the FULL 250-600 nm spectrum, whatever objective
    (or window) the search itself used, so every candidate can be read in
    physical units and not only through its raw 'score'.
    """
    df = df.copy()
    if df.empty:
        df["positive_band_area"] = []
        df["negative_band_area"] = []
        return df
    df["positive_band_area"] = df.apply(
        lambda r: _largest_band_area(r["spectrum"], r["wavelengths"], "positive"), axis=1)
    df["negative_band_area"] = df.apply(
        lambda r: _largest_band_area(r["spectrum"], r["wavelengths"], "negative"), axis=1)
    return df
