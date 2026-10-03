#!/usr/bin/env python3
"""
design_candidates.py — Random candidate screening and goal-directed inverse
design for helicene ECD spectra, on top of a trained E2E_64 model package.

Three modes:

1. RANDOM SCREENING (random_search): generates many random molecules (random
   number of substituents k, random positions, random substituent identity),
   evaluates them in batch, and ranks by whatever objective you care about
   (e.g. "maximize the integral of the positive part of the spectrum").
   This explores across ALL substitution levels at once -- unlike
   screen_substituents.py, which is exhaustive but only for one fixed k.

2. INVERSE DESIGN, greedy (hill_climb_search): a simple local search. Starting
   from random seed molecules, it repeatedly proposes small structural
   mutations and keeps whichever neighbor improves the objective. Fast and
   simple, but single-path: for a handful of substituents (roughly k<=5) it
   usually finds very good candidates quickly; for larger, more entangled
   search spaces it can get stuck in a local optimum that random restarts
   don't fully escape.

3. INVERSE DESIGN, genetic (genetic_search): a real population-based genetic
   algorithm (crossover + mutation + elitism), for when you want to search
   broadly across k=6, 7, 10, even all 16 positions at once. Because it keeps
   a whole population and recombines good partial solutions from different
   individuals (crossover), it explores much more of the space than a single
   greedy path, at the cost of more evaluations per generation. Prefer this
   over hill_climb_search once k gets large or you suspect the objective has
   many separate local optima (substituents that only help in combination).

All three reuse the vocabulary/descriptor-building/symmetry-deduplication
code from screen_substituents.py and the model loader from predict_ecd.py.

Example
-------
    python design_candidates.py \\
        --model-dir E2E_64_package --label E2E_64 \\
        --library substituent_library.json \\
        --mode genetic --objective positive_integral \\
        --k-min 1 --k-max 10 --population-size 300 --n-generations 80 \\
        --output-dir design_results
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from predict_ecd import load_model_package
from screen_substituents import (
    ALL_64_COLS, N_POSITIONS, build_descriptor_matrix, build_vocabulary,
    is_canonical, load_library, mirror_assignment,
)

Candidate = tuple  # (position_tuple, substituent_tuple), positions sorted ascending
PAIR_INDEX = [(i, N_POSITIONS + 1 - i) for i in range(1, N_POSITIONS // 2 + 1)]  # (1,16),(2,15),...,(8,9)


# --------------------------------------------------------------------------- #
# Spectrum metrics (extends screen_substituents.summarize_spectra with the   #
# positive/negative band integrals requested for ranking candidates)         #
# --------------------------------------------------------------------------- #

def spectrum_metrics(spectrum: np.ndarray, wavelengths: np.ndarray) -> dict:
    positive_part = np.clip(spectrum, 0, None)
    negative_part = np.clip(spectrum, None, 0)
    max_idx = int(np.argmax(np.abs(spectrum)))
    return {
        "positive_integral": float(np.trapezoid(positive_part, wavelengths)),
        "negative_integral": float(np.trapezoid(negative_part, wavelengths)),  # negative number
        "max_abs_intensity": float(np.abs(spectrum)[max_idx]),
        "wavelength_at_max_nm": float(wavelengths[max_idx]),
        "integrated_abs_area": float(np.trapezoid(np.abs(spectrum), wavelengths)),
        "n_sign_changes": int(np.sum(np.diff(np.sign(spectrum)) != 0)),
    }


# Built-in named objectives. Each takes a row (dict-like, the metrics above
# plus 'wavelengths'/'spectrum' arrays) and returns a score to MAXIMIZE.
BUILTIN_OBJECTIVES = {
    "positive_integral": lambda row: row["positive_integral"],
    "negative_integral_abs": lambda row: -row["negative_integral"],
    "max_abs_intensity": lambda row: row["max_abs_intensity"],
    "integrated_abs_area": lambda row: row["integrated_abs_area"],
    "band_at_wavelength": lambda row: row["max_abs_intensity"],  # only meaningful with target_nm set
}


def intensity_at(row: dict, target_nm: float) -> float:
    """Helper for custom objectives: interpolated spectrum value at target_nm."""
    return float(np.interp(target_nm, row["wavelengths"], row["spectrum"]))


def match_target_spectrum_objective(target_spectrum: np.ndarray) -> Callable:
    """Objective factory: higher score = closer (cosine similarity) to a
    given target spectrum (must be on the same wavelength grid)."""
    def objective(row):
        s = row["spectrum"]
        num = float(np.dot(s, target_spectrum))
        den = float(np.linalg.norm(s) * np.linalg.norm(target_spectrum) + 1e-12)
        return num / den
    return objective


def resolve_objective(objective, target_nm=False, window_nm: float = 30.0) -> Callable:
    """target_nm is an ON/OFF switch: False (default) -> the objective is
    computed over the FULL spectrum, as usual. A number -> whichever
    objective you picked is instead computed only within [target_nm -
    window_nm, target_nm + window_nm], e.g. objective='positive_integral'
    with target_nm=450 means "the best positive band, but specifically
    around 450 nm" instead of anywhere in the spectrum."""
    if objective == "band_at_wavelength" and (target_nm is False or target_nm is None):
        raise ValueError("objective='band_at_wavelength' requires target_nm to be set (it's meaningless "
                          "over the whole spectrum) -- or just turn target_nm on with any other objective.")

    if callable(objective):
        base = objective
    elif objective in BUILTIN_OBJECTIVES:
        base = BUILTIN_OBJECTIVES[objective]
    else:
        raise ValueError(f"Unknown objective '{objective}'. Built-ins: {list(BUILTIN_OBJECTIVES)}, "
                          "or pass your own callable(row) -> float.")

    if target_nm is False or target_nm is None:
        return base

    def windowed(row):
        wl, spec = row["wavelengths"], row["spectrum"]
        mask = np.abs(wl - target_nm) <= window_nm
        if not mask.any():
            mask = np.zeros_like(wl, dtype=bool)
            mask[int(np.argmin(np.abs(wl - target_nm)))] = True
        sub_row = dict(row)
        sub_row.update(spectrum_metrics(spec[mask], wl[mask]))
        sub_row["wavelengths"], sub_row["spectrum"] = wl[mask], spec[mask]
        return base(sub_row)

    return windowed


def apply_ad_penalty(score_fn: Callable, ad_penalty_weight=False) -> Callable:
    """ON/OFF switch, same pattern as everything else: ad_penalty_weight=False
    (default) leaves the score untouched -- candidates outside the
    applicability domain compete on equal footing with everything else, and
    you only find out via the 'outside AD' flag afterward. A positive
    number (try 1.0 to start) instead down-weights the score the further a
    candidate sits past the AD threshold, so the search itself steers away
    from unreliable extrapolations rather than just reporting them.

    The penalty is RELATIVE (scaled by how many threshold-widths past the
    threshold the candidate is), so the same ad_penalty_weight works
    regardless of the objective's absolute scale:
        factor = 1 / (1 + ad_penalty_weight * excess_fraction)
        adjusted_score = raw_score * factor
    A candidate exactly AT the threshold is untouched (factor=1); one at
    2x the threshold distance with ad_penalty_weight=1.0 gets roughly
    halved; higher ad_penalty_weight makes the penalty bite harder.
    Assumes the objective's raw scores are non-negative (true for all the
    BUILTIN_OBJECTIVES); with a custom objective that can go negative,
    the direction of the penalty may not behave as expected."""
    if ad_penalty_weight is False or ad_penalty_weight is None:
        return score_fn

    def penalized(row):
        base = score_fn(row)
        threshold, distance = row.get("ad_threshold"), row.get("ad_distance")
        if not threshold:
            return base
        excess_fraction = max(0.0, (distance - threshold) / threshold)
        factor = 1.0 / (1.0 + ad_penalty_weight * excess_fraction)
        return base * factor

    return penalized


# --------------------------------------------------------------------------- #
# Candidate generation                                                        #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Search space: which substituents, which positions, forced symmetry         #
# --------------------------------------------------------------------------- #

SUBSTITUENT_ALIASES = {"br": "bromo", "cl": "cloro", "i": "yodo", "iodo": "yodo", "chloro": "cloro"}


def resolve_substituent_selection(items: list, library: dict) -> list:
    """Maps user-friendly names ('NO2', 'Br', case-insensitive) or raw
    Hammett keys ('0.78') to the internal vocabulary keys."""
    names_to_key = {v.lower(): k for k, v in library.get("names", {}).items()}
    valid_keys = set(library["descriptors"]["vdw"].keys())
    resolved, unresolved = [], []
    for item in items:
        s = str(item).strip()
        low = s.lower()
        if s in valid_keys:
            resolved.append(s)
        elif low in names_to_key:
            resolved.append(names_to_key[low])
        elif low in SUBSTITUENT_ALIASES and SUBSTITUENT_ALIASES[low] in names_to_key:
            resolved.append(names_to_key[SUBSTITUENT_ALIASES[low]])
        else:
            unresolved.append(s)
    if unresolved:
        raise ValueError(f"Unknown substituent(s): {unresolved}. Known names: "
                          f"{sorted(library.get('names', {}).values())}")
    return sorted(set(resolved), key=float)


def build_search_space(library: dict, allowed_substituents=False, allowed_positions=False,
                        force_symmetric: bool = False) -> dict:
    """Builds and validates the restricted (vocabulary, positions, mirror
    pairs) search space from the on/off-style filters:
      allowed_substituents: False -> full vocabulary; else a list of names/keys.
      allowed_positions:    False -> all 16 positions;  else a list of ints.
      force_symmetric:      only generate/search molecules where position i
                             and 17-i always carry the SAME substituent.
    Raises a clear error for infeasible combinations instead of failing silently.
    """
    vocabulary = build_vocabulary(library)
    if allowed_substituents:
        vocabulary = resolve_substituent_selection(allowed_substituents, library)
    if not vocabulary:
        raise ValueError("allowed_substituents resolved to an empty vocabulary.")

    positions = list(range(1, N_POSITIONS + 1))
    if allowed_positions:
        positions = sorted({int(p) for p in allowed_positions})
        bad = [p for p in positions if not (1 <= p <= N_POSITIONS)]
        if bad:
            raise ValueError(f"allowed_positions out of range 1-{N_POSITIONS}: {bad}")
    if not positions:
        raise ValueError("allowed_positions resolved to an empty position list.")

    pairs = None
    if force_symmetric:
        position_set = set(positions)
        pairs = [(i, N_POSITIONS + 1 - i) for i in range(1, N_POSITIONS // 2 + 1)
                 if i in position_set and (N_POSITIONS + 1 - i) in position_set]
        if not pairs:
            raise ValueError(
                "force_symmetric=True but allowed_positions contains no complete mirror pair "
                f"(position i needs its partner 17-i also allowed). allowed_positions={positions}"
            )

    return {"vocabulary": vocabulary, "positions": positions, "pairs": pairs,
            "force_symmetric": force_symmetric}


# --------------------------------------------------------------------------- #
# Candidate generation (space-aware: respects allowed substituents/positions #
# and, if requested, forces exact C2 symmetry)                                #
# --------------------------------------------------------------------------- #

def random_candidate(space: dict, k_min: int, k_max: int, rng: np.random.Generator) -> Candidate:
    vocabulary = space["vocabulary"]

    if space["force_symmetric"]:
        pairs = space["pairs"]
        n_pairs_max = min(len(pairs), max(1, k_max // 2))
        n_pairs_min = min(n_pairs_max, max(1, -(-k_min // 2)))  # ceil(k_min/2)
        n_pairs = int(rng.integers(n_pairs_min, n_pairs_max + 1))
        chosen = rng.choice(len(pairs), size=n_pairs, replace=False)
        positions, substituents = [], []
        for idx in chosen:
            i, j = pairs[int(idx)]
            sub = str(rng.choice(vocabulary))
            positions += [i, j]
            substituents += [sub, sub]
        order = np.argsort(positions)
        return (tuple(np.array(positions)[order].tolist()),
                tuple(np.array(substituents)[order].tolist()))

    positions_pool = space["positions"]
    k = min(int(rng.integers(k_min, k_max + 1)), len(positions_pool))
    positions = tuple(sorted(rng.choice(positions_pool, size=k, replace=False).tolist()))
    substituents = tuple(rng.choice(vocabulary, size=k, replace=True).tolist())
    return positions, substituents


def mutate_symmetric(candidate: Candidate, space: dict, k_min: int, k_max: int,
                      rng: np.random.Generator) -> Candidate:
    pos_to_sub = dict(zip(*candidate))
    pairs, vocabulary = space["pairs"], space["vocabulary"]
    used_pairs = [(i, j) for (i, j) in pairs if i in pos_to_sub]
    free_pairs = [(i, j) for (i, j) in pairs if i not in pos_to_sub]
    n_pairs = len(used_pairs)

    moves = ["substitute"] if used_pairs else []
    if n_pairs < max(1, k_max // 2) and free_pairs:
        moves.append("add")
    if n_pairs > max(1, -(-k_min // 2)):
        moves.append("remove")
    if free_pairs and used_pairs:
        moves.append("move")
    if not moves:
        return candidate
    move = rng.choice(moves)

    if move == "substitute":
        i, j = used_pairs[int(rng.integers(len(used_pairs)))]
        new_sub = str(rng.choice(vocabulary))
        pos_to_sub[i] = new_sub; pos_to_sub[j] = new_sub
    elif move == "add":
        i, j = free_pairs[int(rng.integers(len(free_pairs)))]
        new_sub = str(rng.choice(vocabulary))
        pos_to_sub[i] = new_sub; pos_to_sub[j] = new_sub
    elif move == "remove":
        i, j = used_pairs[int(rng.integers(len(used_pairs)))]
        del pos_to_sub[i]; del pos_to_sub[j]
    elif move == "move":
        i, j = used_pairs[int(rng.integers(len(used_pairs)))]
        sub = pos_to_sub.pop(i); pos_to_sub.pop(j)
        ni, nj = free_pairs[int(rng.integers(len(free_pairs)))]
        pos_to_sub[ni] = sub; pos_to_sub[nj] = sub

    new_positions = sorted(pos_to_sub)
    return tuple(new_positions), tuple(pos_to_sub[p] for p in new_positions)


def mutate(candidate: Candidate, space: dict, k_min: int, k_max: int, rng: np.random.Generator) -> Candidate:
    if space["force_symmetric"]:
        return mutate_symmetric(candidate, space, k_min, k_max, rng)

    vocabulary = space["vocabulary"]
    positions, substituents = candidate
    positions, substituents = list(positions), list(substituents)
    k = len(positions)
    occupied = set(positions)
    free = [p for p in space["positions"] if p not in occupied]

    moves = ["substitute"]
    if k < k_max and free:
        moves.append("add")
    if k > k_min:
        moves.append("remove")
    if free:
        moves.append("move")
    move = rng.choice(moves)

    if move == "substitute":
        i = int(rng.integers(0, k))
        substituents[i] = str(rng.choice(vocabulary))
    elif move == "add":
        new_pos = int(rng.choice(free))
        idx = np.searchsorted(positions, new_pos)
        positions.insert(idx, new_pos)
        substituents.insert(idx, str(rng.choice(vocabulary)))
    elif move == "remove":
        i = int(rng.integers(0, k))
        positions.pop(i); substituents.pop(i)
    elif move == "move":
        i = int(rng.integers(0, k))
        new_pos = int(rng.choice(free))
        sub = substituents.pop(i)
        positions.pop(i)
        idx = np.searchsorted(positions, new_pos)
        positions.insert(idx, new_pos)
        substituents.insert(idx, sub)

    return tuple(positions), tuple(substituents)


# --------------------------------------------------------------------------- #
# Genetic algorithm representation and operators                              #
# --------------------------------------------------------------------------- #
#
# A "genome" is a FIXED-length tuple: 16 genes (one per position), or 8 genes
# (one per mirror pair) if force_symmetric. Each gene is either None
# (unsubstituted, H) or a substituent key. Fixed length makes crossover
# trivial (swap genes between two parents) -- unlike the variable-length
# (positions, substituents) Candidate used by hill-climbing, which doesn't
# recombine cleanly between two molecules of different sizes.

def make_genome(candidate: Candidate, force_symmetric: bool) -> tuple:
    pos_to_sub = dict(zip(*candidate))
    if force_symmetric:
        return tuple(pos_to_sub.get(i) for i, _ in PAIR_INDEX)
    return tuple(pos_to_sub.get(p) for p in range(1, N_POSITIONS + 1))


def genome_to_candidate(genome: tuple, force_symmetric: bool) -> Candidate:
    if force_symmetric:
        positions, substituents = [], []
        for (i, j), sub in zip(PAIR_INDEX, genome):
            if sub is not None:
                positions += [i, j]; substituents += [sub, sub]
    else:
        positions = [p for p, sub in zip(range(1, N_POSITIONS + 1), genome) if sub is not None]
        substituents = [sub for sub in genome if sub is not None]
    order = np.argsort(positions)
    return (tuple(np.array(positions)[order].tolist()), tuple(np.array(substituents)[order].tolist())) \
        if positions else (tuple(), tuple())


def genome_allowed_gene_indices(space: dict) -> list:
    """Which gene indices are actually allowed to hold a substituent, given
    allowed_positions (and, if symmetric, which mirror pairs are complete)."""
    if space["force_symmetric"]:
        pair_set = set(space["pairs"])
        return [k for k, pair in enumerate(PAIR_INDEX) if pair in pair_set]
    position_set = set(space["positions"])
    return [p - 1 for p in range(1, N_POSITIONS + 1) if p in position_set]


def random_genome(space: dict, k_min: int, k_max: int, rng: np.random.Generator) -> tuple:
    return make_genome(random_candidate(space, k_min, k_max, rng), space["force_symmetric"])


def crossover_genome(genome_a: tuple, genome_b: tuple, rng: np.random.Generator) -> tuple:
    """Uniform crossover: each gene independently comes from parent A or B."""
    return tuple(a if rng.random() < 0.5 else b for a, b in zip(genome_a, genome_b))


def mutate_genome(genome: tuple, space: dict, rng: np.random.Generator, mutation_rate: float = 0.15) -> tuple:
    vocabulary = space["vocabulary"]
    genome = list(genome)
    for idx in genome_allowed_gene_indices(space):
        if rng.random() >= mutation_rate:
            continue
        if genome[idx] is None:
            genome[idx] = str(rng.choice(vocabulary))
        elif rng.random() < 0.3:
            genome[idx] = None  # clear this gene
        else:
            genome[idx] = str(rng.choice(vocabulary))  # swap substituent
    return tuple(genome)


def repair_genome_k(genome: tuple, space: dict, k_min: int, k_max: int, rng: np.random.Generator) -> tuple:
    """After crossover/mutation, the number of substituted positions can
    drift outside [k_min, k_max]; randomly add/remove genes to fix it."""
    genome = list(genome)
    unit = 2 if space["force_symmetric"] else 1
    allowed = genome_allowed_gene_indices(space)
    occupied = [idx for idx in allowed if genome[idx] is not None]
    free = [idx for idx in allowed if genome[idx] is None]
    current_k = len(occupied) * unit

    rng.shuffle(occupied)
    while current_k > k_max and occupied:
        genome[occupied.pop()] = None
        current_k -= unit

    rng.shuffle(free)
    while current_k < k_min and free:
        genome[free.pop()] = str(rng.choice(space["vocabulary"]))
        current_k += unit

    return tuple(genome)


def tournament_select(fitness: list, k: int, rng: np.random.Generator) -> int:
    idxs = rng.choice(len(fitness), size=min(k, len(fitness)), replace=False)
    return int(max(idxs, key=lambda i: fitness[i]))


# --------------------------------------------------------------------------- #
# Batch evaluation                                                            #
# --------------------------------------------------------------------------- #

class Evaluator:
    """Wraps a loaded model package for repeated batch evaluation, keeping the
    applicability-domain NearestNeighbors fit once (not re-fit per call)."""

    def __init__(self, model_dir: str, label: str, library: dict):
        self.package = load_model_package(model_dir, label)
        if self.package.scaler_x.n_features_in_ != len(ALL_64_COLS):
            raise ValueError(f"'{label}' isn't a 64-descriptor model; design_candidates.py needs E2E_64.")
        self.library = library
        self.names = library.get("names", {})
        ad = self.package.applicability_domain
        self.ad_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ad["training_scaled_X"])
        self.ad_threshold = float(ad["threshold"])

    def evaluate(self, candidates: list) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame()
        x = build_descriptor_matrix(candidates, self.library)
        x_scaled = self.package.scaler_x.transform(x)
        y_scaled = self.package.model.predict(x_scaled, batch_size=256, verbose=0)
        y = self.package.scaler_y.inverse_transform(y_scaled)
        distances, _ = self.ad_nn.kneighbors(x_scaled)

        rows = []
        for (positions, substituents), spectrum, dist in zip(candidates, y, distances[:, 0]):
            m = spectrum_metrics(spectrum, self.package.wavelengths)
            m.update({
                "positions": ";".join(str(p) for p in positions),
                "substituents": ";".join(self.names.get(k, k) for k in substituents),
                "substituent_hammett_keys": ";".join(substituents),
                "n_substituents": len(positions),
                "ad_distance": float(dist),
                "ad_threshold": self.ad_threshold,
                "outside_applicability_domain": bool(dist > self.ad_threshold),
                "wavelengths": self.package.wavelengths,
                "spectrum": spectrum,
            })
            rows.append(m)
        return pd.DataFrame(rows)



def format_candidate_full(row) -> str:
    """'5;7;8' + 'NO2;CHO;CN' -> 'Pos5=NO2, Pos7=CHO, Pos8=CN' -- the
    complete, unabbreviated position->substituent mapping, with no length
    limit. Meant for a printed legend, not a plot title."""
    positions = str(row["positions"]).split(";")
    substituents = str(row["substituents"]).split(";")
    return ", ".join(f"Pos{p}={s}" for p, s in zip(positions, substituents))


def plot_candidates(df: pd.DataFrame, n: int = 6, label: str = "", title: str = "") -> None:
    """Plots the predicted spectrum of the top n rows of df (as returned by
    random_search / hill_climb_search / genetic_search, BEFORE the
    wavelengths/spectrum columns are dropped for CSV export).

    Each panel's title is just 'Candidate N' -- kept short on purpose so it
    never overflows into the neighboring panel, however many substituents a
    molecule has (1 or 16, doesn't matter). The FULL position->substituent
    detail for every candidate is printed as a legend right after the
    figure, where there's no width limit to fight against."""
    if "spectrum" not in df.columns:
        print("This dataframe doesn't carry spectra anymore (probably loaded back from a saved "
              "CSV, which drops them to keep the file small) -- nothing to plot.")
        return

    top = df.head(n)
    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.6 * nrows))
    axes = np.asarray(axes).ravel()

    line_color = "#C0272D"       # clean, strong red
    accent_color = "#4A4A4A"     # dark neutral grey for the zero line

    for i, (ax, (_, row)) in enumerate(zip(axes, top.iterrows()), start=1):
        wl, spectrum = row["wavelengths"], row["spectrum"]
        ax.plot(wl, spectrum, linewidth=3, color=line_color, solid_capstyle="round")
        ax.axhline(0, color=accent_color, linewidth=1, linestyle="-", alpha=0.6, zorder=0)
        ax.set_xlim(float(wl.min()), float(wl.max()))  # start exactly at 250 nm, no padding

        ad_flag = "  ⚠ outside AD" if row["outside_applicability_domain"] else ""
        ax.set_title(f"Candidate {i}{ad_flag}\nscore = {row['score']:.3g}",
                     fontsize=19, fontweight="bold", pad=14)
        ax.set_xlabel("Wavelength (nm)", fontsize=17)
        ax.set_ylabel("Predicted ECD intensity", fontsize=17)
        ax.tick_params(axis="both", labelsize=15)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.grid(alpha=0.2, linewidth=0.7)
        ax.set_facecolor("#FAFAFA")

    for ax in axes[len(top):]:
        ax.axis("off")

    fig.suptitle(title or f"{label} — top {len(top)} candidates", fontsize=23, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    print("\nWhere exactly each candidate is substituted:")
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        ad_flag = "  ⚠ outside AD" if row["outside_applicability_domain"] else ""
        print(f"  Candidate {i} (score={row['score']:.3g}){ad_flag}: {format_candidate_full(row)}")


# --------------------------------------------------------------------------- #
# Mode 1: random screening                                                    #
# --------------------------------------------------------------------------- #

def random_search(model_dir: str, label: str, library_path: str, n_samples: int,
                   k_min: int = 1, k_max: int = 4, objective="positive_integral",
                   target_nm=False, window_nm: float = 30.0, ad_penalty_weight=False,
                   allowed_substituents=False, allowed_positions=False, force_symmetric: bool = False,
                   seed: int = 42, chunk_size: int = 2000, output_dir: Optional[str] = None,
                   top_n: int = 20, plot_top_n: int = 6) -> pd.DataFrame:
    library = load_library(library_path)
    space = build_search_space(library, allowed_substituents, allowed_positions, force_symmetric)
    evaluator = Evaluator(model_dir, label, library)
    score_fn = apply_ad_penalty(resolve_objective(objective, target_nm, window_nm), ad_penalty_weight)
    rng = np.random.default_rng(seed)

    seen = set()
    results = []
    t0 = time.time()
    n_generated = 0
    while n_generated < n_samples:
        chunk = []
        while len(chunk) < min(chunk_size, n_samples - n_generated):
            cand = random_candidate(space, k_min, k_max, rng)
            canonical = cand if is_canonical(*cand) else mirror_assignment(*cand)
            if canonical in seen:
                continue
            seen.add(canonical)
            chunk.append(canonical)
        df = evaluator.evaluate(chunk)
        df["score"] = df.apply(lambda r: score_fn(r), axis=1)
        results.append(df)
        n_generated += len(chunk)
        print(f"  {n_generated:,}/{n_samples:,} evaluated — {n_generated / (time.time() - t0):.0f} molecules/s",
              end="\r")

    print()
    out = pd.concat(results, ignore_index=True).sort_values("score", ascending=False)
    print(f"\nTop {top_n} by '{objective}':")
    print(out.drop(columns=["wavelengths", "spectrum"]).head(top_n).to_string(index=False))

    if plot_top_n:
        plot_candidates(out, n=plot_top_n, label=label, title=f"{label} — random search top candidates by '{objective}'")

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{label}_random_search_results.csv")
        out.drop(columns=["wavelengths", "spectrum"]).to_csv(path, index=False)
        print(f"\nSaved: {path}")

    return out


# --------------------------------------------------------------------------- #
# Mode 2: inverse design via greedy hill-climbing                             #
# --------------------------------------------------------------------------- #

def hill_climb_search(model_dir: str, label: str, library_path: str, objective="positive_integral",
                       target_nm=False, window_nm: float = 30.0, ad_penalty_weight=False,
                       allowed_substituents=False, allowed_positions=False, force_symmetric: bool = False,
                       k_min: int = 1, k_max: int = 4, n_restarts: int = 8, n_steps: int = 100,
                       n_neighbors_per_step: int = 6, seed: int = 42, output_dir: Optional[str] = None,
                       top_n: int = 20, plot_top_n: int = 6) -> pd.DataFrame:
    library = load_library(library_path)
    space = build_search_space(library, allowed_substituents, allowed_positions, force_symmetric)
    evaluator = Evaluator(model_dir, label, library)
    score_fn = apply_ad_penalty(resolve_objective(objective, target_nm, window_nm), ad_penalty_weight)
    rng = np.random.default_rng(seed)

    all_evaluated = {}  # canonical candidate -> row dict, across the whole run
    trajectory = []      # (restart, step, best_score_so_far) for diagnostics

    def eval_and_record(cands):
        new = [c for c in cands if (c if is_canonical(*c) else mirror_assignment(*c)) not in all_evaluated]
        if not new:
            return
        df = evaluator.evaluate(new)
        df["score"] = df.apply(lambda r: score_fn(r), axis=1)
        for cand, (_, row) in zip(new, df.iterrows()):
            canonical = cand if is_canonical(*cand) else mirror_assignment(*cand)
            all_evaluated[canonical] = row

    t0 = time.time()
    for restart in range(n_restarts):
        current = random_candidate(space, k_min, k_max, rng)
        eval_and_record([current])
        current_score = all_evaluated[current if is_canonical(*current) else mirror_assignment(*current)]["score"]

        for step in range(n_steps):
            neighbors = [mutate(current, space, k_min, k_max, rng) for _ in range(n_neighbors_per_step)]
            eval_and_record(neighbors)
            scored = [(n, all_evaluated[n if is_canonical(*n) else mirror_assignment(*n)]["score"])
                      for n in neighbors]
            best_neighbor, best_neighbor_score = max(scored, key=lambda t: t[1])

            if best_neighbor_score > current_score:
                current, current_score = best_neighbor, best_neighbor_score
            else:
                break  # local optimum for this restart

            trajectory.append({"restart": restart, "step": step, "score": current_score})

        print(f"  restart {restart + 1}/{n_restarts}: best so far "
              f"{max(r['score'] for r in all_evaluated.values()):.3f} "
              f"({len(all_evaluated)} unique molecules evaluated, {time.time() - t0:.0f}s)", end="\r")

    print()
    out = pd.DataFrame(list(all_evaluated.values())).sort_values("score", ascending=False)
    print(f"\nTop {top_n} by '{objective}' (hill-climbing, {len(all_evaluated)} unique molecules evaluated "
          f"across {n_restarts} restarts):")
    print(out.drop(columns=["wavelengths", "spectrum"]).head(top_n).to_string(index=False))

    if plot_top_n:
        plot_candidates(out, n=plot_top_n, label=label,
                         title=f"{label} — hill-climb top candidates by '{objective}'")

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{label}_hill_climb_results.csv")
        out.drop(columns=["wavelengths", "spectrum"]).to_csv(path, index=False)
        print(f"\nSaved: {path}")
        pd.DataFrame(trajectory).to_csv(os.path.join(output_dir, f"{label}_hill_climb_trajectory.csv"), index=False)

    return out


# --------------------------------------------------------------------------- #
# Mode 3: inverse design via a real genetic algorithm                         #
# --------------------------------------------------------------------------- #

def genetic_search(model_dir: str, label: str, library_path: str, objective="positive_integral",
                    target_nm=False, window_nm: float = 30.0, ad_penalty_weight=False,
                    allowed_substituents=False, allowed_positions=False, force_symmetric: bool = False,
                    k_min: int = 1, k_max: int = 16, population_size: int = 300, n_generations: int = 60,
                    mutation_rate: float = 0.15, elite_frac: float = 0.1, tournament_size: int = 4,
                    seed: int = 42, output_dir: Optional[str] = None,
                    top_n: int = 20, plot_top_n: int = 6) -> pd.DataFrame:
    """Population-based genetic search: crossover recombines good partial
    solutions from different individuals (unlike hill_climb_search, which
    only ever refines one solution at a time), so it covers large k (6, 7,
    10, even 16 -- all positions at once) far more broadly for a given
    evaluation budget."""
    library = load_library(library_path)
    space = build_search_space(library, allowed_substituents, allowed_positions, force_symmetric)
    evaluator = Evaluator(model_dir, label, library)
    score_fn = apply_ad_penalty(resolve_objective(objective, target_nm, window_nm), ad_penalty_weight)
    rng = np.random.default_rng(seed)

    all_evaluated = {}  # canonical candidate -> row, across the whole run
    history = []         # (generation, best_score, mean_score) for diagnostics

    def eval_population(genomes: list) -> list:
        candidates = [genome_to_candidate(g, space["force_symmetric"]) for g in genomes]
        canonical = [c if is_canonical(*c) else mirror_assignment(*c) for c in candidates]
        new_idx = [i for i, c in enumerate(canonical) if c not in all_evaluated]
        if new_idx:
            df = evaluator.evaluate([candidates[i] for i in new_idx])
            df["score"] = df.apply(lambda r: score_fn(r), axis=1)
            for i, (_, row) in zip(new_idx, df.iterrows()):
                all_evaluated[canonical[i]] = row
        return [float(all_evaluated[c]["score"]) for c in canonical]

    population = [random_genome(space, k_min, k_max, rng) for _ in range(population_size)]
    fitness = eval_population(population)
    n_elite = max(1, int(population_size * elite_frac))
    t0 = time.time()

    for gen in range(n_generations):
        order = np.argsort(fitness)[::-1]
        elite = [population[i] for i in order[:n_elite]]

        children = []
        while len(children) < population_size - n_elite:
            i1 = tournament_select(fitness, tournament_size, rng)
            i2 = tournament_select(fitness, tournament_size, rng)
            child = crossover_genome(population[i1], population[i2], rng)
            child = mutate_genome(child, space, rng, mutation_rate)
            child = repair_genome_k(child, space, k_min, k_max, rng)
            children.append(child)

        population = elite + children
        fitness = eval_population(population)
        history.append({"generation": gen, "best_score": max(fitness), "mean_score": float(np.mean(fitness))})
        print(f"  gen {gen + 1}/{n_generations}: best={max(fitness):.3f}  mean={np.mean(fitness):.3f}  "
              f"({len(all_evaluated)} unique molecules evaluated, {time.time() - t0:.0f}s)", end="\r")

    print()
    out = pd.DataFrame(list(all_evaluated.values())).sort_values("score", ascending=False)
    print(f"\nTop {top_n} by '{objective}' (genetic algorithm, {n_generations} generations x "
          f"{population_size} population, {len(all_evaluated)} unique molecules evaluated):")
    print(out.drop(columns=["wavelengths", "spectrum"]).head(top_n).to_string(index=False))

    if plot_top_n:
        plot_candidates(out, n=plot_top_n, label=label,
                         title=f"{label} — genetic algorithm top candidates by '{objective}'")

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{label}_genetic_results.csv")
        out.drop(columns=["wavelengths", "spectrum"]).to_csv(path, index=False)
        print(f"\nSaved: {path}")
        pd.DataFrame(history).to_csv(os.path.join(output_dir, f"{label}_genetic_history.csv"), index=False)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random screening and inverse design of ECD candidates.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", default="E2E_64")
    p.add_argument("--library", required=True)
    p.add_argument("--mode", choices=["random", "hill_climb", "genetic"], default="random")
    p.add_argument("--objective", default="positive_integral",
                    help=f"One of {list(BUILTIN_OBJECTIVES) + ['band_at_wavelength']}, "
                         "or edit the script to pass a custom callable.")
    p.add_argument("--target-nm", type=float, default=None,
                    help="ON/OFF switch: omit for 'anywhere in the spectrum', or give a wavelength "
                         "to restrict the objective to that band region.")
    p.add_argument("--window-nm", type=float, default=30.0,
                    help="How far from --target-nm still counts as 'that band'.")
    p.add_argument("--ad-penalty-weight", type=float, default=None,
                    help="ON/OFF switch: omit to not penalize candidates outside the applicability "
                         "domain at all. A positive number (try 1.0) down-weights their score, "
                         "proportional to how far past the AD threshold they are.")
    p.add_argument("--allowed-substituents", nargs="+", default=None,
                    help="Restrict to these substituents only (names like NO2 CN F Br, or Hammett keys). "
                         "Omit for 'all'.")
    p.add_argument("--allowed-positions", type=int, nargs="+", default=None,
                    help="Restrict to these positions only (e.g. 1 2 3). Omit for 'all 16'.")
    p.add_argument("--force-symmetric", action="store_true",
                    help="Only consider molecules symmetric under position i <-> 17-i.")
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=20000, help="(random mode) how many molecules to try.")
    p.add_argument("--n-restarts", type=int, default=8, help="(hill_climb mode)")
    p.add_argument("--n-steps", type=int, default=100, help="(hill_climb mode) max steps per restart.")
    p.add_argument("--population-size", type=int, default=300, help="(genetic mode)")
    p.add_argument("--n-generations", type=int, default=60, help="(genetic mode)")
    p.add_argument("--mutation-rate", type=float, default=0.15, help="(genetic mode) per-gene mutation chance.")
    p.add_argument("--elite-frac", type=float, default=0.1, help="(genetic mode) fraction kept unchanged each gen.")
    p.add_argument("--tournament-size", type=int, default=4, help="(genetic mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--plot-top-n", type=int, default=6, help="How many top spectra to plot (0 to disable).")
    args, _unknown = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "random":
        random_search(args.model_dir, args.label, args.library, args.n_samples,
                       args.k_min, args.k_max, args.objective,
                       target_nm=args.target_nm, window_nm=args.window_nm, ad_penalty_weight=args.ad_penalty_weight,
                       allowed_substituents=args.allowed_substituents or False,
                       allowed_positions=args.allowed_positions or False,
                       force_symmetric=args.force_symmetric, seed=args.seed,
                       output_dir=args.output_dir, top_n=args.top_n, plot_top_n=args.plot_top_n)
    elif args.mode == "hill_climb":
        hill_climb_search(args.model_dir, args.label, args.library, args.objective,
                           target_nm=args.target_nm, window_nm=args.window_nm, ad_penalty_weight=args.ad_penalty_weight,
                           allowed_substituents=args.allowed_substituents or False,
                           allowed_positions=args.allowed_positions or False,
                           force_symmetric=args.force_symmetric,
                           k_min=args.k_min, k_max=args.k_max, n_restarts=args.n_restarts,
                           n_steps=args.n_steps, seed=args.seed, output_dir=args.output_dir,
                           top_n=args.top_n, plot_top_n=args.plot_top_n)
    else:
        genetic_search(args.model_dir, args.label, args.library, args.objective,
                        target_nm=args.target_nm, window_nm=args.window_nm, ad_penalty_weight=args.ad_penalty_weight,
                        allowed_substituents=args.allowed_substituents or False,
                        allowed_positions=args.allowed_positions or False,
                        force_symmetric=args.force_symmetric,
                        k_min=args.k_min, k_max=args.k_max,
                        population_size=args.population_size, n_generations=args.n_generations,
                        mutation_rate=args.mutation_rate, elite_frac=args.elite_frac,
                        tournament_size=args.tournament_size, seed=args.seed, output_dir=args.output_dir,
                        top_n=args.top_n, plot_top_n=args.plot_top_n)
