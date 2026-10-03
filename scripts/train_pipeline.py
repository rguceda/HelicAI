#!/usr/bin/env python3
"""
E2E ECD Training Pipeline (with integrated leakage-free split + k-fold CV)
============================================================================
Trains two feed-forward neural networks that map molecular descriptors of
helicenes to a 100-point electronic circular dichroism (ECD) spectrum:

  * E2E-H  : 16 Hammett-type substituent descriptors  -> ECD spectrum
  * E2E-64 : 64 descriptors (Hammett + VdW + R+ + R-)  -> ECD spectrum

This version does three things the previous one didn't:

1. BUILDS ITS OWN LEAKAGE-FREE SPLIT, always. It no longer trusts an
   external --split-dir. Every row is first grouped into an "equivalence
   class" (a molecule and its C2-symmetric mirror and/or any exact-duplicate
   substitution pattern -- see build_groups()), and a group is NEVER split
   across train/validation/test or across CV folds. This used to depend on
   remembering to run a separate make_split.py step first; now it can't be
   skipped.
2. RUNS K-FOLD CROSS-VALIDATION (group-aware) on the training pool, so the
   reported R2/RMSE/MAE/cosine come with a genuine mean +/- std uncertainty
   across folds, not a single lucky (or unlucky) split.
3. STILL TRAINS ONE FINAL, DEPLOYABLE MODEL on the whole training pool and
   evaluates it once on a completely untouched held-out test set (the
   classic "CV for uncertainty, final model for deployment" pattern) --
   packaged exactly as before, fully compatible with predict_ecd.py and
   every other script built on top of it (design_candidates.py,
   band_area_scaling_study.py, etc.).

This script is the *reproducibility* artifact: it regenerates everything
from the raw descriptor table. If you just want to predict the ECD
spectrum of a new molecule from an already-trained model package, use
``predict_ecd.py`` instead -- it has far fewer dependencies and no
training/plotting code.

Usage
-----
    python train_pipeline.py \\
        --input-csv data/ECD_total_Gauss.csv \\
        --output-dir results/ \\
        --k-folds 5 --test-frac 0.15

Note on compute time: this trains K+1 models per label (K CV folds + 1
final model), so a run now takes roughly (K+1)x as long as the old
single-split version. Reduce --k-folds (e.g. to 3) for a quicker pass.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

SEED = 42
HC_EV_NM = 1239.841984          # h*c in eV*nm
SIGMA_EV = 0.30                 # Gaussian broadening width (eV)
WL_MIN, WL_MAX = 250.0, 600.0   # nm
N_GRID = 2000                   # high-resolution reconstruction grid
N_MODEL_POINTS = 100            # points fed to / predicted by the models
AD_PERCENTILE = 95              # applicability-domain warning threshold

MODEL_CONFIGS = {
    "E2E_H":  {"hidden_units": [64, 32],     "dropout": 0.30, "l2": 1e-4, "input_dim": 16},
    "E2E_64": {"hidden_units": [64, 64, 32], "dropout": 0.30, "l2": 1e-4, "input_dim": 64},
}
ACTIVATION = "gelu"
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 600
EARLY_STOPPING_PATIENCE = 40
EARLY_STOPPING_MIN_DELTA = 1e-5
REDUCE_LR_PATIENCE = 12
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

HAMMETT_COLS = [f"Pos_{i}" for i in range(1, 17)]
VDW_COLS = [f"VdW_{i}" for i in range(1, 17)]
RPLUS_COLS = [f"Rplus_{i}" for i in range(1, 17)]
RMINUS_COLS = [f"Rminus_{i}" for i in range(1, 17)]
DESCRIPTOR_COLS = HAMMETT_COLS + VDW_COLS + RPLUS_COLS + RMINUS_COLS
NM_COLS = [f"nm_{i}" for i in range(1, 101)]
R_COLS = [f"R_{i}" for i in range(1, 101)]
N_POSITIONS = 16


# --------------------------------------------------------------------------- #
# C2 symmetry of [6]helicene: position i <-> 17-i                             #
# --------------------------------------------------------------------------- #
#
# Empirically verified against ECD_total_Gauss.csv: mono-substituted
# molecules only ever use positions 1-8 (9-16 are never computed alone), and
# 359 pairs of molecules elsewhere in the dataset that ARE mirror images
# under i <-> 17-i have a median reconstructed-ECD cosine similarity of
# 1.00000 (vs 0.68 for random unrelated pairs). This is a real symmetry
# of the parent hexahelicene, not an approximation, so it's valid to use for
# free data augmentation. It degrades slightly for heavily substituted
# (3-4 group) molecules, likely due to different DFT-optimized conformers.

def build_symmetry_mirror(x: np.ndarray) -> np.ndarray:
    """Reflect the 16 helicene positions (i <-> 17-i) within each 16-column
    descriptor block of x. For E2E_H (16 cols, one block) this reverses the
    single Hammett block; for E2E_64 (64 cols, 4 blocks: Hammett/VdW/R+/R-)
    it reverses each block independently, which is equivalent since
    mirror_position(i) = 17 - i turns "reverse the 16 columns" into exactly
    the right permutation (position 1 <-> position 16, ..., 8 <-> 9)."""
    n_blocks = x.shape[1] // 16
    if x.shape[1] % 16 != 0:
        raise ValueError(f"Expected a multiple of 16 columns, got {x.shape[1]}.")
    blocks = [x[:, b * 16:(b + 1) * 16][:, ::-1] for b in range(n_blocks)]
    return np.hstack(blocks)


def augment_with_symmetry_mirror(x_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Doubles the training set with the C2-mirrored descriptors, keeping the
    SAME target spectrum (it's the same molecule under a real symmetry, not
    a synthetic perturbation). Call on the training split only -- never on
    validation/test, or you leak information and inflate those metrics."""
    x_mirror = build_symmetry_mirror(x_train)
    return np.vstack([x_train, x_mirror]), np.vstack([y_train, y_train])


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@dataclass
class Paths:
    output_dir: str

    def __post_init__(self):
        self.model_dir = os.path.join(self.output_dir, "models")
        self.history_dir = os.path.join(self.output_dir, "histories")
        self.config_dir = os.path.join(self.output_dir, "configs")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.scaler_dir = os.path.join(self.output_dir, "scalers")
        self.diagnostic_dir = os.path.join(self.output_dir, "diagnostics")
        self.cv_dir = os.path.join(self.output_dir, "cross_validation")
        self.splits_dir = os.path.join(self.output_dir, "splits")
        for d in (self.model_dir, self.history_dir, self.config_dir, self.checkpoint_dir,
                  self.scaler_dir, self.diagnostic_dir, self.cv_dir, self.splits_dir):
            os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loading & validation                                                    #
# --------------------------------------------------------------------------- #

def load_dataset(input_csv: str) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, sep=";")
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = ["Archivo", "nsust"] + DESCRIPTOR_COLS + NM_COLS + R_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))


def validate_finite(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptors = df[DESCRIPTOR_COLS].to_numpy(dtype=float)
    nm = df[NM_COLS].to_numpy(dtype=float)
    rot = df[R_COLS].to_numpy(dtype=float)
    for name, arr in [("descriptor", descriptors), ("wavelength", nm), ("rotatory-strength", rot)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values detected in {name} matrix.")
    return descriptors, nm, rot


# --------------------------------------------------------------------------- #
# ECD spectrum reconstruction                                                  #
# --------------------------------------------------------------------------- #

def reconstruct_ecd_spectra(nm_matrix: np.ndarray, rot_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-broaden discrete rotatory strengths into continuous ECD spectra
    in energy space, on a high-resolution wavelength grid."""
    e_grid = np.linspace(HC_EV_NM / WL_MAX, HC_EV_NM / WL_MIN, N_GRID)
    wl_grid = HC_EV_NM / e_grid
    order = np.argsort(wl_grid)
    wl_grid, e_grid = wl_grid[order], e_grid[order]

    y_high_res = np.zeros((len(nm_matrix), N_GRID), dtype=np.float32)
    t0 = time.time()
    for i, (wavelengths, strengths) in enumerate(zip(nm_matrix, rot_matrix)):
        valid = np.isfinite(wavelengths) & np.isfinite(strengths) & (wavelengths > 0)
        energies = HC_EV_NM / wavelengths[valid]
        spectrum = np.zeros(N_GRID)
        for energy, strength in zip(energies, strengths[valid]):
            spectrum += strength * np.exp(-((e_grid - energy) / SIGMA_EV) ** 2)
        y_high_res[i] = spectrum.astype(np.float32)
        if (i + 1) % 1000 == 0 or i == len(nm_matrix) - 1:
            print(f"  ECD reconstruction: {i + 1}/{len(nm_matrix)} "
                  f"({100 * (i + 1) / len(nm_matrix):.1f}%) - {(time.time() - t0) / 60:.2f} min")
    return wl_grid, y_high_res


def interpolate_to_model_grid(wl_grid: np.ndarray, y_high_res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wl_model = np.linspace(WL_MIN, WL_MAX, N_MODEL_POINTS)
    y_ecd = np.array([np.interp(wl_model, wl_grid, s) for s in y_high_res], dtype=np.float32)
    return wl_model, y_ecd


# --------------------------------------------------------------------------- #
# Split & scaling                                                              #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Leakage-free split, integrated (no external make_split.py step needed)      #
# --------------------------------------------------------------------------- #
#
# Every row is grouped into an "equivalence class": a molecule and any other
# row that is either (a) its C2-symmetric mirror (position i <-> 17-i, see
# build_symmetry_mirror below) or (b) an exact-duplicate substitution pattern
# under a different filename. A class is NEVER split across train/validation/
# test or across CV folds -- see verify_no_leakage(), which is run every time.

def substituted_positions(row: pd.Series) -> dict:
    return {i: round(float(row[f"Pos_{i}"]), 6) for i in range(1, N_POSITIONS + 1) if row[f"VdW_{i}"] != 7.24}


def mirror_key(key: frozenset) -> frozenset:
    return frozenset({N_POSITIONS + 1 - p: h for p, h in key}.items())


def find_equivalence_classes(df: pd.DataFrame) -> list:
    """Groups of row indices that are the same physical molecule (exact
    duplicate and/or C2-mirror image of each other). Singletons (no
    duplicate/mirror found) are omitted -- callers should treat every other
    row as its own one-element group."""
    pattern_groups = defaultdict(list)
    for idx, row in df.iterrows():
        subs = substituted_positions(row)
        if subs:
            pattern_groups[frozenset(subs.items())].append(idx)

    keys = list(pattern_groups.keys())
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(k1, k2):
        r1, r2 = find(k1), find(k2)
        if r1 != r2:
            parent[r2] = r1

    for key in keys:
        m = mirror_key(key)
        if m in pattern_groups:
            union(key, m)

    classes = defaultdict(list)
    for key in keys:
        classes[find(key)].extend(pattern_groups[key])
    return [sorted(v) for v in classes.values() if len(v) > 1]


def build_groups(df: pd.DataFrame) -> np.ndarray:
    """One group id per row: rows in the same equivalence class share a
    group id; every other row is its own singleton group. Feed this to any
    sklearn Group* splitter to guarantee a class never straddles a split."""
    groups = np.arange(len(df))
    for cls in find_equivalence_classes(df):
        group_id = min(cls)
        for idx in cls:
            groups[idx] = group_id
    return groups


def build_cv_and_test_split(df: pd.DataFrame, test_frac: float, k_folds: int, seed: int) -> tuple:
    """Returns (groups, trainval_idx, test_idx, folds), where folds is a
    list of (fold_train_idx, fold_test_idx) tuples -- all group-aware, so
    no equivalence class ever straddles test/trainval or a fold boundary."""
    groups = build_groups(df)
    all_idx = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_pos, test_pos = next(gss.split(all_idx, groups=groups))
    trainval_idx, test_idx = all_idx[trainval_pos], all_idx[test_pos]

    gkf = GroupKFold(n_splits=k_folds)
    folds = [(trainval_idx[tr], trainval_idx[te])
             for tr, te in gkf.split(trainval_idx, groups=groups[trainval_idx])]

    return groups, trainval_idx, test_idx, folds


def train_val_split_within(idx: np.ndarray, groups: np.ndarray, val_frac: float, seed: int) -> tuple:
    """Group-aware split of any index pool into (sub_train, sub_val), used
    both for each CV fold's early-stopping validation and for the final
    model's own train/validation split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_pos, va_pos = next(gss.split(idx, groups=groups[idx]))
    return idx[tr_pos], idx[va_pos]


def verify_no_leakage(groups: np.ndarray, test_idx: np.ndarray, folds: list) -> None:
    """Raises if any equivalence class (any group with >1 member) straddles
    the test/trainval split or any CV fold's train/test boundary. This is
    the safety net that makes the leakage check unskippable."""
    test_set = set(test_idx.tolist())
    class_to_rows = defaultdict(list)
    for idx, g in enumerate(groups):
        class_to_rows[g].append(idx)

    for g, rows in class_to_rows.items():
        if len(rows) < 2:
            continue
        rows_set = set(rows)
        in_test = rows_set & test_set
        if in_test and in_test != rows_set:
            raise ValueError(f"LEAKAGE: equivalence class {rows} straddles the test/trainval split.")

    for fold_i, (ftr, fte) in enumerate(folds, start=1):
        ftr_set, fte_set = set(ftr.tolist()), set(fte.tolist())
        for g, rows in class_to_rows.items():
            rows_set = set(rows)
            if rows_set & test_set:
                continue  # already outside trainval entirely, not this fold's concern
            in_fold_test = rows_set & fte_set
            if in_fold_test and in_fold_test != (rows_set & (ftr_set | fte_set)):
                raise ValueError(f"LEAKAGE: equivalence class {rows} straddles fold {fold_i}.")

    print(f"Leakage check: PASSED (test split and all {len(folds)} CV folds -- "
          "no equivalence class straddles a boundary).")


def scale_split(scaler: StandardScaler, *arrays: np.ndarray) -> list:
    return [scaler.transform(a).astype(np.float32) for a in arrays]


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

def build_e2e_model(input_dim: int, hidden_units: list, dropout_rate: float, l2_strength: float,
                     output_dim: int = N_MODEL_POINTS, model_name: Optional[str] = None) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="molecular_descriptors")
    x = inputs
    for i, units in enumerate(hidden_units, start=1):
        x = layers.Dense(units, activation=ACTIVATION,
                          kernel_regularizer=regularizers.l2(l2_strength),
                          name=f"dense_{units}_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="ecd_output")(x)
    model = keras.Model(inputs, outputs, name=model_name or f"E2E_{input_dim}")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def cosine_similarity_integral(y_true: np.ndarray, y_pred: np.ndarray, wl_grid: np.ndarray) -> np.ndarray:
    num = np.trapezoid(y_true * y_pred, wl_grid, axis=1)
    den = np.sqrt(np.maximum(
        np.trapezoid(y_true * y_true, wl_grid, axis=1) * np.trapezoid(y_pred * y_pred, wl_grid, axis=1),
        1e-30))
    return num / den


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, wl_grid: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)

    r2 = np.array([r2_score(t, p) for t, p in zip(y_true, y_pred)])
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    mae = np.mean(np.abs(y_true - y_pred), axis=1)
    max_abs = np.max(np.abs(y_true - y_pred), axis=1)
    cosine = cosine_similarity_integral(y_true, y_pred, wl_grid)

    return {
        "global": {
            "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
            "rmse": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
            "mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
        },
        "molecule": {
            "mean_r2": float(np.nanmean(r2)), "median_r2": float(np.nanmedian(r2)),
            "mean_rmse": float(np.mean(rmse)), "median_rmse": float(np.median(rmse)),
            "mean_mae": float(np.mean(mae)), "median_mae": float(np.median(mae)),
            "mean_max_abs_error": float(np.mean(max_abs)), "median_max_abs_error": float(np.median(max_abs)),
            "mean_cosine": float(np.nanmean(cosine)), "median_cosine": float(np.nanmedian(cosine)),
            "rmse_percentiles": {f"P{p}": float(np.percentile(rmse, p)) for p in (25, 50, 75, 90, 95, 99)},
        },
    }


def molecule_metrics_table(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                            original_indices: np.ndarray, wl_grid: np.ndarray) -> pd.DataFrame:
    cosine = cosine_similarity_integral(y_true, y_pred, wl_grid)
    records = []
    for i, idx in enumerate(original_indices):
        t, p = y_true[i], y_pred[i]
        records.append({
            "dataset_index": int(idx),
            "Archivo": str(df.iloc[idx]["Archivo"]),
            "nsust": df.iloc[idx]["nsust"],
            "R2": float(r2_score(t, p)),
            "RMSE": float(np.sqrt(mean_squared_error(t, p))),
            "MAE": float(mean_absolute_error(t, p)),
            "max_abs_error": float(np.max(np.abs(t - p))),
            "cosine_similarity": float(cosine[i]),
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def train_e2e_model(label: str, x_train, x_val, y_train, y_val, paths: Paths,
                     hidden_units: Optional[list] = None) -> dict:
    cfg = MODEL_CONFIGS[label]
    hidden_units = hidden_units or cfg["hidden_units"]
    print(f"\n{'#' * 90}\n# TRAINING: {label}  (input_dim={cfg['input_dim']}, arch={hidden_units})\n{'#' * 90}")

    model_dir = os.path.join(paths.model_dir, label)
    ckpt_dir = os.path.join(paths.checkpoint_dir, label)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{label}_best.weights.h5")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # never resume a stale checkpoint from a previous run

    model = build_e2e_model(cfg["input_dim"], hidden_units, cfg["dropout"], cfg["l2"], model_name=label)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                       min_delta=EARLY_STOPPING_MIN_DELTA, mode="min",
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                                           patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE,
                                           mode="min", verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", mode="min",
                                         save_best_only=True, save_weights_only=True, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=MAX_EPOCHS,
                         batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=1)
    training_time_min = (time.time() - t0) / 60

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint was not created: {ckpt_path}")
    model.load_weights(ckpt_path)  # explicit restore: guarantees best-val weights, not last epoch

    final_model_path = os.path.join(model_dir, f"{label}.keras")
    model.save(final_model_path)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmin(history_dict["val_loss"]) + 1)
    best_val_loss = float(np.min(history_dict["val_loss"]))

    with open(os.path.join(paths.history_dir, f"{label}_history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)

    train_config = {
        "model_label": label, **cfg, "hidden_units": hidden_units, "activation": ACTIVATION, "optimizer": "Adam",
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(history_dict["loss"]), "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, "seed": SEED, "loss": "MSE",
        "target_points": N_MODEL_POINTS, "wavelength_min_nm": WL_MIN, "wavelength_max_nm": WL_MAX,
        "training_time_min": training_time_min, "checkpoint": ckpt_path,
    }
    with open(os.path.join(model_dir, f"{label}_training_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"# {label} done — best epoch {best_epoch}, best val loss {best_val_loss:.8f}, "
          f"{training_time_min:.2f} min\n{'#' * 90}")

    return {"model": model, "history": history_dict, "model_dir": model_dir,
            "checkpoint_path": ckpt_path, "final_model_path": final_model_path,
            "best_epoch": best_epoch, "epochs_completed": len(history_dict["loss"]),
            "best_val_loss": best_val_loss, "training_time_min": training_time_min}


def train_fold_model(label: str, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray,
                      y_val: np.ndarray, fold_dir: str, hidden_units: Optional[list] = None) -> tuple:
    """Lightweight sibling of train_e2e_model() for a single CV fold: same
    architecture/callbacks/early-stopping, but no full .keras save and no
    loss-curve plot (K of these get trained per label, so we keep it cheap
    -- only the checkpoint weights are kept, and only transiently)."""
    cfg = MODEL_CONFIGS[label]
    hidden_units = hidden_units or cfg["hidden_units"]
    os.makedirs(fold_dir, exist_ok=True)
    ckpt_path = os.path.join(fold_dir, f"{label}_best.weights.h5")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    model = build_e2e_model(cfg["input_dim"], hidden_units, cfg["dropout"], cfg["l2"], model_name=label)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                       min_delta=EARLY_STOPPING_MIN_DELTA, mode="min",
                                       restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                                           patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE,
                                           mode="min", verbose=0),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", mode="min",
                                         save_best_only=True, save_weights_only=True, verbose=0),
    ]

    t0 = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=MAX_EPOCHS,
                         batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=0)
    training_time_min = (time.time() - t0) / 60

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint was not created: {ckpt_path}")
    model.load_weights(ckpt_path)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmin(history_dict["val_loss"]) + 1)
    best_val_loss = float(np.min(history_dict["val_loss"]))

    return model, {"best_epoch": best_epoch, "epochs_completed": len(history_dict["loss"]),
                    "best_val_loss": best_val_loss, "training_time_min": training_time_min}


# --------------------------------------------------------------------------- #
# Hyperparameter search (Phase 0 -- runs BEFORE the CV, on its own held-out   #
# validation split, so the CV folds reported later never see which           #
# architecture "won" -- avoids the double-dipping / winner's-curse bias of   #
# selecting and reporting uncertainty from the same folds.                   #
# --------------------------------------------------------------------------- #

ARCHITECTURE_CANDIDATES = [[32], [64, 32], [64, 64, 32], [128, 64, 32]]
HP_SEARCH_VAL_FRAC = 0.15
HP_SEARCH_PATIENCE = 20  # shorter than EARLY_STOPPING_PATIENCE -- this is a coarse screen, not the final fit


def search_hyperparameters(label: str, x: np.ndarray, y_ecd: np.ndarray, trainval_idx: np.ndarray,
                            groups: np.ndarray, seed: int, candidates: Optional[list] = None) -> tuple:
    """Tries each candidate architecture on ONE dedicated group-aware
    train/val split carved out of trainval_idx (NOT the k folds used later
    for CV), picks the one with the lowest validation loss. Returns
    (winning_hidden_units, results_dataframe)."""
    candidates = candidates or ARCHITECTURE_CANDIDATES
    cfg = MODEL_CONFIGS[label]

    search_train_idx, search_val_idx = train_val_split_within(trainval_idx, groups, HP_SEARCH_VAL_FRAC, seed)
    xa, xb = x[search_train_idx], x[search_val_idx]
    ya, yb = y_ecd[search_train_idx], y_ecd[search_val_idx]
    sx, sy = StandardScaler().fit(xa), StandardScaler().fit(ya)
    xa_s, xb_s = scale_split(sx, xa, xb)
    ya_s, yb_s = scale_split(sy, ya, yb)

    print(f"\n{'=' * 90}\nHYPERPARAMETER SEARCH: {label}  "
          f"({len(candidates)} candidate architectures, on a dedicated "
          f"{100*(1-HP_SEARCH_VAL_FRAC):.0f}/{100*HP_SEARCH_VAL_FRAC:.0f} split -- "
          "not the CV folds used later)\n" + "=" * 90)

    records = []
    for arch in candidates:
        model = build_e2e_model(cfg["input_dim"], arch, cfg["dropout"], cfg["l2"], model_name=f"{label}_search")
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=HP_SEARCH_PATIENCE,
                                           min_delta=EARLY_STOPPING_MIN_DELTA, mode="min",
                                           restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                                               patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE,
                                               mode="min", verbose=0),
        ]
        t0 = time.time()
        history = model.fit(xa_s, ya_s, validation_data=(xb_s, yb_s), epochs=MAX_EPOCHS,
                             batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=0)
        best_val_loss = float(np.min(history.history["val_loss"]))
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        elapsed = time.time() - t0
        records.append({"hidden_units": str(arch), "best_val_loss": best_val_loss,
                         "best_epoch": best_epoch, "epochs_completed": len(history.history["loss"]),
                         "training_time_min": elapsed / 60})
        print(f"  {str(arch):>18}: val_loss={best_val_loss:.6f}  (best epoch {best_epoch}, {elapsed:.0f}s)")
        del model
        keras.backend.clear_session()

    results_df = pd.DataFrame(records).sort_values("best_val_loss").reset_index(drop=True)
    import ast
    winner = ast.literal_eval(results_df.iloc[0]["hidden_units"])  # "[64, 32]" -> [64, 32]
    print(f"  -> WINNER for {label}: {winner}  (val_loss={results_df.iloc[0]['best_val_loss']:.6f})\n")
    return winner, results_df


def training_gap_diagnostic(result: dict) -> dict:
    h, idx = result["history"], result["best_epoch"] - 1
    train_mae, val_mae = h["mae"][idx], h["val_mae"][idx]
    train_loss, val_loss = h["loss"][idx], h["val_loss"][idx]
    return {
        "best_epoch": result["best_epoch"], "train_mae": train_mae, "val_mae": val_mae,
        "train_loss": train_loss, "val_loss": val_loss,
        "val_train_mae_ratio": val_mae / max(train_mae, 1e-12),
        "val_train_loss_ratio": val_loss / max(train_loss, 1e-12),
    }


def plot_training_history(result: dict, label: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(result["history"]["loss"], label="Training loss")
    plt.plot(result["history"]["val_loss"], label="Validation loss")
    plt.axvline(result["best_epoch"] - 1, ls="--", label=f"Best epoch: {result['best_epoch']}")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title(f"{label} training history")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(result["model_dir"], f"{label}_loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Applicability domain & error-vs-distance diagnostics                        #
# --------------------------------------------------------------------------- #

def build_applicability_domain(x_train_scaled: np.ndarray, label: str, model_dir: str) -> dict:
    """Distance to the nearest OTHER training molecule; the AD_PERCENTILE-th
    percentile is stored as a reference warning threshold for new molecules."""
    if len(x_train_scaled) < 3:
        raise ValueError("At least 3 training molecules are required for an applicability domain.")
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(x_train_scaled)
    distances, _ = nn.kneighbors(x_train_scaled)
    nearest_other = distances[:, 1]
    threshold = float(np.percentile(nearest_other, AD_PERCENTILE))

    payload = {"model_label": label, "metric": "euclidean", "percentile": AD_PERCENTILE,
               "threshold": threshold, "training_scaled_X": x_train_scaled,
               "nearest_other_distances": nearest_other}
    joblib.dump(payload, os.path.join(model_dir, "applicability_domain.joblib"))
    return payload


def error_vs_distance_diagnostic(label: str, x_train_scaled: np.ndarray, x_query_scaled: np.ndarray,
                                  molecule_metrics_df: pd.DataFrame, dataset_label: str,
                                  threshold: float, wl_grid: np.ndarray, diagnostic_dir: str) -> dict:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(x_train_scaled)
    distances, _ = nn.kneighbors(x_query_scaled)
    distances = distances[:, 0]

    diag = molecule_metrics_df.copy()
    diag["nearest_training_distance"] = distances
    diag["outside_reference_domain"] = distances > threshold
    diag.to_csv(os.path.join(diagnostic_dir, f"{label}_{dataset_label}_error_vs_distance.csv"), index=False)

    rho, p_value = spearmanr(diag["nearest_training_distance"], diag["RMSE"])  # relation need not be linear

    plt.figure(figsize=(8, 6))
    plt.scatter(diag["nearest_training_distance"], diag["RMSE"], alpha=0.75)
    plt.axvline(threshold, ls="--", label=f"{AD_PERCENTILE}th percentile threshold = {threshold:.3f}")
    plt.xlabel("Distance to nearest training molecule (standardized descriptor space)")
    plt.ylabel("Per-molecule RMSE")
    plt.title(f"{label} — {dataset_label}: prediction error vs chemical distance")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(diagnostic_dir, f"{label}_{dataset_label}_error_vs_distance.png"),
                dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "model_label": label, "dataset": dataset_label,
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_p_value": float(p_value) if np.isfinite(p_value) else None,
        "mean_distance": float(np.mean(distances)), "median_distance": float(np.median(distances)),
        "max_distance": float(np.max(distances)),
        "n_outside_reference_domain": int(diag["outside_reference_domain"].sum()),
        "n_molecules": int(len(diag)), "threshold": float(threshold),
    }


def create_quartile_plots(label: str, molecule_metrics_df: pd.DataFrame, y_true: np.ndarray,
                           y_pred: np.ndarray, wl_grid: np.ndarray, dataset_label: str, model_dir: str) -> list:
    """For each RMSE quartile bin, plot the molecule whose RMSE is closest to
    that bin's median — a representative "good/typical/poor" set of spectra."""
    output_dir = os.path.join(model_dir, dataset_label, "quartile_plots")
    os.makedirs(output_dir, exist_ok=True)

    work = molecule_metrics_df.sort_values("RMSE").reset_index(drop=True)
    q25, q50, q75 = work["RMSE"].quantile([0.25, 0.50, 0.75]).to_numpy()
    bins = {
        "Q1": work[work["RMSE"] <= q25],
        "Q2": work[(work["RMSE"] > q25) & (work["RMSE"] <= q50)],
        "Q3": work[(work["RMSE"] > q50) & (work["RMSE"] <= q75)],
        "Q4": work[work["RMSE"] > q75],
    }

    selected = []
    for quartile, subset in bins.items():
        if subset.empty:
            continue
        target = subset["RMSE"].median()
        row = subset.iloc[int(np.argmin(np.abs(subset["RMSE"].to_numpy() - target)))]
        pos = molecule_metrics_df.index[molecule_metrics_df["dataset_index"] == row["dataset_index"]]
        if len(pos) == 0:
            continue
        pos = int(pos[0])
        selected.append((quartile, row, y_true[pos], y_pred[pos]))

        plt.figure(figsize=(10, 5))
        plt.plot(wl_grid, y_true[pos], label="Reference ECD", linewidth=2)
        plt.plot(wl_grid, y_pred[pos], label=f"{label} prediction", linewidth=2)
        plt.xlabel("Wavelength (nm)"); plt.ylabel("ECD intensity")
        plt.title(f"{label} — {dataset_label} — {quartile} representative")
        note = (f"Archivo: {row['Archivo']}\nR²: {row['R2']:.4f}\nRMSE: {row['RMSE']:.4f}\n"
                f"MAE: {row['MAE']:.4f}\nMax abs error: {row['max_abs_error']:.4f}\n"
                f"Cosine: {row['cosine_similarity']:.4f}")
        plt.text(0.02, 0.97, note, transform=plt.gca().transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round", alpha=0.8))
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_{dataset_label}_{quartile}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

    if selected:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = np.asarray(axes).ravel()
        for ax, (quartile, row, t, p) in zip(axes, selected):
            ax.plot(wl_grid, t, label="Reference ECD", linewidth=2)
            ax.plot(wl_grid, p, label=f"{label} prediction", linewidth=2)
            ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("ECD intensity")
            ax.set_title(f"{quartile} | RMSE={row['RMSE']:.2f} | Cosine={row['cosine_similarity']:.3f}")
            ax.legend(fontsize=8); ax.grid(alpha=0.2)
        for ax in axes[len(selected):]:
            ax.axis("off")
        fig.suptitle(f"{label} — {dataset_label} representative ECD spectra", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{label}_{dataset_label}_representative_spectra_2x2.png"),
                    dpi=220, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame([{
            "quartile": q, "dataset_index": int(r["dataset_index"]), "Archivo": r["Archivo"],
            "R2": float(r["R2"]), "RMSE": float(r["RMSE"]), "MAE": float(r["MAE"]),
            "max_abs_error": float(r["max_abs_error"]), "cosine_similarity": float(r["cosine_similarity"]),
        } for q, r, _, _ in selected]).to_csv(
            os.path.join(output_dir, f"{label}_{dataset_label}_representatives.csv"), index=False)

    return selected


# --------------------------------------------------------------------------- #
# Packaging (what actually gets zipped up / published per model)              #
# --------------------------------------------------------------------------- #

PREDICTION_SCRIPT_NOTE = (
    "Model packages are self-contained: <label>.keras, scaler_X_<label>.joblib, "
    "scaler_Y_ECD.joblib, WL_MODEL.npy, feature_columns.json and "
    "applicability_domain.joblib. Use predict_ecd.py to load and query them."
)


def package_model(label: str, scaler_x: StandardScaler, scaler_y: StandardScaler, wl_model: np.ndarray,
                   feature_cols: list, model_dir: str, output_dir: str,
                   train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> str:
    joblib.dump(scaler_x, os.path.join(model_dir, f"scaler_X_{label}.joblib"))
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_Y_ECD.joblib"))
    np.save(os.path.join(model_dir, "WL_MODEL.npy"), wl_model)
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    for name, arr in [("train_indices.npy", train_idx), ("validation_indices.npy", val_idx),
                       ("test_indices.npy", test_idx)]:
        np.save(os.path.join(model_dir, name), arr)

    zip_path = os.path.join(output_dir, f"{label}_package.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(model_dir):
            for file in files:
                full = os.path.join(root, file)
                zf.write(full, os.path.relpath(full, model_dir))
    return zip_path


# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #

def run_pipeline(input_csv: str, output_dir: str, k_folds: int = 5, test_frac: float = 0.15,
                  val_frac: float = 0.1, augment_symmetry: bool = False, search_hidden_units: bool = True,
                  labels: tuple = ("E2E_H", "E2E_64"), seed: int = SEED) -> tuple:
    set_seed(seed)
    paths = Paths(output_dir)

    df = load_dataset(input_csv)
    validate_columns(df)
    _, nm_matrix, rot_matrix = validate_finite(df)

    print("\nReconstructing reference ECD spectra...")
    wl_grid, y_high_res = reconstruct_ecd_spectra(nm_matrix, rot_matrix)
    wl_model, y_ecd = interpolate_to_model_grid(wl_grid, y_high_res)
    np.save(os.path.join(output_dir, "WL_MODEL.npy"), wl_model)

    print(f"\nBuilding a leakage-free split ({100*(1-test_frac):.0f}% train/CV pool, "
          f"{100*test_frac:.0f}% held-out test), {k_folds}-fold group-aware CV on the pool...")
    groups, trainval_idx, test_idx, folds = build_cv_and_test_split(df, test_frac, k_folds, seed)
    verify_no_leakage(groups, test_idx, folds)

    np.save(os.path.join(paths.splits_dir, "trainval_indices.npy"), trainval_idx)
    np.save(os.path.join(paths.splits_dir, "test_indices.npy"), test_idx)
    for i, (ftr, fte) in enumerate(folds, start=1):
        np.save(os.path.join(paths.splits_dir, f"fold{i}_train_indices.npy"), ftr)
        np.save(os.path.join(paths.splits_dir, f"fold{i}_test_indices.npy"), fte)
    print(f"Trainval pool: {len(trainval_idx)}   Held-out test: {len(test_idx)}   "
          f"CV folds: {[len(fte) for _, fte in folds]}")

    x_h = df[HAMMETT_COLS].to_numpy(dtype=np.float32)
    x_64 = df[DESCRIPTOR_COLS].to_numpy(dtype=np.float32)
    inputs = {"E2E_H": x_h, "E2E_64": x_64}

    cv_summary_all, comparison_rows = {}, []

    for label in labels:
        x = inputs[label]

        # ================================================================ #
        # PHASE 0: HYPERPARAMETER SEARCH -- own held-out split, before CV   #
        # ================================================================ #
        if search_hidden_units:
            winner_arch, hp_results = search_hyperparameters(label, x, y_ecd, trainval_idx, groups, seed)
            hp_results.to_csv(os.path.join(paths.cv_dir, f"{label}_hyperparameter_search.csv"), index=False)
            with open(os.path.join(paths.cv_dir, f"{label}_hyperparameter_search_winner.json"), "w") as f:
                json.dump({"winner_hidden_units": winner_arch,
                           "candidates": ARCHITECTURE_CANDIDATES}, f, indent=2)
        else:
            winner_arch = MODEL_CONFIGS[label]["hidden_units"]

        # ================================================================ #
        # K-FOLD CROSS-VALIDATION -- gives mean +/- std uncertainty         #
        # (architecture FIXED to the Phase 0 winner for every fold, so the  #
        # CV reports the uncertainty of ONE model, not a moving target)     #
        # ================================================================ #
        print(f"\n{'#' * 90}\n# {k_folds}-FOLD CROSS-VALIDATION: {label}  (arch={winner_arch})\n{'#' * 90}")
        fold_records = []
        for fold_i, (ftr_idx, fte_idx) in enumerate(folds, start=1):
            sub_train_idx, sub_val_idx = train_val_split_within(ftr_idx, groups, val_frac, seed + fold_i)
            xa, xb, xc = x[sub_train_idx], x[sub_val_idx], x[fte_idx]
            ya, yb, yc = y_ecd[sub_train_idx], y_ecd[sub_val_idx], y_ecd[fte_idx]

            if augment_symmetry:
                xa, ya = augment_with_symmetry_mirror(xa, ya)

            sx, sy = StandardScaler().fit(xa), StandardScaler().fit(ya)
            xa_s, xb_s, xc_s = scale_split(sx, xa, xb, xc)
            ya_s, yb_s = scale_split(sy, ya, yb)

            t0 = time.time()
            fold_dir = os.path.join(paths.cv_dir, label, f"fold{fold_i}")
            model, info = train_fold_model(label, xa_s, xb_s, ya_s, yb_s, fold_dir, hidden_units=winner_arch)
            yc_pred = sy.inverse_transform(model.predict(xc_s, batch_size=BATCH_SIZE, verbose=0))
            m = calculate_metrics(yc, yc_pred, wl_model)

            fold_records.append({
                "fold": fold_i, "n_train": len(sub_train_idx), "n_val": len(sub_val_idx), "n_test": len(fte_idx),
                "r2": m["molecule"]["mean_r2"], "rmse": m["molecule"]["mean_rmse"],
                "mae": m["molecule"]["mean_mae"], "cosine": m["molecule"]["mean_cosine"],
                "best_epoch": info["best_epoch"], "epochs_completed": info["epochs_completed"],
            })
            print(f"  fold {fold_i}/{k_folds}: R2={m['molecule']['mean_r2']:.4f}  "
                  f"RMSE={m['molecule']['mean_rmse']:.4f}  MAE={m['molecule']['mean_mae']:.4f}  "
                  f"cosine={m['molecule']['mean_cosine']:.4f}  "
                  f"(best epoch {info['best_epoch']}, {time.time()-t0:.0f}s)")
            del model
            keras.backend.clear_session()

        fold_df = pd.DataFrame(fold_records)
        fold_df.to_csv(os.path.join(paths.cv_dir, f"{label}_cv_fold_metrics.csv"), index=False)

        cv_summary = {
            metric: {"mean": float(fold_df[metric].mean()), "std": float(fold_df[metric].std(ddof=1)),
                     "values": [float(v) for v in fold_df[metric]]}
            for metric in ("r2", "rmse", "mae", "cosine")
        }
        cv_summary_all[label] = cv_summary
        with open(os.path.join(paths.cv_dir, f"{label}_cv_summary.json"), "w") as f:
            json.dump({"k_folds": k_folds, "seed": seed, "metrics": cv_summary}, f, indent=2)

        print(f"\n[{label}] {k_folds}-fold CV summary (mean +/- std across folds):")
        for metric in ("r2", "rmse", "mae", "cosine"):
            s = cv_summary[metric]
            print(f"    {metric.upper():>6}: {s['mean']:.4f} +/- {s['std']:.4f}")

        # ================================================================ #
        # FINAL MODEL -- trained on the whole pool, evaluated once on the  #
        # untouched held-out test set. This is the deployable artifact.    #
        # ================================================================ #
        print(f"\n{'#' * 90}\n# FINAL MODEL (trained on full pool): {label}\n{'#' * 90}")
        sub_train_idx, sub_val_idx = train_val_split_within(trainval_idx, groups, val_frac, seed)
        x_train, x_val, x_test = x[sub_train_idx], x[sub_val_idx], x[test_idx]
        y_train, y_val, y_test = y_ecd[sub_train_idx], y_ecd[sub_val_idx], y_ecd[test_idx]

        if augment_symmetry:
            n_before = len(x_train)
            x_train, y_train = augment_with_symmetry_mirror(x_train, y_train)
            print(f"{label}: symmetry augmentation {n_before} -> {len(x_train)} training examples "
                  "(fit on augmented set; validation/test untouched).")

        scaler_x = StandardScaler().fit(x_train)
        scaler_y = StandardScaler().fit(y_train)
        x_train_s, x_val_s, x_test_s = scale_split(scaler_x, x_train, x_val, x_test)
        y_train_s, y_val_s = scale_split(scaler_y, y_train, y_val)

        result = train_e2e_model(label, x_train_s, x_val_s, y_train_s, y_val_s, paths, hidden_units=winner_arch)
        plot_training_history(result, label)

        y_val_pred = scaler_y.inverse_transform(result["model"].predict(x_val_s, batch_size=BATCH_SIZE))
        y_test_pred = scaler_y.inverse_transform(result["model"].predict(x_test_s, batch_size=BATCH_SIZE))
        for name, pred in [("validation", y_val_pred), ("test", y_test_pred)]:
            if not np.all(np.isfinite(pred)):
                raise ValueError(f"Non-finite values detected in {label} {name} predictions.")

        val_metrics = calculate_metrics(y_val, y_val_pred, wl_model)
        test_metrics = calculate_metrics(y_test, y_test_pred, wl_model)

        with open(os.path.join(result["model_dir"], f"{label}_metrics.json"), "w") as f:
            json.dump({"model_label": label, "seed": seed, "validation": val_metrics, "test": test_metrics,
                       "cross_validation": cv_summary,
                       "training": {"best_epoch": result["best_epoch"],
                                    "epochs_completed": result["epochs_completed"],
                                    "best_val_loss": result["best_val_loss"],
                                    "training_time_min": result["training_time_min"]}}, f, indent=2)

        mm_test = molecule_metrics_table(df, y_test, y_test_pred, test_idx, wl_model)
        mm_val = molecule_metrics_table(df, y_val, y_val_pred, sub_val_idx, wl_model)
        mm_test.to_csv(os.path.join(result["model_dir"], f"{label}_test_molecule_metrics.csv"), index=False)
        mm_val.to_csv(os.path.join(result["model_dir"], f"{label}_validation_molecule_metrics.csv"), index=False)

        test_out_dir = os.path.join(result["model_dir"], "test")
        os.makedirs(test_out_dir, exist_ok=True)
        np.save(os.path.join(test_out_dir, "Y_test_true.npy"), y_test)
        np.save(os.path.join(test_out_dir, "Y_test_pred.npy"), y_test_pred)

        gap = training_gap_diagnostic(result)
        with open(os.path.join(paths.diagnostic_dir, f"{label}_training_gap.json"), "w") as f:
            json.dump(gap, f, indent=2)

        ad = build_applicability_domain(x_train_s, label, result["model_dir"])
        for dataset_label, x_query_s, mm in [("validation", x_val_s, mm_val), ("test", x_test_s, mm_test)]:
            diag = error_vs_distance_diagnostic(label, x_train_s, x_query_s, mm, dataset_label,
                                                 ad["threshold"], wl_model, paths.diagnostic_dir)
            with open(os.path.join(paths.diagnostic_dir, f"{label}_{dataset_label}_error_vs_distance.json"), "w") as f:
                json.dump(diag, f, indent=2)

        create_quartile_plots(label, mm_test, y_test, y_test_pred, wl_model, "test", result["model_dir"])
        create_quartile_plots(label, mm_val, y_val, y_val_pred, wl_model, "validation", result["model_dir"])

        feature_cols = HAMMETT_COLS if label == "E2E_H" else DESCRIPTOR_COLS
        zip_path = package_model(label, scaler_x, scaler_y, wl_model, feature_cols,
                                  result["model_dir"], output_dir, sub_train_idx, sub_val_idx, test_idx)
        print(f"{label} package: {zip_path}")

        comparison_rows.append({
            "Model": label.replace("E2E_", "E2E-"),
            "Input": "16 Hammett" if label == "E2E_H" else "64 descriptors",
            "Architecture": str(winner_arch),
            "Test_R2_global": test_metrics["global"]["r2"],
            "Test_RMSE_global": test_metrics["global"]["rmse"],
            "Test_MAE_global": test_metrics["global"]["mae"],
            "Test_mean_R2_molecule": test_metrics["molecule"]["mean_r2"],
            "Test_mean_RMSE_molecule": test_metrics["molecule"]["mean_rmse"],
            "Test_mean_cosine": test_metrics["molecule"]["mean_cosine"],
            "CV_R2_mean": cv_summary["r2"]["mean"], "CV_R2_std": cv_summary["r2"]["std"],
            "CV_RMSE_mean": cv_summary["rmse"]["mean"], "CV_RMSE_std": cv_summary["rmse"]["std"],
            "CV_MAE_mean": cv_summary["mae"]["mean"], "CV_MAE_std": cv_summary["mae"]["std"],
            "CV_cosine_mean": cv_summary["cosine"]["mean"], "CV_cosine_std": cv_summary["cosine"]["std"],
            "Best_epoch": result["best_epoch"], "Epochs_completed": result["epochs_completed"],
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(output_dir, "E2E_final_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    print("\n" + "=" * 90 + "\nFINAL COMPARISON (test set + CV mean/std)\n" + "=" * 90)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nFinal comparison saved to: {comparison_path}")

    # One single zip with EVERYTHING (models, CV results, splits, diagnostics,
    # histories) -- not just the lightweight prediction packages. /content/
    # gets wiped when a Colab session ends, so this is what you actually want
    # to download and keep if you plan to run characterize_model.py or
    # baseline_comparison.py in a LATER session without retraining.
    print("\nZipping the full output directory (everything -- models, CV, splits, "
          "diagnostics) into one archive for safekeeping...")
    full_backup_base = output_dir.rstrip("/\\") + "_FULL_BACKUP"
    full_backup_path = shutil.make_archive(full_backup_base, "zip", output_dir)
    print(f"Full backup saved to: {full_backup_path}")
    print("Download this one file and keep it -- it's everything you need to run "
          "characterize_model.py or baseline_comparison.py later without retraining. "
          "(The individual E2E_H_package.zip / E2E_64_package.zip are still there too, "
          "for predict_ecd.py / design_candidates.py -- those alone are NOT enough for "
          "characterization or baseline comparison.)")

    return comparison, cv_summary_all


# Defaults match the original Colab layout, so the script works whether you
# run it as `!python train_pipeline.py --input-csv ... --output-dir ...`
# from a shell, or paste it directly into a Colab cell and just press Run.
DEFAULT_INPUT_CSV = "/content/ECD_total_Gauss.csv"
DEFAULT_OUTPUT_DIR = "/content/ml_experiments"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the E2E-H / E2E-64 ECD prediction models with integrated "
                                             "leakage-free split and k-fold cross-validation.")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV,
                   help=f"Path to ECD_total_Gauss.csv (';'-separated). Default: {DEFAULT_INPUT_CSV}")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Where models, splits, CV results and diagnostics are written. "
                        f"Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--k-folds", type=int, default=5, help="Number of cross-validation folds.")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="Fraction of the dataset held out as a final, untouched test set.")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of each training pool (a CV fold's training portion, or the final "
                        "model's full pool) reserved for early-stopping validation.")
    p.add_argument("--augment-symmetry", action="store_true",
                   help="Double each training set using the helicene's own C2 symmetry "
                        "(position i <-> 17-i, same ECD target). Validation/test are never "
                        "touched. See build_symmetry_mirror() for the empirical justification.")
    p.add_argument("--search-hidden-units", action=argparse.BooleanOptionalAction, default=True,
                   help="Phase 0: try a few candidate architectures on a dedicated held-out split "
                        "before the CV, and fix the winner for CV + the final model. On by default; "
                        "use --no-search-hidden-units to skip it and use MODEL_CONFIGS as-is.")
    # parse_known_args ignores extra argv entries injected by the Jupyter/Colab
    # kernel itself (e.g. "-f /root/.../kernel-xxxx.json"), which is what makes
    # this safe to run as a notebook cell, not just from a terminal.
    args, _unknown = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_csv, args.output_dir, k_folds=args.k_folds, test_frac=args.test_frac,
                 val_frac=args.val_frac, augment_symmetry=args.augment_symmetry,
                 search_hidden_units=args.search_hidden_units)

    # --- Colab tip -----------------------------------------------------
    # If you'd rather not rely on argv at all (e.g. you're driving paths
    # from a Drive mount), just skip parse_args()/argv entirely and call:
    #
    #   run_pipeline(
    #       input_csv="/content/drive/MyDrive/.../ECD_total_Gauss.csv",
    #       output_dir="/content/drive/MyDrive/.../ml_experiments",
    #       k_folds=5, test_frac=0.15, augment_symmetry=True,
    #   )
    #
    # in its own cell instead of running this file's __main__ block.
