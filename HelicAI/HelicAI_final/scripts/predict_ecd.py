#!/usr/bin/env python3
"""
predict_ecd.py — Predict the ECD spectrum of a heliceno from a trained
E2E-H or E2E-64 model package.

This is the artifact meant for other researchers: it only needs a trained
model package (as produced by ``train_pipeline.py`` / ``package_model``)
and a vector of molecular descriptors. It does not need the original
dataset, the training split, or matplotlib.

Package layout expected (E2E_H_package.zip / E2E_64_package.zip, unzipped):
    <label>.keras                  trained Keras model
    scaler_X_<label>.joblib        StandardScaler fit on the descriptors
    scaler_Y_ECD.joblib            StandardScaler fit on the ECD spectra
    WL_MODEL.npy                   100-point wavelength grid (nm)
    feature_columns.json           ordered list of descriptor names
    applicability_domain.joblib    reference distances for the AD warning

Example (CLI)
-------------
    python predict_ecd.py \\
        --model-dir E2E_H_package/ \\
        --label E2E_H \\
        --descriptors 0.16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \\
        --plot prediction.png

Example (as a library)
-----------------------
    from predict_ecd import load_model_package, predict_ecd

    pkg = load_model_package("E2E_H_package/", "E2E_H")
    wavelengths, ecd, ad_info = predict_ecd(descriptors, pkg)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras

VALID_LABELS = ("E2E_H", "E2E_64")


@dataclass
class ModelPackage:
    label: str
    model: keras.Model
    scaler_x: object
    scaler_y: object
    wavelengths: np.ndarray
    feature_columns: list
    applicability_domain: dict


def load_model_package(model_dir: str, label: str) -> ModelPackage:
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}.")

    def p(name: str) -> str:
        return os.path.join(model_dir, name)

    model = keras.models.load_model(p(f"{label}.keras"))
    scaler_x = joblib.load(p(f"scaler_X_{label}.joblib"))
    scaler_y = joblib.load(p("scaler_Y_ECD.joblib"))
    wavelengths = np.load(p("WL_MODEL.npy"))
    with open(p("feature_columns.json")) as f:
        feature_columns = json.load(f)
    applicability_domain = joblib.load(p("applicability_domain.joblib"))

    return ModelPackage(label, model, scaler_x, scaler_y, wavelengths,
                         feature_columns, applicability_domain)


def _applicability_domain_check(descriptors_scaled: np.ndarray, ad: dict) -> dict:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(ad["training_scaled_X"])
    distance = float(nn.kneighbors(descriptors_scaled)[0][0, 0])
    threshold = float(ad["threshold"])
    return {
        "nearest_training_distance": distance,
        "threshold": threshold,
        "outside_reference_domain": distance > threshold,
        "percentile_reference": ad["percentile"],
    }


def predict_ecd(descriptors, package: ModelPackage) -> tuple:
    """Predict the ECD spectrum for one molecule.

    Parameters
    ----------
    descriptors : sequence of float
        Values in the exact order given by ``package.feature_columns``
        (16 values for E2E_H, 64 for E2E_64).

    Returns
    -------
    wavelengths : np.ndarray, shape (100,)
    ecd_spectrum : np.ndarray, shape (100,)
    ad_info : dict
        Applicability-domain diagnostic: whether this molecule's
        descriptors are further from the training set than
        ``package.applicability_domain['percentile']``% of training
        molecules are from their own nearest neighbour. This is a
        heuristic reliability flag, not a hard cutoff.
    """
    descriptors = np.asarray(descriptors, dtype=float).reshape(1, -1)
    expected = package.scaler_x.n_features_in_
    if descriptors.shape[1] != expected:
        raise ValueError(
            f"{package.label} expects {expected} descriptors "
            f"({package.feature_columns}), received {descriptors.shape[1]}."
        )

    descriptors_scaled = package.scaler_x.transform(descriptors)
    prediction_scaled = package.model.predict(descriptors_scaled, verbose=0)
    ecd_spectrum = package.scaler_y.inverse_transform(prediction_scaled)[0]
    ad_info = _applicability_domain_check(descriptors_scaled, package.applicability_domain)

    return package.wavelengths, ecd_spectrum, ad_info


def _plot(wavelengths: np.ndarray, ecd_spectrum: np.ndarray, ad_info: dict, out_path: str, label: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(wavelengths, ecd_spectrum, linewidth=2)
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("ECD intensity")
    title = f"{label} predicted ECD spectrum"
    if ad_info["outside_reference_domain"]:
        title += "  ⚠ outside applicability domain"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict the ECD spectrum of a heliceno from descriptors.")
    p.add_argument("--model-dir", default=None, help="Unzipped model package directory.")
    p.add_argument("--label", default=None, choices=VALID_LABELS)
    p.add_argument("--descriptors", default=None,
                    help="Comma-separated descriptor values, in the order of feature_columns.json.")
    p.add_argument("--out-csv", default=None, help="Optional path to save wavelength,ECD as CSV.")
    p.add_argument("--plot", default=None, help="Optional path to save a PNG plot of the spectrum.")
    # parse_known_args ignores extra argv entries injected by the Jupyter/Colab
    # kernel itself (e.g. "-f /root/.../kernel-xxxx.json").
    args, _unknown = p.parse_known_args()
    if args.model_dir is None or args.label is None or args.descriptors is None:
        raise SystemExit(
            "predict_ecd.py needs --model-dir, --label and --descriptors.\n"
            "In Colab it's usually easier to skip the CLI entirely and call the "
            "functions directly, e.g.:\n\n"
            "    from predict_ecd import load_model_package, predict_ecd\n"
            "    pkg = load_model_package('/content/E2E_H_package', 'E2E_H')\n"
            "    wavelengths, ecd, ad_info = predict_ecd([0.16, 0, 0, ...], pkg)\n"
        )
    return args


def main() -> None:
    args = parse_args()
    package = load_model_package(args.model_dir, args.label)
    descriptors = [float(v) for v in args.descriptors.split(",")]

    wavelengths, ecd_spectrum, ad_info = predict_ecd(descriptors, package)

    print(f"Model: {args.label}")
    print(f"Nearest-training-molecule distance: {ad_info['nearest_training_distance']:.4f} "
          f"(reference threshold, P{ad_info['percentile_reference']}: {ad_info['threshold']:.4f})")
    if ad_info["outside_reference_domain"]:
        print("WARNING: this molecule falls outside the model's applicability domain — "
              "treat the prediction with caution.")

    if args.out_csv:
        np.savetxt(args.out_csv, np.column_stack([wavelengths, ecd_spectrum]),
                   delimiter=",", header="wavelength_nm,ecd_intensity", comments="")
        print(f"Spectrum saved to: {args.out_csv}")

    if args.plot:
        _plot(wavelengths, ecd_spectrum, ad_info, args.plot, args.label)
        print(f"Plot saved to: {args.plot}")

    if not args.out_csv and not args.plot:
        for wl, val in zip(wavelengths, ecd_spectrum):
            print(f"{wl:.2f}\t{val:.4f}")


if __name__ == "__main__":
    main()
