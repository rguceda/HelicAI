# HelicAI

Predict and design the chiroptical (ECD) spectrum of [6]helicene derivatives with a neural network — no quantum chemistry required at prediction time.

## What's in this package

```
HelicAI/
├── HelicAI.ipynb                 <- START HERE. Block I (predict) and Block II (design).
├── README.md                      <- this file
├── E2E_H_package.zip              Trained E2E_H model (Hammett-only, 16 descriptors)
├── E2E_64_package.zip             Trained E2E_64 model (Hammett + VdW + R+ + R-, 64 descriptors)
├── scripts/                        every Python module the notebook (and your own scripts) can import
│   ├── predict_ecd.py                           Core: load a model package, predict one spectrum
│   ├── screen_substituents.py                   Vocabulary, descriptor matrices, C2-symmetry handling
│   ├── design_candidates.py                     Genetic algorithm / inverse design
│   ├── characterize_model.py                    Full model report (precision, quartiles, error analysis)
│   ├── validate_new_substituents.py             Generalization test vs. real DFT spectra
│   ├── validate_new_substituents_progression.py Same, across increasing substitution levels
│   ├── baseline_comparison.py                   Neural net vs. 1-NN / linear baselines
│   ├── band_area_scaling_study.py               Population-level band statistics by substitution level
│   ├── spectrum_scan.py                         Save the full predicted spectrum of every candidate
│   └── train_pipeline.py                        How the models were originally trained
└── data/
    ├── ECD_total_Gauss.csv                      The full training dataset (DFT/xtb-stda ECD spectra)
    ├── substituent_library.json                 The 16 substituents the models trained on
    └── new_substituent_library_EXAMPLE.json      4 example never-before-seen substituents (CF3, etc.)
```

Everything is bundled — there is nothing to upload, mount, or configure. Clone the repo (or download+unzip it), open `HelicAI.ipynb`, run the setup cells, and both blocks work immediately: the setup cell unzips the two model packages into `models/E2E_H` and `models/E2E_64` the first time it runs.

## Citation / credit placeholder

XXX
