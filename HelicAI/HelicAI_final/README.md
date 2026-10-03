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

## How to publish this so anyone can use it

### Option A — GitHub (recommended, especially for collaborators)
1. Create a repository, e.g. `github.com/<you>/HelicAI` (public, or private and shared with your collaborators).
2. Push this whole folder to it.
3. Add this to the very top of `HelicAI.ipynb` as a markdown cell, so it renders as a clickable button on GitHub:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<you>/HelicAI/blob/main/HelicAI.ipynb)
   ```
4. Also update the `REPO_URL` line in the notebook's first setup cell to point at your actual repo URL.
5. Anyone with the link clicks the badge and gets a live, fully working copy in their own Colab — the first setup cell clones the repo automatically, nothing to download by hand.

This is the option we'd recommend pointing collaborators and reviewers to — it's also what most reviewers expect to find linked from a methods section.

### Option B — Just the zip (simpler, no GitHub account needed)
1. Share the zip directly (email, Drive link, whatever).
2. The person unzips it anywhere in their own Google Drive or local machine, e.g. `/content/drive/MyDrive/HelicAI/`.
3. They open `HelicAI.ipynb` in Colab and skip (or adapt) the "clone the repo" setup cell, since they already have the folder — just make sure `BASE_DIR` in that cell points to wherever they put it.

Faster to set up, but not really "shareable with a link" the way Option A is.

## A note on honesty for reviewers

Every prediction in this notebook comes from a neural network trained to approximate quantum-chemistry-computed spectra (DFT / xtb-stda), not from a new quantum-chemistry calculation. Always check the applicability-domain flag (explained in Block I), and where possible, validate standout candidates — especially ones from Block II's genetic search — against an independent method or, ideally, an experimental measurement before relying on them.

## Citation / credit placeholder

If you publish results produced with HelicAI, consider adding a line here pointing to the paper once it exists.
