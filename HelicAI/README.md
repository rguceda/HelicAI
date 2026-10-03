# HelicAI

Predict and design the chiroptical (ECD) spectrum of [6]helicene derivatives with a neural network — no quantum chemistry required at prediction time.

## What's in this package

```
HelicAI/
├── HelicAI.ipynb          <- START HERE. The notebook with Block I (predict) and Block II (design).
├── README.md              <- this file
├── scripts/                <- every Python module the notebook (and your own scripts) can import
│   ├── predict_ecd.py                           Core: load a model package, predict one spectrum
│   ├── design_candidates.py                     Genetic algorithm / inverse design
│   ├── characterize_model.py                    Full model report (precision, quartiles, error analysis)
│   ├── validate_new_substituents.py             Generalization test vs. real DFT spectra
│   ├── validate_new_substituents_progression.py Same, across increasing substitution levels
│   ├── baseline_comparison.py                   Neural net vs. 1-NN / linear baselines
│   ├── band_area_scaling_study.py               Population-level band statistics by substitution level
│   ├── spectrum_scan.py                         Save the full predicted spectrum of every candidate
│   ├── train_pipeline.py                        How the models were originally trained
│   └── screen_substituents.py                   *** MISSING -- see below ***
└── data/
    ├── substituent_library.json                 The 16 substituents the models trained on
    └── new_substituent_library_EXAMPLE.json      4 example never-before-seen substituents (CF3, etc.)
```

**Not included** (too large / lives on your own Drive): the trained model packages themselves (`E2E_H_package`, `E2E_64_package` — the `.keras` model + scalers + applicability domain) and the raw training CSV (`ECD_total_Gauss.csv`). The notebook tells you exactly where to point it.

## ⚠️ One missing file

`screen_substituents.py` is imported by `design_candidates.py`, `spectrum_scan.py`, and `band_area_scaling_study.py`, but wasn't part of this upload. Block I (prediction) works fully without it — the notebook builds descriptor vectors itself. Block II (the genetic algorithm) needs it. Add it to `scripts/` before running Block II; `HelicAI.ipynb` lists exactly what functions it must provide.

## How to publish this so anyone can use it

Two realistic options:

### Option A — GitHub (recommended for "anyone, including collaborators")
1. Create a public (or private, for just your collaborators) GitHub repository, e.g. `github.com/<you>/HelicAI`.
2. Push this whole folder to it (`git init`, `git add .`, `git commit`, `git push`) — either from your own machine, or directly from the GitHub website's "upload files" button if you don't want to use git locally.
3. Add this line to the very top of `HelicAI.ipynb` (as a markdown cell) so it renders as a clickable "Open in Colab" button on GitHub:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<you>/HelicAI/blob/main/HelicAI.ipynb)
   ```
4. Anyone with the link can now click that badge and get a live, editable copy of the notebook in their own Colab — no downloading or unzipping anything. They'll still need to set `SCRIPTS_DIR` to wherever they clone/upload the `scripts/` and `data/` folders (the notebook's setup cell has a line for exactly this — a `git clone` of the same repo works great there).
5. Your trained model packages themselves are too large for a typical GitHub repo (and shouldn't be public if unpublished) — keep those on Drive, as the notebook already assumes.

This is the better option if you want this to look and feel like a real, citable research artifact — it's also what most reviewers expect to find linked from a methods section nowadays.

### Option B — Just the zip (simpler, no GitHub account needed)
1. Share this `HelicAI.zip` directly (email, Drive link, whatever).
2. The person unzips it into their own Google Drive, e.g. `/content/drive/MyDrive/HelicAI/`.
3. They open `HelicAI.ipynb` in Colab (Drive → right-click → "Open with" → "Google Colaboratory", or upload it directly to colab.research.google.com), and edit the `SCRIPTS_DIR` line in the setup cell to match where they put it.

Faster to set up, but less polished, and not really "shareable with a link" the way Option A is — each person has to manually download and unzip.

**Our recommendation**: Option A for the version you point collaborators and reviewers to; Option B is fine for a quick one-off share.

## Citation / credit placeholder

If you publish results produced with HelicAI, consider adding a line here pointing to the paper once it exists — makes it easy for others (and future you) to credit it correctly.
