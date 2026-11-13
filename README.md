# HelicAI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rguceda/HelicAI/blob/main/HelicAI.ipynb)

Machine-learning prediction and inverse design of the full electronic circular dichroism (ECD) spectrum of substituted [6]helicenes. No quantum-chemical calculation, 3D structure or geometry optimization is needed at prediction time.

**Open the notebook in Google Colab** (badge above, or <https://colab.research.google.com/github/rguceda/HelicAI/blob/main/HelicAI.ipynb>). It runs in the browser without installation and has two modules:

- **Predict your helicene** — choose the substituent at each of the 16 positions of the scaffold and obtain the complete ECD spectrum (250–600 nm) with E2E-64 and/or E2E-H, together with an applicability-domain check. New substituents can be used with their four descriptors (E2E-64) or with their Hammett constant alone (E2E-H).
- **Design your helicene** — a genetic algorithm coupled to E2E-64 searches for the substitution patterns that best meet a user-defined objective (maximize or minimize a given ECD band, optionally within a wavelength window), under constraints on substituents, positions, number of substituents and *C*<sub>2</sub> symmetry. The settings used in the article are the defaults.

## The models

| Model | Inputs (per position) | |
|---|---|---|
| **E2E-64** | Hammett σ<sub>p</sub>, van der Waals volume, σ<sub>R</sub><sup>+</sup>, σ<sub>R</sub><sup>−</sup> (64 inputs) | most accurate; used for design |
| **E2E-H** | Hammett σ<sub>p</sub> (16 inputs) | any σ<sub>p</sub> value at any position; first-pass exploration |

Both are feed-forward neural networks trained on TD-DFT (M06/TZVP, CH<sub>2</sub>Cl<sub>2</sub>) spectra of 12,757 (*P*)-[6]helicenes bearing one to six substituents (–F, –Cl, –Br, –I, –OH, –OMe, –SH, –SMe, –NH<sub>2</sub>, –CHO, –COOH, –NO<sub>2</sub>, –CN, –CH<sub>3</sub>, –C≡CH, –C≡C–Ph). They predict 100 points between 250 and 600 nm, in *R*·10<sup>40</sup> / esu cm erg G<sup>−1</sup> (Gaussian broadening, σ = 0.30 eV).

## Inverse design as in the article

The design objectives are the areas of the largest positive or negative band (`scripts/custom_objectives.py`). The genetic-algorithm settings of the article are the defaults of `genetic_search`: population 1,000, 50 generations, elite fraction 0.10, tournament size 4, uniform crossover, mutation rate 0.15 per gene, applicability-domain penalty weight 1.0, seed 42.

```python
import design_candidates as dc
from custom_objectives import band_objective, annotate_band_metrics

designs = dc.genetic_search(
    model_dir="models/E2E_64", label="E2E_64",
    library_path="data/substituent_library.json",
    objective=band_objective(metric="positive_band", direction="max"),   # or "negative_band", "min", ...
    target_nm=False, ad_penalty_weight=1.0,
    allowed_substituents=False, allowed_positions=False, force_symmetric=False,
    k_min=1, k_max=4, population_size=1000, n_generations=50,
    output_dir="design_results", top_n=10, plot_top_n=10,
)
designs = annotate_band_metrics(designs)    # A+max and |A-max| in physical units
```

The six design subcases of the article (Supporting Information, Table S19) differ only in these arguments:

| Subcase | `metric` | `direction` | `k_max` | `target_nm` |
|---|---|---|---|---|
| 1 | `positive_band` | `max` | 4 | – |
| 2 | `positive_band` | `max` | 6 | – |
| 3 | `positive_band` | `max` | 4 | 500 (± 30 nm) |
| 4 | `positive_band` | `min` | 4 | – |
| 5 | `positive_band` | `min` | 6 | – |
| 6 | `negative_band` | `min` | 4 | – |

The genetic algorithm is stochastic (seed 42), and small numerical differences between TensorFlow versions can change its trajectory. The top candidates may therefore differ slightly from those of the article, while the same substitution motifs are recovered.

or from the command line:

```bash
python scripts/design_candidates.py --model-dir models/E2E_64 --label E2E_64 \
    --library data/substituent_library.json --mode genetic \
    --objective positive_band:min --k-max 4 --output-dir design_results
```

## Contents

```
HelicAI/
├── LICENSE                       MIT License
├── CITATION.cff                  how to cite HelicAI
├── requirements.txt              Python packages
├── HelicAI.ipynb                 START HERE: "Predict your helicene" and "Design your helicene"
├── E2E_H_package.zip             trained E2E-H model (unpacked to models/ by the notebook)
├── E2E_64_package.zip            trained E2E-64 model
├── scripts/
│   ├── predict_ecd.py                           load a model package, predict one spectrum
│   ├── screen_substituents.py                   vocabulary, descriptor matrices, C2-symmetry handling
│   ├── design_candidates.py                     genetic algorithm, hill climbing and random search
│   ├── custom_objectives.py                     band-area design objectives used in the article
│   ├── characterize_model.py                    model report (precision, quartiles, error analysis)
│   ├── validate_new_substituents.py             validation against TD-DFT for new substituents
│   ├── validate_new_substituents_progression.py same, by substitution level
│   ├── baseline_comparison.py                   neural networks vs. 1-NN and ridge baselines
│   ├── band_area_scaling_study.py               band statistics by substitution level (Case II)
│   ├── spectrum_scan.py                         full predicted spectrum of every candidate
│   └── train_pipeline.py                        training of the models
└── data/
    ├── ECD_total_Gauss.csv                      training dataset (TD-DFT ECD spectra)
    ├── substituent_library.json                 descriptors of the 16 training substituents
    └── new_substituent_library_EXAMPLE.json     the four new substituents of Case III
```

## The dataset

`data/ECD_total_Gauss.csv` (semicolon-separated, 12,757 rows, one molecule per row) contains the TD-DFT data used for training:

| Columns | Content |
|---|---|
| `Archivo` | name of the TD-DFT output file, which encodes the substitution pattern |
| `nsust` | number of substituents (1–6) |
| `Pos_1` … `Pos_16` | Hammett σ<sub>p</sub> of the group at each position (0 = H) |
| `VdW_1` … `VdW_16` | van der Waals volume (Å<sup>3</sup>) |
| `Rplus_1` … `Rplus_16`, `Rminus_1` … `Rminus_16` | resonance constants σ<sub>R</sub><sup>+</sup> and σ<sub>R</sub><sup>−</sup> |
| `R_1`, `nm_1` … `R_100`, `nm_100` | rotatory strength (10<sup>−40</sup> esu cm erg G<sup>−1</sup>) and wavelength (nm) of the 100 lowest-energy singlet transitions |

The continuous spectra are rebuilt from these transitions by `scripts/train_pipeline.py` (Gaussian broadening, σ = 0.30 eV; Supporting Information, Section S1.2).

## Running locally

Clone the repository, run `pip install -r requirements.txt`, and open `HelicAI.ipynb` from the repository folder.

## License

HelicAI (code, notebook, trained models, substituent libraries and dataset) is released under the [MIT License](LICENSE): it can be freely used, modified and redistributed, including for commercial purposes, provided that the copyright and license notice is kept.

## Citation

If you use HelicAI, please cite: *[article reference to be added on publication]*.
