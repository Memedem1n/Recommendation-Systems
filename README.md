<div align="center">

# BlueSense — Data Pipeline and Hybrid Concern Recommendation Pilot

<img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
<img alt="pandas" src="https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white" />
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white" />

</div>

This repository contains a clean data pipeline for product/ingredient datasets, LLM‑free ingredient→concern enrichment, and a dynamic‑weighted hybrid recommendation prototype. It also ships a ready‑to‑present `.pptx` with charts.

## Table of Contents
- Overview
- Repository Structure
- Setup & Quickstart
- Data Pipeline (Dataset_Pipeline)
- Ingredient Knowledge Base Extensions
- Model Training & Evaluation
- Hybrid Scoring & Weight Calibration
- Outputs & Presentation
- Reproducibility & Tips

---

## Overview
- 16,556 products across 13 categories are normalized and merged into a unified schema.
- Ingredient names are canonicalized via synonyms and fuzzy grouping (≥0.85 similarity).
- Ingredient→Concern coverage is expanded with Excel labels, description‑keyword heuristics, category‑based “first ingredient” seeding, and a TF‑IDF + logistic classifier — all offline (no LLM).
- Model selection uses GridSearchCV and probability calibration.
- The hybrid score combines category/ingredient/name/model signals, applies dynamic weights based on category–concern match, adds a rarity boost for informative ingredients, and subtracts a small penalty for unknowns.

## Repository Structure
- `Dataset/` — Raw CSVs (by category)
- `Dataset_Pipeline/` — Pipeline and artifacts
  - `data_pipeline.py` — IO, canonicalization, merge utilities
  - `unified_products.csv`, `ingredient_normalisation_map.csv`, `unique_ingredients.csv`
- `Experiments/Hybrid_Concern_Test/` — Hybrid notebook + outputs
  - `Hybrid_Concern_Test.ipynb`
  - `Experiments/Hybrid_Concern_Test/recommendations.csv`, `bundles.csv`
- `Hybrid_Concern_Presentation.pptx` — Auto‑generated presentation

## Setup & Quickstart
Requirements: Python 3.11, `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `openpyxl`, `python-pptx`.

```bash
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl python-pptx
```

Run options:
- Open and execute `Experiments/Hybrid_Concern_Test/Hybrid_Concern_Test.ipynb` end‑to‑end.
- Or generate the PPTX from repo root:
  ```bash
  python generate_presentation.py
  ```

> Note: LLM enrichment is disabled; all expansion steps are offline.

## Data Pipeline (Dataset_Pipeline)
Main steps (`Dataset_Pipeline/data_pipeline.py`):
1) Category normalization — canonical labels from raw file names.
2) Ingredient canonicalization — manual synonyms + fuzzy (≥0.85) + numeric signatures.
3) Schema unification — align column sets and concatenate.
4) Product ID rebuild — shuffle (random_state=42) then assign `product_id`.
5) Ingredient lists — `ingredients_list_raw/normalised` and counts per product.

Artifacts produced:
- `Dataset_Pipeline/unified_products.csv` — clean product table
- `Dataset_Pipeline/ingredient_normalisation_map.csv` — raw→normal→canonical map
- `Dataset_Pipeline/unique_ingredients.csv` — 18,018 unique canonical ingredients

## Ingredient Knowledge Base Extensions
- Excel labels → initial dictionary (`ingredient_concern_candidates`).
- Description heuristics → concern keywords mined from JSON descriptions.
- Category‑based “first ingredient” → for each concern, from matching categories take the first non‑water ingredient as a weak label (with thresholds and balance).
- TF‑IDF + multi‑label LR → add up to two concerns per ingredient if probability passes a configurable threshold.

Coverage tracking (example): Excel → +A, Descriptions → +B, Category First → +C, TF‑IDF → +D; final coverage is compared to `unique_ingredients`.

## Model Training & Evaluation
Features: `ing_*`, `rare_*` (rarity boost), `cat_*`, `name_*`, `total_*`, `unknown_ratio`.

- Split: `StratifiedShuffleSplit` (80/20) with per‑class presence guaranteed.
- Selection: `LogisticRegression` via `GridSearchCV` (3‑fold, `C∈{0.5,1.0,1.5}`) + `CalibratedClassifierCV (isotonic)`.
- Baselines: RF/GB/KNN with 3‑fold `f1_macro`.

Illustrative outcomes (may vary with environment):
- Calibrated Logistic test macro F1 ≈ 0.94–0.97; accuracy ≈ 97%
- Baselines: GB ≈ 0.975 macro F1, RF ≈ 0.968, KNN ≈ 0.92

## Hybrid Scoring & Weight Calibration
Signals: Category, Ingredient (with rarity), Name, Model; plus `unknown_penalty`.

- Dynamic weights: if the user’s top concern matches the category’s primary concern, increase category weight; otherwise strengthen ingredient+name (keeping the total fixed).
- Weight search: 3×3×3 grid scored by both hit‑rate and nDCG@5; best combo selected.
- Example defaults: Category 0.50, Ingredient 0.20, Name 0.10, Model 0.20.

## Outputs & Presentation
- Recommendations: `Experiments/Hybrid_Concern_Test/Experiments/Hybrid_Concern_Test/recommendations.csv`
- Bundles: `Experiments/Hybrid_Concern_Test/Experiments/Hybrid_Concern_Test/bundles.csv`
- Presentation: `Hybrid_Concern_Presentation.pptx` (auto‑generated)

> CSV columns include score components (`category_component`, `ingredient_component`, `name_component`, `model_component`, `unknown_penalty`) and concern/model probabilities.

## Reproducibility & Tips
- LLM is off; all enrichment is offline and deterministic given seeds.
- For stability on Windows, `n_jobs=1` is used in heavy CV blocks.
- Notebook cells include detailed Markdown notes; the PPT summarizes key visuals and outcomes.

---

Questions or feedback: the “Diagnostics” section in `Experiments/Hybrid_Concern_Test/Hybrid_Concern_Test.ipynb` provides quick visuals to inspect distributions and confidence. You can also tweak weights and thresholds via the `CONFIG` block in the notebook.

