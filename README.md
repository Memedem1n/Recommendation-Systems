# BlueSense Recommendation Systems

This repository hosts every stage of the BlueSense cosmetics recommendation stack: data collection, normalisation, labelling, model training, and experiment notebooks. Each sub-directory includes its own documentation; this file provides the high-level map and workflow.

## Directory Overview

```
.
|- bt_bert_model/            # Concern classification model, configs, data splits, SLURM jobs
|- Dataset/                  # Raw category-level scrapes from the EWG Skin Deep catalogue
|- Dataset_Pipeline/         # Ingredient and product normalisation scripts and notebooks
|- Ewg_Scraper/              # Selenium-based scrapers (tracked as a Git submodule)
|- Experiments/              # Hybrid concern notebooks, configs, cached recommendation bundles
|- about_ingredients.json    # Consolidated ingredient metadata
|- ingredient_concern_map.xlsx
```

### Key Components

- `bt_bert_model/` – End-to-end BT-BERT implicit concern classifier: data prep, training, evaluation, explainability, and automated experiment manager. See `bt_bert_model/README.md`.
- `Ewg_Scraper/` – Production scraper suite with reusable Selenium helpers and category-specific runners. Clone the repository with submodules to keep it in sync with its upstream origin.
- `Dataset/` – Category CSV exports (`Anti-aging.csv`, `Mask_part1.csv`, `Serums__Essences.csv`, etc.) produced by the scraper.
- `Dataset_Pipeline/` – Utilities for merging raw CSVs, standardising ingredient names, and generating unified product tables.
- `Experiments/Hybrid_Concern_Test/` – Reference configs (`config.yaml`), a Jupyter notebook, and cached outputs for blending rule-based scores with BT-BERT predictions.

## Getting Started

1. **Clone with submodules**
   ```bash
   git clone --recurse-submodules https://github.com/<your-account>/Recommendation-Systems.git
   cd Recommendation-Systems
   ```
   Already cloned? Run:
   ```bash
   git submodule update --init --recursive
   ```

2. **Create a Python environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   python -m pip install --upgrade pip
   python -m pip install -r bt_bert_model/requirements.txt
   ```
   Install scraper dependencies when scraping fresh data:
   ```bash
   python -m pip install -r Ewg_Scraper/requirements.txt
   ```

3. **Expose local packages (optional)**
   The downstream scripts expect `Dataset_Pipeline` to be importable. Either install it in editable mode:
   ```bash
   python -m pip install -e Dataset_Pipeline
   ```
   or export `PYTHONPATH=%CD%`.

## Typical Workflow

1. **Scrape source data**
   - Configure category drivers in `Ewg_Scraper/Ready_Scrapers/`.
   - Run `python Ready_Scrapers/run_parallel_scrapers.py` to generate fresh CSV files per category.
   - Inspect `Ready_Scrapers/url_cache/` to confirm pagination coverage.

2. **Normalise products and ingredients**
   - Execute `Dataset_Pipeline/data_pipeline.py` or walk through `Data_Normalisation_Pipeline.ipynb`.
   - Outputs include:
     - `unified_products.csv`
     - `ingredient_normalisation_map.csv`
     - `unique_ingredients.csv`

3. **Label and split data for BT-BERT**
   ```bash
   python bt_bert_model/src/data_prep.py --config bt_bert_model/config.yaml
   ```
   Generates `labels.csv`, `train.csv`, `val.csv`, `test.csv`, and `label_summary.json` under `bt_bert_model/data/`.

4. **Train and evaluate BT-BERT**
   ```bash
   python bt_bert_model/src/train.py --config bt_bert_model/config.yaml
   python bt_bert_model/src/evaluate.py --config bt_bert_model/config.yaml \
       --checkpoint bt_bert_model/outputs/checkpoints/bt_bert_epoch1.pt \
       --split test \
       --output bt_bert_model/outputs/eval_metrics.json
   ```
   Switch configs in `bt_bert_model/configs/` for scenario experiments or drive sweeps via `src/experiment_manager.py` (see the subproject README).

5. **Blend with rule-based heuristics**
   - Open `Experiments/Hybrid_Concern_Test/Hybrid_Concern_Test.ipynb`.
   - Load the latest BT-BERT outputs and `product_concern_weights.csv`.
   - Export hybrid recommendations to `Experiments/Hybrid_Concern_Test/Experiments/Hybrid_Concern_Test/`.

## Hugging Face Cache (offline clusters)

Cluster nodes without internet access must pre-download transformer weights:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('bert-base-uncased', cache_dir='C:/Users/barut/hf_cache')"
scp -r C:/Users/barut/hf_cache <user>@<cluster-host>:~/hf_cache
```
SLURM scripts inside `bt_bert_model/scripts/` set `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_OFFLINE=1` so training remains offline.

## Large Artefacts

- `bt_bert_model/outputs/checkpoints/` – PyTorch checkpoints; keep only the necessary ones to limit repository size.
- `bt_bert_model/Papers/` – Supporting literature for experimentation.
- `Experiments/Hybrid_Concern_Test/product_concern_weights.csv` – Current weighting table used by the hybrid recommender.
- Root-level JSON/XLS files – Ingredient metadata consumed by both the pipeline and the model.

## Data Integrity Checklist

- Ensure new scrapes populate `id`, `category`, `product_url`, `name`, and `ingredients` columns before feeding them into the pipeline.
- Rerun `Dataset_Pipeline` when ingredient mappings change.
- Regenerate BT-BERT data splits after schema updates to avoid stale labels.

## Git Workflow Notes

- Commit inside `Ewg_Scraper/` first; the root repository tracks only the submodule pointer.
- After updating the scraper, run `git add Ewg_Scraper` from the root to record the new submodule revision.
- Add `.gitmodules` to version control so collaborators can initialise the scraper submodule automatically.
- Prefer storing large intermediate outputs outside the repository when possible (release archives, object storage, etc.).

## References

- `bt_bert_model/README.md` – Model configuration, experiment manager usage, and explainability tooling.
- `Ewg_Scraper/README.md` – Scraper configuration, batching strategy, and troubleshooting.
- `Dataset_Pipeline/Data_Normalisation_Pipeline.ipynb` – Guided walkthrough of the cleansing pipeline.
- `Experiments/Hybrid_Concern_Test/Hybrid_Concern_Test.ipynb` – Notebook that merges concern scores from multiple sources.

With these components you can progress from raw EWG pages to trained concern classifiers and hybrid recommendation bundles ready for downstream products.
