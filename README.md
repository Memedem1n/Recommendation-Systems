# Recommendation Systems

BlueSense Recommendation Systems combines data ingestion, canonicalisation, model training, and serving so we can deliver skin concern recommendations from a single project. The repository holds the scraping utilities, the BT-BERT concern classifier, notebooks that orchestrate the full workflow, and the FastAPI service that powers user-facing recommendations.

## How the pieces fit

1. **Data acquisition**  
   `Ewg_Scraper/` fetches raw category pages from the EWG Skin Deep catalogue. Each run deposits category specific CSV files in `Dataset/`.
2. **Canonical data pipeline**  
   The scripts and notebooks in `Dataset_Pipeline/` clean ingredient names, merge duplicate products, and export a harmonised catalogue (`unified_products.csv`) together with ingredient metadata.
3. **Concern labelling and dataset build**  
   `bt_bert_model/src/data_prep.py` converts the harmonised catalogue into BT-BERT ready splits (`train.csv`, `val.csv`, `test.csv`) and produces concern frequency reports for auditing.
4. **BT-BERT fine-tuning**  
   Training, evaluation, and experiment management live under `bt_bert_model/`. Fine-tuned checkpoints are stored in `bt_bert_model/outputs/`. A checkpoint exported from the cluster (for example `outputs/new_bt_bert/hp_0/bt_bert_epoch1.pt`) is the default model used by notebooks and the service.
5. **Product-to-concern scoring**  
   Inference utilities (`service/recommender.py`) load the trained model, calculate concern scores for every product, and cache them for downstream consumers.
6. **End-to-end demonstration**  
   `Experiments/Hybrid_Concern_Test/Full_Pipeline.ipynb` orchestrates the entire workflow: data preparation recap, loading the trained model, generating product concern tables, defining demo users, and producing recommendation lists ready to present.
7. **Serving layer**  
   `service/main.py` exposes the recommender through a FastAPI application with health, concern catalogue, and recommendation endpoints.

## Repository layout

- `bt_bert_model/` - BT-BERT configs, training scripts, experiment manager, and model checkpoints.
- `Dataset/` - Raw category exports produced by the scraper.
- `Dataset_Pipeline/` - Normalisation scripts and helper notebooks that transform raw CSVs into canonical datasets.
- `Ewg_Scraper/` - Selenium based scrapers (kept as a submodule so you can pull updates independently).
- `Experiments/Hybrid_Concern_Test/` - End-to-end pipeline notebook, cached outputs, and concern scoring experiments.
- `service/` - FastAPI entrypoint, configuration, and reusable recommender wrapper.
- `cluster_package.zip` - Ready-to-upload bundle that contains the files required to reproduce training on the rorqual cluster.
- `requirements-service.txt` - Minimal dependency set for running the service outside of the full training environment.

## Environment setup

The repository ships with two Python workflows:

1. **Full training environment** (GPU ready) defined by `bt_bert_model/requirements.txt`.
2. **Lightweight CPU service environment** defined by `requirements-service.txt`, which also pins `pydantic<2` so the service modules can be imported from notebooks.

Recommended steps on Windows PowerShell:

```powershell
python -m venv .venv_cpu
.\.venv_cpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r bt_bert_model/requirements.txt
python -m pip install -r requirements-service.txt
```

If you only need the service or the pipeline notebooks you may install the smaller requirement set and skip GPU specific packages.

## Running the full pipeline notebook

`Experiments/Hybrid_Concern_Test/Full_Pipeline.ipynb` is the canonical demonstration:

- The first section documents data preparation assumptions and points to the scripts that generate the harmonised datasets.
- The notebook loads the fine-tuned BT-BERT checkpoint from `bt_bert_model/outputs/new_bt_bert/hp_0/`.
- Product concern scores are generated inside the notebook and stored in memory for recommendation steps.
- Seven demo users are created with distinct dominant concerns so you can verify the recommendation ranking.
- The final section prints ranked product slates per user and can be adapted to feed QA scenarios or UI prototypes.

Run the notebook with the `.venv_cpu` kernel to avoid pydantic version conflicts. GPU execution is optional because inference runs comfortably on CPU.

## Recommendation service quickstart

1. Activate your environment.
2. Install dependencies if you have not already: `python -m pip install -r requirements-service.txt`.
3. Ensure the checkpoint exists at `bt_bert_model/outputs/new_bt_bert/hp_0/bt_bert_epoch1.pt` (replace the path with your preferred checkpoint if needed).
4. Launch the API:

```powershell
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

Key endpoints:
- `GET /health` - verifies paths, model version, and dataset freshness.
- `GET /concerns` - returns the concern vocabulary supported by the loaded checkpoint.
- `POST /recommendations` - accepts a concern, optional filters (`top_k`, `category`, `include_only`), and yields ranked products.

Logs default to `bt_bert_model/outputs/recommendation_logs.jsonl`. Set `BLUESENSE_RECOMMENDATION_LOG_PATH` to stream logs elsewhere.

## Training on the rorqual cluster

`cluster_package.zip` bundles the configuration, scripts, and minimal data required to launch training jobs on the cluster. Upload the archive to your home directory, extract it, and submit jobs with the SLURM scripts under `bt_bert_model/scripts/`. The scripts set `TRANSFORMERS_OFFLINE=1` so you can point `HF_HOME` at a pre-populated Hugging Face cache when the cluster has no internet access.

## Artefact locations

- Model checkpoints: `bt_bert_model/outputs/new_bt_bert/hp_0/`.
- Thresholds and concern metadata: `bt_bert_model/outputs/new_bt_bert/hp_0/thresholds.json`.
- Canonical product catalogue: `Dataset_Pipeline/output/unified_products.csv`.
- Notebook exports: `Experiments/Hybrid_Concern_Test/Full_Pipeline.ipynb` and the paired `Full_Pipeline.py` script (generated via Jupytext for version control).
- Service configuration defaults: `service/config.py`.

Keep large intermediate CSVs and caches outside the repository when possible to reduce noise in git history.

## Roadmap

- Automate nightly EWG scraping and pipeline refresh with scheduler friendly entry points.
- Add data quality checks to the dataset pipeline (ingredient coverage, duplicate detection, concern distribution drift).
- Containerise the FastAPI service with GPU optional images for on-prem deployments.
- Implement evaluation dashboards that compare notebook generated recommendations against production logs.
- Expand unit and integration tests around `service/recommender.py` to guard against regression when swapping checkpoints.

## See also

- `bt_bert_model/README.md` for detailed training and experiment manager documentation.
- `Dataset_Pipeline/` notebooks for canonicalisation logic and ingredient mapping audits.
- `service/README.md` (if present) for deployment specifics.

With these components you can reproduce the full BlueSense workflow: scrape, normalise, train, package, and serve concern aware product recommendations.
