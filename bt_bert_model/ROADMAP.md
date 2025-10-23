# BT-BERT Migration Roadmap

## Objectives
- Implement the implicit, attention-logit BT-BERT architecture described in *Beauty Beyond Words* for our catalogue.
- Limit scope to the 7 concerns available in our data (`redness`, `eyebag`, `acne`, `oiliness`, `wrinkle`, `age`, `moisture`).
- Deliver a reproducible pipeline: data labelling › dataset builders › model training/evaluation › explainability reports.

## 1. Data Labelling Strategy
1. **Positive labels via category rules**  
   - Maintain mapping `category › {concerns}` (existing hints).  
   - All products in mapped categories are positive for those concerns.
2. **Positive labels via ingredient heuristics**  
   - Reuse hybrid rules (e.g. niacinamide ? redness/moisture, salicylic acid ? acne).  
   - Apply only when ingredient present *and* category-compatible to reduce noise.
3. **Negative labels**  
   - For a given concern, any product without category/ingredient trigger is treated as negative.  
   - Additional hard negatives: products in categories mutually exclusive with the concern (e.g. `oil controller` ? negative for `moisture`).
4. **Label audit**  
   - Generate summary stats (per concern positives/negatives, rule overlaps).  
   - Persist label table under `bt_bert_model/data/labels.csv`.

## 2. Data Preparation Tasks
- `data_prep.py`: load unified products, apply labelling rules, output train/val/test splits (stratified per concern).  
- `dataset.py`: Torch `Dataset` producing `(input_ids, attention_mask, label)` for (product, concern) pairs.  
- Tokenizer helper to compose `[CLS] concern tokens [SEP] ingredient tokens [SEP] title tokens` truncated to 512.

## 3. Model Implementation
- `model.py`: BT-BERT class
  - Base encoder = `bert-base-uncased` (configurable).  
  - Disable FFN in last layer; extract final-layer attention, sum over heads › logits = 16 × attention[:, :, 0, 0].
- Loss: BCEWithLogits for single-label binary classification.
- Metrics utils (accuracy, precision, recall, F1 at concern level).

## 4. Training & Evaluation
- `train.py` CLI
  - Hyperparameters from `config.yaml` (batch=8, LR=3e-5, AdamW with ß²=0.95, weight decay schedule).  
  - Implicit augmentation: iterate over every concern per product (positive or negative).
- `evaluate.py`
  - Report metrics on validation/test split.  
  - Produce per-concern confusion matrices.  
  - Save attention heatmaps for sampled products.

## 5. Explainability & Reports
- `explain.py`: implement Algorithm 2 (top attention tokens).  
- Generate CSV/JSON with top tokens per (product, concern).  
- Notebook `bt_bert_analysis.ipynb`: illustrate attention highlights and compare against hybrid scores.

## 6. Integration & Comparison
- Export predicted probabilities for recommendation pipeline compatibility (`bt_bert_outputs.csv`).  
- Add comparison script vs. hybrid model (precision, overlap, duplicate rate).  
- Document usage in `README.md`.

## 7. Open Items / Dependencies
- Finalize ingredient›concern mapping table for heuristics.  
- Decide on GPU availability & training budget.  
- Optional: explicit baseline for parity with paper (future work).

## Folder Plan
- `bt_bert_model/config.yaml`
- `bt_bert_model/data/` (raw labels, splits)
- `bt_bert_model/src/` (`data_prep.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `explain.py`)
- `bt_bert_model/notebooks/bt_bert_analysis.ipynb`
- `bt_bert_model/outputs/` (checkpoints, metrics, explainability artefacts)
- `bt_bert_model/README.md`
