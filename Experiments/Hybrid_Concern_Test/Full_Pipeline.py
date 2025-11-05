# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (bluesense)
#     language: python
#     name: bluesense
# ---

# %% [markdown]
# # Full Pipeline: Data Prep to BT-BERT Recommendations
#
# Bu notebook, BlueSense veri setini ham halinden normalize edilmiş formata dönüştürür, BT-BERT etiketleme scriptini çalıştırır ve eğitilmiş model üzerinden kullanıcı profillerine göre öneriler üretir.
#

# %% [markdown]
# ## 0. Kurulum
# Gerekli yolları `sys.path` üzerine ekleyip temel paketleri yüklüyoruz. Bu notebook proje kök dizininden çalıştırılacak şekilde hazırlanmıştır.
#

# %%
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

NOTEBOOK_ROOT = Path.cwd().resolve()
for candidate in [NOTEBOOK_ROOT, *NOTEBOOK_ROOT.parents]:
    if (candidate / 'bt_bert_model').exists() and (candidate / 'Dataset_Pipeline').exists():
        REPO_ROOT = candidate
        break
else:
    raise RuntimeError('Repository root not found. Notebook must be inside project tree.')

PROJECT_ROOT = REPO_ROOT / 'bt_bert_model'
DATASET_ROOT = REPO_ROOT / 'Dataset'
DATASET_PIPELINE_ROOT = REPO_ROOT / 'Dataset_Pipeline'
SERVICE_ROOT = REPO_ROOT / 'service'

for path_candidate in (REPO_ROOT, PROJECT_ROOT, DATASET_PIPELINE_ROOT, SERVICE_ROOT):
    if str(path_candidate) not in sys.path:
        sys.path.append(str(path_candidate))

print(f'Repo root: {REPO_ROOT}')
print(f'Project root: {PROJECT_ROOT}')
print(f'Dataset source: {DATASET_ROOT}')

VENV_GPU_ROOT = REPO_ROOT / '.venv_gpu'
if VENV_GPU_ROOT.exists():
    expected_python = (
        VENV_GPU_ROOT / ('Scripts' if os.name == 'nt' else 'bin') / ('python.exe' if os.name == 'nt' else 'python')
    )
    current_python = Path(sys.executable).resolve()
    if expected_python.exists() and current_python != expected_python.resolve():
        raise EnvironmentError(
            'GPU environment is not active. Activate `.venv_gpu` and select that kernel before running.'
        )
else:
    print('Warning: `.venv_gpu` directory not found; GPU environment may be missing.')

total_cores = os.cpu_count() or 1
max_threads = max(1, total_cores // 2)
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, str(max_threads))

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

if psutil is not None:
    try:
        proc = psutil.Process()
        if hasattr(proc, 'cpu_affinity'):
            cores = list(range(total_cores))
            proc.cpu_affinity(cores[:max_threads] or cores)
    except Exception as affinity_err:
        print(f'CPU affinity could not be set: {affinity_err}')
# %% [markdown]
# ## 1. Dataset Pipeline (Ham Veriden Normalize Çıktıya)
# EWG kaynaklı ham CSV dosyaları `Dataset_Pipeline` modülü ile temizlenir ve `bt_bert_model/data/raw/` altına yazılır. Dosyalar zaten mevcutsa işlem atlanır.
#

# %%
import importlib.util
from datetime import datetime

module_path = DATASET_PIPELINE_ROOT / 'data_pipeline.py'
if not module_path.exists():
    raise FileNotFoundError(f'Data pipeline module not found at {module_path}')
spec = importlib.util.spec_from_file_location('dataset_pipeline', module_path)
data_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline)

RAW_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw'
unified_path = RAW_OUTPUT_DIR / 'unified_products.csv'

if not RAW_OUTPUT_DIR.exists() or not unified_path.exists():
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = datetime.now()
    pipeline_outputs = data_pipeline.run_pipeline(DATASET_ROOT, RAW_OUTPUT_DIR)
    duration = datetime.now() - start
    print(f'Dataset pipeline tamamlandı ({duration}).')
    display(pipeline_outputs.products.head())
else:
    print('Normalize edilmiş ürün dosyaları zaten mevcut; pipeline adımı atlandı.')
    display(pd.read_csv(unified_path).head())


# %% [markdown]
# ## 2. BT-BERT Etiketleme ve Veri Ayrımı
# `bt_bert_model/src/data_prep.py` scripti, normalize ürün tablosu üzerinden concern etiketlerini üretir ve train/val/test CSV'lerini oluşturur. CSV'ler mevcutsa tekrar çalıştırmak için `FORCE_LABELS = True` yapabilirsiniz.
#

# %%
import subprocess

labels_path = PROJECT_ROOT / 'data' / 'labels.csv'
FORCE_LABELS = False
if FORCE_LABELS or not labels_path.exists():
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'src' / 'data_prep.py'),
        '--config',
        str(PROJECT_ROOT / 'config.yaml'),
    ]
    print('Running:', ' '.join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)
else:
    print('Etiket dosyaları mevcut; data_prep adımı atlandı.')

label_summary = pd.read_json(PROJECT_ROOT / 'data' / 'label_summary.json')
label_summary


# %% [markdown]
# ## 3. Egitilmis BT-BERT Modelini Yukle ve Olasilik Matrisi Olustur
# Servis katmanindaki `BTBertRecommender` ile egitilmis modeli yüklüyor ve tum urunler icin concern olasiliklarini hesapliyoruz (yalnizca CPU kullanarak).
#

# %%
from service.config import ServiceSettings
from service.recommender import BTBertRecommender
import torch

checkpoint_path = PROJECT_ROOT / 'outputs' / 'new_bt_bert' / 'hp_0' / 'bt_bert_epoch1.pt'
products_csv = PROJECT_ROOT / 'data' / 'raw' / 'unified_products.csv'
config_path = PROJECT_ROOT / 'config.yaml'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
inference_batch_size = 128 if device == 'cuda' else 32
print(f'Using device: {device} (batch size: {inference_batch_size})')
if device != 'cuda':
    raise EnvironmentError(
        'CUDA cihazı tespit edilemedi. GPU destekli kerneli seçip yeniden başlatın.'
    )

settings = ServiceSettings(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    products_csv=products_csv,
    device=device,
    inference_batch_size=inference_batch_size,
)
recommender = BTBertRecommender(settings)
concerns = recommender.available_concerns()
products_df = recommender.products.copy()
products_df.head()


# %%
probability_rows = []
product_ids = products_df.index.tolist()

for concern in concerns:
    results = recommender.score_products(concern, product_ids)
    for res in results:
        probability_rows.append({
            'product_id': res.product_id,
            'concern': concern,
            'probability': res.probability,
            'name': res.name,
            'category': res.category,
        })

model_predictions = pd.DataFrame(probability_rows)
model_predictions.head()


# %%
prob_matrix = model_predictions.pivot_table(
    index='product_id',
    columns='concern',
    values='probability',
    fill_value=0.0
)
prob_matrix.head()


# %% [markdown]
# ## 4. Kullanıcı Profiline Göre Sıralama Fonksiyonu
# Concern ağırlık vektörü verilen ürünlerin sıralanması için yardımcı fonksiyon.
#

# %%
def rank_products_for_profile(profile: dict[str, float], top_k: int = 10, min_score: float = 0.0) -> pd.DataFrame:
    weights = {k: float(v) for k, v in profile.items() if float(v) > 0 and k in prob_matrix.columns}
    total = sum(weights.values())
    if not weights or total == 0:
        raise ValueError('Profilde kullanılabilir concern bulunamadı.')
    weights = {k: v / total for k, v in weights.items()}
    score_series = sum(prob_matrix[c] * w for c, w in weights.items())
    ranking = products_df.join(score_series.rename('score')).sort_values('score', ascending=False)
    if min_score > 0:
        ranking = ranking[ranking['score'] >= min_score]
    columns = ['title_text', 'category', 'product_url', 'score']
    return ranking.loc[:, [col for col in columns if col in ranking.columns]].head(top_k)



# %% [markdown]
# ## 5. Demo Kullanıcı Profilleri
# Her concern için baskın ağırlığa sahip 7 kullanıcının top-5 önerilerini listeliyoruz.
#

# %%
from IPython.display import display

concerns = list(prob_matrix.columns)
demo_users = {f'user_{c}': {concern: (1.0 if concern == c else 0.05) for concern in concerns} for c in concerns}

for user_id, profile in demo_users.items():
    print(f'=== {user_id} ===')
    display(rank_products_for_profile(profile, top_k=5))
    print()


# %% [markdown]
# ## 6. Model Olasılıklarını Dışa Aktar
# Pivot tabloyu CSV olarak kaydediyoruz; servis veya başka analizlerde tekrar kullanılabilir.
#

# %%
output_path = PROJECT_ROOT / 'outputs' / 'bt_bert_model_probabilities.csv'
prob_matrix.reset_index().to_csv(output_path, index=False)
print(f'Model olasılık matrisi kaydedildi: {output_path}')

