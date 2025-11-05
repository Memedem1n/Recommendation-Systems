from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "bt_bert_model"
SRC_ROOT = PROJECT_ROOT / "src"
DATASET_PIPELINE_ROOT = REPO_ROOT / "Dataset_Pipeline"

for path in (SRC_ROOT, DATASET_PIPELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from train import training_loop  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Training module could not be imported: {exc}") from exc


def _check_python_version() -> None:
    if sys.version_info >= (3, 13):
        raise SystemExit(
            "Python 3.13+ detected. PyTorch GPU dağıtımları henüz bu sürümü desteklemiyor.\n"
            "Lütfen Python 3.11 veya 3.12 içeren bir sanal ortam oluşturup tekrar deneyin."
        )


def _ensure_debug_split(train_path: Path, val_path: Path) -> None:
    import pandas as pd  # lazy import

    data_dir = PROJECT_ROOT / "data"
    full_train = data_dir / "train.csv"
    full_val = data_dir / "val.csv"
    if not full_train.exists() or not full_val.exists():
        raise SystemExit(
            "train.csv veya val.csv bulunamadı. Önce notebook üzerinden veri hazırlama adımlarını çalıştırın."
        )
    if train_path.exists() and val_path.exists():
        return

    train_df = pd.read_csv(full_train)
    val_df = pd.read_csv(full_val)
    train_sample = train_df.sample(n=min(2048, len(train_df)), random_state=123)
    val_sample = val_df.sample(n=min(512, len(val_df)), random_state=456)
    train_sample.to_csv(train_path, index=False)
    val_sample.to_csv(val_path, index=False)


def _resolve_config(preset: str) -> Path:
    if preset == "debug":
        train_debug = PROJECT_ROOT / "data" / "train_debug_gpu.csv"
        val_debug = PROJECT_ROOT / "data" / "val_debug_gpu.csv"
        _ensure_debug_split(train_debug, val_debug)
        return PROJECT_ROOT / "configs" / "gpu_debug.yaml"
    if preset == "full":
        return PROJECT_ROOT / "config.yaml"
    raise ValueError(f"Bilinmeyen preset: {preset}")


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA etkin değil. GPU destekli PyTorch kurulumu gereklidir.\n"
            "Örnek kurulum:\n"
            "  conda create -n bluesense python=3.11\n"
            "  conda activate bluesense\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "Ardından bu scripti aynı ortamdan yeniden çalıştırın."
        )
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    return device


def run_training(preset: str) -> Path:
    _check_python_version()
    config_path = _resolve_config(preset)
    device = _select_device()

    print(f"GPU cihazı: {torch.cuda.get_device_name(device)}")
    print(f"Konfigürasyon: {config_path}")

    start = time.time()
    training_loop(config_path=config_path, device=device)
    duration = time.time() - start
    checkpoint_dir = (config_path.parent / "outputs" if preset == "full" else PROJECT_ROOT).resolve()
    print(f"Eğitim süresi: {duration / 60:.1f} dakika")
    return checkpoint_dir


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU destekli yerel BT-BERT eğitimi çalıştır.")
    parser.add_argument(
        "--preset",
        choices=["debug", "full"],
        default="debug",
        help="Debug modunda küçük bir veri alt kümesiyle hızlı eğitim yapılır. 'full' tam veriyle çalışır.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_training(args.preset)


if __name__ == "__main__":
    main()
