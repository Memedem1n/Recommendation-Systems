"""Training script for BT-BERT implicit concern classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dataset import BTBertDataset
from model import BTBertConfig, BTBertModel  # noqa: E402


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    adamw_beta2: float
    weight_decay: float
    warmup_steps: int
    gradient_accumulation: int
    early_stopping_patience: Optional[int] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_dicts(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    extends = data.pop("extends", None)
    if extends:
        base_path = (path.parent / extends).resolve()
        base_cfg = load_yaml(base_path)
        data = merge_dicts(base_cfg, data)

    return data


def build_dataloader(
    split_csv: Path,
    products_csv: Path,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    text_options: Optional[Dict[str, object]] = None,
    concern_weights: Optional[Dict[str, float]] = None,
) -> DataLoader:
    dataset = BTBertDataset(
        labels_csv=split_csv,
        products_csv=products_csv,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        text_options=text_options,
        concern_weights=concern_weights,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(
    model: BTBertModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels")
            batch.pop("product_id", None)
            batch.pop("sample_weight", None)
            outputs = model(**batch)
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return compute_metrics(logits, labels)


def training_loop(
    config_path: Path,
    device: torch.device,
) -> None:
    cfg = load_yaml(config_path)
    base_dir = (config_path.parent / cfg["project"]["base_dir"]).resolve()
    set_seed(cfg["project"]["random_seed"])

    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"].copy()
    loss_positive_weight_setting = training_cfg.pop("loss_positive_weight", None)
    concern_weights_setting = training_cfg.pop("concern_loss_weights", {})
    train_cfg = TrainingConfig(**training_cfg)
    text_options = cfg.get("text_options", {})
    concern_weights = {
        str(k): float(v)
        for k, v in (concern_weights_setting or {}).items()
    }

    products_path = (base_dir / data_cfg["unified_products_path"]).resolve()
    data_dir = (base_dir / data_cfg["label_output_dir"]).resolve()

    train_csv_name = data_cfg.get("train_csv", "train.csv")
    val_csv_name = data_cfg.get("val_csv", "val.csv")

    train_loader = build_dataloader(
        split_csv=data_dir / train_csv_name,
        products_csv=products_path,
        tokenizer_name=tokenizer_cfg["pretrained_name"],
        max_length=tokenizer_cfg["max_length"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        text_options=text_options,
        concern_weights=concern_weights,
    )
    val_loader = build_dataloader(
        split_csv=data_dir / val_csv_name,
        products_csv=products_path,
        tokenizer_name=tokenizer_cfg["pretrained_name"],
        max_length=tokenizer_cfg["max_length"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        text_options=text_options,
        concern_weights=concern_weights,
    )

    loss_pos_weight: Optional[float] = None
    if isinstance(loss_positive_weight_setting, str) and loss_positive_weight_setting.lower() == "auto":
        labels = train_loader.dataset.labels["label"].astype(float)
        positive = float(labels.sum())
        negative = float(len(labels) - positive)
        if positive > 0:
            loss_pos_weight = torch.tensor(negative / positive, dtype=torch.float32).item()
    elif isinstance(loss_positive_weight_setting, (int, float)):
        value = float(loss_positive_weight_setting)
        if value > 0:
            loss_pos_weight = value

    if loss_pos_weight is not None:
        print(f"Applying BCE positive class weight: {loss_pos_weight:.4f}")

    model = BTBertModel(
        BTBertConfig(
            pretrained_model_name=model_cfg["pretrained_name"],
            attention_scale=model_cfg["attention_scale"],
        ),
        loss_pos_weight=loss_pos_weight,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(0.9, train_cfg.adamw_beta2),
        weight_decay=train_cfg.weight_decay,
    )

    total_steps = len(train_loader) * train_cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    logging_cfg = cfg["logging"]
    checkpoint_dir = (base_dir / logging_cfg["checkpoint_dir"]).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = checkpoint_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history: List[Dict[str, float]] = []
    patience_counter = 0

    for epoch in range(1, train_cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.num_epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels")
            batch.pop("product_id", None)
            sample_weight = batch.pop("sample_weight", None)
            if sample_weight is not None:
                sample_weight = sample_weight.to(device)
            outputs = model(**batch, labels=labels.to(device), sample_weight=sample_weight)
            loss = outputs["loss"] / train_cfg.gradient_accumulation
            loss.backward()
            running_loss += loss.item() * train_cfg.gradient_accumulation

            if step % train_cfg.gradient_accumulation == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress.set_postfix({"loss": running_loss / step})

        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = running_loss / len(train_loader)
        history.append(val_metrics)
        print(f"Validation metrics: {val_metrics}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            checkpoint_path = checkpoint_dir / f"bt_bert_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1

        if (
            train_cfg.early_stopping_patience
            and patience_counter >= int(train_cfg.early_stopping_patience)
        ):
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(patience={train_cfg.early_stopping_patience})."
            )
            break

    metrics_path = (base_dir / logging_cfg["metrics_path"]).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training history -> {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BT-BERT implicit model.")
    parser.add_argument(
        "--config",
        default="bt_bert_model/config.yaml",
        help="Path to configuration file.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config).resolve()

    training_loop(config_path, device)


if __name__ == "__main__":
    main()
