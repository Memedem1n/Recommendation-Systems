"""Evaluation script for BT-BERT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dataset import BTBertDataset
from model import BTBertConfig, BTBertModel  # noqa: E402
from train import compute_metrics, load_yaml  # noqa: E402


def evaluate_checkpoint(
    checkpoint: Path,
    config_path: Path,
    split: str,
    device: torch.device,
) -> dict:
    cfg = load_yaml(config_path)
    base_dir = (config_path.parent / cfg["project"]["base_dir"]).resolve()

    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    text_options = cfg.get("text_options", {})
    concern_weights = {
        str(k): float(v)
        for k, v in (cfg.get("training", {}).get("concern_loss_weights", {}) or {}).items()
    }

    products_path = (base_dir / data_cfg["unified_products_path"]).resolve()
    data_dir = (base_dir / data_cfg["label_output_dir"]).resolve()
    split_file = data_cfg.get(f"{split}_csv", f"{split}.csv")
    split_path = data_dir / split_file

    dataset = BTBertDataset(
        labels_csv=split_path,
        products_csv=products_path,
        tokenizer_name=tokenizer_cfg["pretrained_name"],
        max_length=tokenizer_cfg["max_length"],
        text_options=text_options,
        concern_weights=concern_weights,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = BTBertModel(
        BTBertConfig(
            pretrained_model_name=model_cfg["pretrained_name"],
            attention_scale=model_cfg["attention_scale"],
        )
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch.pop("product_id", None)
            batch.pop("sample_weight", None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(logits, labels)
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BT-BERT checkpoint.")
    parser.add_argument(
        "--config",
        default="bt_bert_model/config.yaml",
        help="Path to configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="bt_bert_model/outputs/eval_metrics.json",
        help="Path to store evaluation metrics JSON.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent / checkpoint_path).resolve()
    metrics = evaluate_checkpoint(
        checkpoint=checkpoint_path,
        config_path=config_path,
        split=args.split,
        device=device,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
