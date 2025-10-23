"""Generate prediction scores for all product-concern pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dataset import BTBertDataset  # noqa: E402
from model import BTBertConfig, BTBertModel  # noqa: E402
from train import load_yaml  # noqa: E402


def generate_predictions(
    checkpoint: Path,
    config_path: Path,
    split: str,
    device: torch.device,
) -> pd.DataFrame:
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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

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

    rows = []
    pointer = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            product_ids = batch.pop("product_id").cpu().numpy()
            batch.pop("sample_weight", None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            labels_np = labels.cpu().numpy()
            concerns = (
                dataset.labels.iloc[pointer : pointer + len(product_ids)]["concern"].tolist()
            )
            pointer += len(product_ids)
            for pid, concern, label, prob in zip(product_ids, concerns, labels_np, probs):
                rows.append(
                    {
                        "product_id": int(pid),
                        "product_name": dataset.products.loc[int(pid)]["title_text"],
                        "concern": concern,
                        "label": int(label),
                        "probability": float(prob),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prediction scores for BT-BERT.")
    parser.add_argument(
        "--config",
        default="bt_bert_model/config.yaml",
        help="Configuration YAML path.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path (.pt).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to score.",
    )
    parser.add_argument(
        "--output",
        default="bt_bert_model/outputs/bt_bert_outputs.csv",
        help="CSV file to write predictions.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent / checkpoint_path).resolve()
    df = generate_predictions(
        checkpoint=checkpoint_path,
        config_path=config_path,
        split=args.split,
        device=device,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
