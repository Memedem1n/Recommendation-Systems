"""Attention-based explanation generator for BT-BERT predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dataset import BTBertDataset  # noqa: E402
from model import BTBertConfig, BTBertModel  # noqa: E402
from train import load_yaml  # noqa: E402


def extract_top_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    attentions: torch.Tensor,
    top_k: int = 10,
) -> List[str]:
    # attentions: (heads, seq_len, seq_len) for single example (already summed over batch)
    heads, seq_len, _ = attentions.shape
    flattened = attentions.view(heads * seq_len, seq_len)
    scores, indices = torch.topk(flattened, k=min(top_k, flattened.numel()), dim=1)
    top_indices = indices.flatten().unique()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[top_indices])
    filtered = [
        tok for tok in tokens if tok not in {"[CLS]", "[SEP]", "[PAD]", ","} and not tok.startswith("##.")
    ]
    return filtered[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attention-based explanations.")
    parser.add_argument(
        "--config",
        default="bt_bert_model/config.yaml",
        help="Configuration YAML path.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint (.pt) to load.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of product-concern pairs to explain.",
    )
    parser.add_argument(
        "--output",
        default="bt_bert_model/outputs/attention_reports/explanations.json",
        help="Path to write explanation JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    base_dir = (config_path.parent / cfg["project"]["base_dir"]).resolve()
    products_path = (base_dir / cfg["data"]["unified_products_path"]).resolve()
    labels_dir = (base_dir / cfg["data"]["label_output_dir"]).resolve()
    split_path = labels_dir / f"{args.split}.csv"

    dataset = BTBertDataset(
        labels_csv=split_path,
        products_csv=products_path,
        tokenizer_name=cfg["tokenizer"]["pretrained_name"],
        max_length=cfg["tokenizer"]["max_length"],
        text_options=cfg.get("text_options", {}),
    )
    labels_df = dataset.labels.copy()
    sample_size = min(args.sample, len(labels_df))
    if sample_size and sample_size < len(labels_df):
        labels_df = labels_df.sample(
            n=sample_size, random_state=cfg["project"]["random_seed"]
        )

    tokenizer = dataset.tokenizer
    model = BTBertModel(
        BTBertConfig(
            pretrained_model_name=cfg["model"]["pretrained_name"],
            attention_scale=cfg["model"]["attention_scale"],
        )
    )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent / checkpoint_path).resolve()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    with torch.no_grad():
        for row in labels_df.itertuples():
            product_id = int(row.product_id)
            concern = str(row.concern)
            label = int(row.label)
            product = dataset.products.loc[product_id]
            concern_text = concern.replace("_", " ")
            text_pair = dataset.build_product_text(product_id, concern)
            encoded = tokenizer(
                concern_text,
                text_pair=text_pair,
                truncation=True,
                padding="max_length",
                max_length=cfg["tokenizer"]["max_length"],
                return_tensors="pt",
            )
            outputs = model.encoder(
                **encoded, output_attentions=True, return_dict=True
            )
            attentions = outputs.attentions[-1][0]  # (heads, seq, seq)
            tokens = extract_top_tokens(
                tokenizer=tokenizer,
                input_ids=encoded["input_ids"][0],
                attentions=attentions,
                top_k=10,
            )
            results.append(
                {
                    "product_id": product_id,
                    "concern": concern,
                    "label": label,
                    "predicted_logit": float(
                        cfg["model"]["attention_scale"]
                        * outputs.attentions[-1][:, :, 0, 0].sum(dim=1)[0].item()
                    ),
                    "top_tokens": tokens,
                    "title": product["title_text"],
                }
            )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} explanations -> {output_path}")


if __name__ == "__main__":
    main()
