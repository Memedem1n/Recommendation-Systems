"""PyTorch dataset utilities for BT-BERT training."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from Dataset_Pipeline import data_pipeline
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Dataset_Pipeline package not found. Please ensure the repository root is on PYTHONPATH."
    ) from exc


def load_products(path: Path) -> pd.DataFrame:
    converters = {"ingredients_list_normalised": ast.literal_eval}
    df = pd.read_csv(path, converters=converters)
    def normalise_items(items: object) -> List[str]:
        if isinstance(items, (list, tuple)):
            iterable = items
        else:
            return []
        return [str(item) for item in iterable if item]

    df["ingredients_list"] = df["ingredients_list_normalised"].apply(normalise_items)
    df["ingredients_text"] = df["ingredients_list"].apply(
        lambda items: ", ".join(items)
    )
    df["category_norm"] = df["category"].apply(
        lambda value: data_pipeline.normalise_category(value) if isinstance(value, str) else ""
    )
    df["title_text"] = df["name"].fillna("")
    return df.set_index("product_id")


class BTBertDataset(Dataset):
    """Dataset that yields tokenized inputs for (product, concern) pairs."""

    def __init__(
        self,
        labels_csv: Path,
        products_csv: Path,
        tokenizer_name: str,
        max_length: int = 512,
        text_options: Optional[Dict[str, object]] = None,
        concern_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.labels = pd.read_csv(labels_csv)
        if "product_id" not in self.labels:
            raise ValueError("labels file must contain 'product_id' column")
        self.products = load_products(products_csv)
        self.text_options = text_options or {}
        concern_weights = concern_weights or {}
        self.concern_weights: Dict[str, float] = {
            str(key): float(value)
            for key, value in concern_weights.items()
        }

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def _format_ingredients(self, ingredients: Iterable[str]) -> List[str]:
        opts = self.text_options
        tokens = list(ingredients)

        exclude_tokens = {
            str(token).strip().lower()
            for token in opts.get("exclude_ingredients", []) or []
            if str(token).strip()
        }
        if exclude_tokens:
            tokens = [
                token
                for token in tokens
                if token.strip() and token.strip().lower() not in exclude_tokens
            ]

        top_k = opts.get("ingredient_top_k")
        if isinstance(top_k, int) and top_k > 0:
            tokens = tokens[:top_k]

        if opts.get("use_rank_tokens"):
            prefix_template = opts.get("rank_token_template", "[INGR_{idx:02d}]")
            tokens = [
                f"{prefix_template.format(idx=idx)} {token}"
                for idx, token in enumerate(tokens, start=1)
            ]

        repeat_top_n = int(opts.get("repeat_top_n", 0) or 0)
        repeat_factor = int(opts.get("repeat_factor", 1) or 1)
        if repeat_top_n > 0 and repeat_factor > 1 and tokens:
            emphasised: List[str] = []
            for idx, token in enumerate(tokens):
                emphasised.append(token)
                if idx < repeat_top_n:
                    emphasised.extend([token] * (repeat_factor - 1))
            tokens = emphasised

        return tokens

    def _build_product_text(self, product_row: pd.Series, concern: str) -> str:
        opts = self.text_options
        ingredients_tokens = self._format_ingredients(product_row["ingredients_list"])
        separator = str(opts.get("ingredient_separator", ", "))
        ingredients_text = separator.join(ingredients_tokens)

        pieces: List[str] = []

        if opts.get("include_concern_prompt"):
            concern_text = concern.replace("_", " ")
            pieces.append(f"Concern: {concern_text}.")

        if opts.get("include_category"):
            category = str(product_row.get("category_norm") or product_row.get("category") or "").strip()
            if category:
                pieces.append(f"Category: {category}.")

        if opts.get("include_product_name", True):
            title = str(product_row.get("title_text", "")).strip()
            if title:
                pieces.append(f"Product: {title}.")

        if ingredients_text:
            pieces.append(f"Ingredients: {ingredients_text}.")
        else:
            fallback = str(product_row.get("ingredients_text", "")).strip()
            if fallback:
                pieces.append(f"Ingredients: {fallback}.")

        extra_context = opts.get("extra_context")
        if isinstance(extra_context, str) and extra_context.strip():
            pieces.append(extra_context.strip())

        return " ".join(pieces).strip()

    def _encode(
        self, concern: str, product_text: str
    ) -> Dict[str, torch.Tensor]:
        concern_text = concern.replace("_", " ")

        encoded = self.tokenizer(
            concern_text,
            text_pair=product_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def build_product_text(self, product_id: int, concern: str) -> str:
        try:
            product_row = self.products.loc[product_id]
        except KeyError as exc:
            raise KeyError(f"Product {product_id} not found in products dataset") from exc
        return self._build_product_text(product_row, concern)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.labels.iloc[idx]
        product_id = int(row["product_id"])
        concern = str(row["concern"])
        label = float(row["label"])

        try:
            product_row = self.products.loc[product_id]
        except KeyError as exc:
            raise KeyError(f"Product {product_id} not found in products dataset") from exc

        product_text = self._build_product_text(product_row, concern)

        encoded = self._encode(
            concern=concern,
            product_text=product_text,
        )
        encoded["labels"] = torch.tensor(label, dtype=torch.float)
        encoded["product_id"] = torch.tensor(product_id, dtype=torch.long)
        weight = self.concern_weights.get(concern, 1.0)
        encoded["sample_weight"] = torch.tensor(float(weight), dtype=torch.float)
        return encoded
