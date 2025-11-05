from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import ServicePaths, ServiceSettings

# Ensure training source modules are importable without altering repository layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "bt_bert_model" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import load_products  # type: ignore[import]  # noqa: E402
from model import BTBertConfig, BTBertModel  # type: ignore[import]  # noqa: E402
from train import load_yaml  # type: ignore[import]  # noqa: E402
from Dataset_Pipeline import data_pipeline  # type: ignore[import]  # noqa: E402


def _resolve_device(device_spec: str) -> torch.device:
    device_spec = device_spec.lower()
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_spec)


def _sigmoid_logits(logits: Tensor) -> np.ndarray:
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


@dataclass
class RecommendationResult:
    product_id: int
    probability: float
    name: str
    category: str
    category_norm: str
    product_url: Optional[str]
    ingredients: List[str]
    ingredient_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "product_id": self.product_id,
            "probability": self.probability,
            "name": self.name,
            "category": self.category,
            "category_norm": self.category_norm,
            "product_url": self.product_url,
            "ingredients": self.ingredients,
            "ingredient_count": self.ingredient_count,
        }


class ProductTextBuilder:
    """Utility that mirrors BTBertDataset text construction for inference."""

    def __init__(
        self,
        products: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_options: Optional[Dict[str, object]],
        max_length: int,
    ) -> None:
        self.products = products
        self.tokenizer = tokenizer
        self.text_options = text_options or {}
        self.max_length = max_length

    def build_text(self, product_id: int, concern: str) -> str:
        try:
            product_row = self.products.loc[product_id]
        except KeyError as exc:
            raise KeyError(f"Product {product_id} not found in products dataset") from exc
        return self._build_product_text(product_row, concern)

    def encode(self, product_id: int, concern: str) -> Dict[str, Tensor]:
        product_text = self.build_text(product_id, concern)
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

    def _format_ingredients(self, ingredients: Iterable[str]) -> List[str]:
        opts = self.text_options
        tokens = [token for token in ingredients if token]

        exclude_tokens = {
            str(token).strip().lower()
            for token in (opts.get("exclude_ingredients") or [])
            if str(token).strip()
        }
        if exclude_tokens:
            tokens = [
                token
                for token in tokens
                if token.strip().lower() not in exclude_tokens
            ]

        top_k = opts.get("ingredient_top_k")
        if isinstance(top_k, int) and top_k > 0:
            tokens = tokens[:top_k]

        if opts.get("use_rank_tokens"):
            template = opts.get("rank_token_template", "[INGR_{idx:02d}]")
            tokens = [
                f"{template.format(idx=idx)} {token}"
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
        ingredients_tokens = self._format_ingredients(product_row.get("ingredients_list", []))
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
            title = str(product_row.get("title_text") or product_row.get("name") or "").strip()
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


class BTBertRecommender:
    """Wrapper that exposes inference helpers for the fine-tuned BT-BERT model."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.device = _resolve_device(settings.device)
        self.paths: ServicePaths = settings.resolve_paths()
        self._lock = RLock()
        self._scores_cache: Dict[str, pd.DataFrame] = {}
        self._load_config()
        self._load_model()
        self._load_products(self.paths.products_csv)
        if settings.preload_scores:
            for concern in self.concerns:
                self._score_concern(concern)

    def _load_config(self) -> None:
        cfg = load_yaml(self.paths.config_path)
        self.config = cfg
        self.concerns: List[str] = [str(c) for c in cfg.get("concerns", [])]
        if not self.concerns:
            raise ValueError("No concerns defined in configuration file.")
        tokenizer_cfg = cfg.get("tokenizer") or {}
        self.tokenizer_name = tokenizer_cfg.get("pretrained_name", "bert-base-uncased")
        self.max_length = int(tokenizer_cfg.get("max_length", 512))
        self.text_options = cfg.get("text_options") or {}
        model_cfg = cfg.get("model") or {}
        self.model_config = BTBertConfig(
            pretrained_model_name=model_cfg.get("pretrained_name", self.tokenizer_name),
            attention_scale=float(model_cfg.get("attention_scale", 16.0)),
        )

    def _load_model(self) -> None:
        checkpoint_path = self.paths.checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                "Set BLUESENSE_CHECKPOINT_PATH to the fine-tuned model state_dict."
            )
        self.model = BTBertModel(self.model_config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.tokenizer_name
        )

    def _load_products(self, products_csv: Path) -> None:
        if not products_csv.exists():
            raise FileNotFoundError(
                f"Products CSV not found at {products_csv}. "
                "Run the dataset pipeline or update BLUESENSE_PRODUCTS_CSV."
            )
        df = load_products(products_csv)
        df = df.copy()
        df["category_norm"] = df["category"].apply(
            lambda value: data_pipeline.normalise_category(value) if isinstance(value, str) else ""
        )
        self.products = df
        self.text_builder = ProductTextBuilder(
            products=df,
            tokenizer=self.tokenizer,
            text_options=self.text_options,
            max_length=self.max_length,
        )
        self._scores_cache.clear()
        self.paths = self.paths.copy(update={"products_csv": products_csv})

    def available_concerns(self) -> List[str]:
        return list(self.concerns)

    def products_count(self) -> int:
        return int(len(self.products))

    def canonicalise_concern(self, concern: str) -> str:
        return self._validate_concern(concern)

    def refresh_products(self, products_csv: Optional[Path] = None) -> None:
        with self._lock:
            target_path = Path(products_csv) if products_csv else self.paths.products_csv
            self._load_products(target_path)

    def score_products(
        self,
        concern: str,
        product_ids: Sequence[int],
    ) -> List[RecommendationResult]:
        concern = self._validate_concern(concern)
        if not product_ids:
            return []

        scores_df = self._score_concern(concern)
        lookup = {
            int(row.product_id): row
            for row in scores_df.itertuples(index=False)
        }

        results: List[RecommendationResult] = []
        for raw_id in product_ids:
            product_id = int(raw_id)
            row = lookup.get(product_id)
            if row is None:
                raise KeyError(f"Product {product_id} not found in scored results.")
            results.append(
                RecommendationResult(
                    product_id=product_id,
                    probability=float(row.probability),
                    name=str(row.name),
                    category=str(row.category),
                    category_norm=str(row.category_norm),
                    product_url=row.product_url
                    if isinstance(row.product_url, str) and row.product_url
                    else None,
                    ingredients=list(row.ingredients),
                    ingredient_count=int(row.ingredient_count),
                )
            )
        return results

    def recommend_for_concern(
        self,
        concern: str,
        top_k: int = 10,
        category: Optional[str] = None,
        allowed_product_ids: Optional[Sequence[int]] = None,
    ) -> List[RecommendationResult]:
        concern = self._validate_concern(concern)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        scores_df = self._score_concern(concern)
        df = scores_df
        if category:
            norm = data_pipeline.normalise_category(category)
            df = df[df["category_norm"] == norm]
        if allowed_product_ids is not None:
            allowed_set = {int(pid) for pid in allowed_product_ids}
            df = df[df["product_id"].isin(allowed_set)]
        df = df.head(top_k)
        return [
            RecommendationResult(
                product_id=int(row.product_id),
                probability=float(row.probability),
                name=str(row.name),
                category=str(row.category),
                category_norm=str(row.category_norm),
                product_url=row.product_url if isinstance(row.product_url, str) and row.product_url else None,
                ingredients=list(row.ingredients),
                ingredient_count=int(row.ingredient_count),
            )
            for row in df.itertuples(index=False)
        ]

    def _validate_concern(self, concern: str) -> str:
        concern = concern.strip()
        if not concern:
            raise ValueError("concern cannot be empty.")
        norm = concern.replace(" ", "_")
        available = {c: c for c in self.concerns}
        available.update({c.replace("_", " "): c for c in self.concerns})
        if norm in available:
            return available[norm]
        if concern in available:
            return available[concern]
        raise ValueError(
            f"Unknown concern '{concern}'. Valid concerns: {', '.join(self.concerns)}"
        )

    def _score_concern(self, concern: str) -> pd.DataFrame:
        with self._lock:
            if concern in self._scores_cache:
                return self._scores_cache[concern]
            batch_size = self.settings.inference_batch_size
            product_ids = self.products.index.to_list()
            encoded_inputs = [self.text_builder.encode(pid, concern) for pid in product_ids]
            rows: List[Dict[str, object]] = []
            with torch.no_grad():
                for start in range(0, len(product_ids), batch_size):
                    end = start + batch_size
                    batch_inputs = encoded_inputs[start:end]
                    if not batch_inputs:
                        continue
                    stacked = {
                        key: torch.stack([inputs[key] for inputs in batch_inputs], dim=0).to(self.device)
                        for key in batch_inputs[0]
                    }
                    outputs = self.model(**stacked)
                    probs = _sigmoid_logits(outputs["logits"])
                    for idx, probability in enumerate(probs):
                        product_id = product_ids[start + idx]
                        product_row = self.products.loc[product_id]
                        rows.append(
                            {
                                "product_id": int(product_id),
                                "probability": float(probability),
                                "name": str(product_row.get("title_text") or product_row.get("name") or ""),
                                "category": str(product_row.get("category") or ""),
                                "category_norm": str(product_row.get("category_norm") or ""),
                                "product_url": str(product_row.get("product_url") or "") or None,
                                "ingredients": list(product_row.get("ingredients_list") or []),
                                "ingredient_count": int(product_row.get("ingredient_count") or 0),
                            }
                        )
            df = pd.DataFrame(rows)
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)
            self._scores_cache[concern] = df
            return df
