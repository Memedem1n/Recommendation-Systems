"""Utilities for preparing pseudo labels and dataset splits for BT-BERT."""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

try:
    from Dataset_Pipeline import data_pipeline
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Dataset_Pipeline package not found. Please ensure the repository root is on PYTHONPATH."
    ) from exc


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalise_category(value: str) -> str:
    return data_pipeline.normalise_category(value) if value else ""


def load_products(path: Path) -> pd.DataFrame:
    converters = {"ingredients_list_normalised": ast.literal_eval}
    df = pd.read_csv(path, converters=converters)
    df["category_norm"] = df["category"].apply(normalise_category)
    df["ingredients_set"] = (
        df["ingredients_list_normalised"]
        .apply(
            lambda values: {
                data_pipeline.normalise_token(str(v)) for v in values if v
            }
        )
        .apply(lambda values: {v for v in values if v})
    )
    df["ingredients_text"] = df["ingredients_list_normalised"].apply(
        lambda items: ", ".join(str(item) for item in items if item)
    )
    df["title_text"] = df["name"].fillna("")
    df["combined_text_lower"] = (
        df["title_text"].fillna("").astype(str)
        + " "
        + df["ingredients_text"].fillna("").astype(str)
    ).str.lower()
    return df


def match_ingredients(
    ingredients: Set[str], mapping: Dict[str, Iterable[str]]
) -> Set[str]:
    matched: Set[str] = set()
    for key, concerns in mapping.items():
        key_norm = data_pipeline.normalise_token(str(key))
        if key_norm in ingredients:
            matched.update(concerns)
    return matched


def normalise_category_mapping(mapping: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    normalised = {}
    for key, concerns in mapping.items():
        normalised[data_pipeline.normalise_category(key)] = list(concerns)
    return normalised


def normalise_ingredient_mapping(mapping: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    normalised = {}
    for key, concerns in mapping.items():
        normalised[data_pipeline.normalise_token(str(key))] = list(concerns)
    return normalised


def normalise_keyword_mapping(mapping: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    normalised: Dict[str, List[str]] = {}
    for key, concerns in mapping.items():
        token = str(key).strip().lower()
        if not token:
            continue
        normalised[token] = [str(c).strip() for c in concerns]
    return normalised


def build_labels(
    products: pd.DataFrame,
    concerns: List[str],
    category_positive_map: Dict[str, Iterable[str]],
    category_negative_map: Dict[str, Iterable[str]],
    ingredient_positive_map: Dict[str, Iterable[str]],
    ingredient_negative_map: Dict[str, Iterable[str]],
    keyword_positive_map: Dict[str, Iterable[str]],
    keyword_negative_map: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    records: List[dict] = []
    all_concerns = set(concerns)

    for row in products.itertuples():
        product_id = getattr(row, "product_id")
        category = getattr(row, "category_norm", "")
        ingredients = getattr(row, "ingredients_set", set())

        label_values = {concern: 0 for concern in concerns}
        label_sources: Dict[str, Set[str]] = defaultdict(set)

        # Category positives
        if category in category_positive_map:
            positives = [
                concern for concern in category_positive_map[category] if concern in all_concerns
            ]
            for concern in positives:
                label_values[concern] = 1
                label_sources[concern].add(f"category:{category}")

        # Ingredient positives
        positives_from_ingredients = match_ingredients(ingredients, ingredient_positive_map)
        for concern in positives_from_ingredients:
            if concern in all_concerns:
                label_values[concern] = 1
                label_sources[concern].add("ingredient")

        # Ingredient negatives override positives when explicit
        negatives_from_ingredients = match_ingredients(ingredients, ingredient_negative_map)
        for concern in negatives_from_ingredients:
            if concern in all_concerns:
                label_values[concern] = 0
                label_sources[concern].add("ingredient_negative")

        # Category negatives (only if no positive source other than negative)
        if category in category_negative_map:
            for concern in category_negative_map[category]:
                if concern in all_concerns:
                    label_values[concern] = 0
                    label_sources[concern].add(f"category_negative:{category}")

        combined_text = str(getattr(row, "combined_text_lower", "")).lower()

        for keyword, mapped_concerns in keyword_positive_map.items():
            if keyword and keyword in combined_text:
                for concern in mapped_concerns:
                    if concern in all_concerns:
                        label_values[concern] = 1
                        label_sources[concern].add(f"keyword:{keyword}")

        for keyword, mapped_concerns in keyword_negative_map.items():
            if keyword and keyword in combined_text:
                for concern in mapped_concerns:
                    if concern in all_concerns:
                        label_values[concern] = 0
                        label_sources[concern].add(f"keyword_negative:{keyword}")

        for concern in concerns:
            source_list = sorted(label_sources.get(concern, []))
            records.append(
                {
                    "product_id": product_id,
                    "concern": concern,
                    "label": int(label_values[concern]),
                    "sources": "|".join(source_list) if source_list else "",
                    "category": category,
                }
            )

    return pd.DataFrame(records)


def create_splits(
    labels_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert abs(train_ratio + val_ratio) < 1

    train_val_df, test_df = train_test_split(
        labels_df,
        test_size=1 - (train_ratio + val_ratio),
        random_state=seed,
        stratify=labels_df["label"],
    )
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def summarise_labels(labels_df: pd.DataFrame, output_path: Path) -> None:
    summary = {}
    grouped = labels_df.groupby(["concern", "label"]).size().unstack(fill_value=0)
    for concern, counts in grouped.iterrows():
        summary[concern] = {
            "positive": int(counts.get(1, 0)),
            "negative": int(counts.get(0, 0)),
        }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    random.seed(config["project"]["random_seed"])
    np.random.seed(config["project"]["random_seed"])

    base_dir = (config_path.parent / config["project"]["base_dir"]).resolve()
    data_cfg = config["data"]

    products_path = (base_dir / data_cfg["unified_products_path"]).resolve()
    products_df = load_products(products_path)
    category_positive_map = normalise_category_mapping(
        config.get("category_positive_map", {})
    )
    category_negative_map = normalise_category_mapping(
        config.get("category_negative_map", {})
    )
    ingredient_positive_map = normalise_ingredient_mapping(
        config.get("ingredient_positive_map", {})
    )
    ingredient_negative_map = normalise_ingredient_mapping(
        config.get("ingredient_negative_map", {})
    )
    keyword_positive_map = normalise_keyword_mapping(
        config.get("keyword_positive_map", {})
    )
    keyword_negative_map = normalise_keyword_mapping(
        config.get("keyword_negative_map", {})
    )

    labels_df = build_labels(
        products=products_df,
        concerns=config["concerns"],
        category_positive_map=category_positive_map,
        category_negative_map=category_negative_map,
        ingredient_positive_map=ingredient_positive_map,
        ingredient_negative_map=ingredient_negative_map,
        keyword_positive_map=keyword_positive_map,
        keyword_negative_map=keyword_negative_map,
    )

    label_dir = (base_dir / data_cfg["label_output_dir"]).resolve()
    label_dir.mkdir(parents=True, exist_ok=True)
    labels_path = label_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)

    splits = create_splits(
        labels_df=labels_df,
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
        seed=config["project"]["random_seed"],
    )

    for split_name, df in splits.items():
        split_path = label_dir / f"{split_name}.csv"
        df.to_csv(split_path, index=False)

    summary_path = label_dir / "label_summary.json"
    summarise_labels(labels_df, summary_path)

    print(f"Label table written -> {labels_path}")
    for split_name in ("train", "val", "test"):
        path = label_dir / f"{split_name}.csv"
        print(f"{split_name.title()} split -> {path} ({len(splits[split_name])} rows)")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare pseudo labels for BT-BERT.")
    parser.add_argument(
        "--config",
        default="bt_bert_model/config.yaml",
        help="Path to configuration YAML file.",
    )
    main(parser.parse_args())
