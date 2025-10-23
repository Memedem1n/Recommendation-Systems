"""Augment training data with high-confidence model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def load_products(products_path: Path) -> pd.Series:
    """Return a mapping from product_id to category name."""
    products_df = pd.read_csv(products_path, usecols=["product_id", "category"])
    products_df = products_df.drop_duplicates(subset="product_id")
    products_df = products_df.set_index("product_id")
    return products_df["category"]


def select_candidates(
    predictions: pd.DataFrame,
    threshold: float,
    existing_pairs: Iterable[Tuple[int, str]],
) -> pd.DataFrame:
    """Pick confident positive and negative pairs based on probability thresholds."""
    required_cols = {"product_id", "concern", "probability"}
    missing_cols = required_cols.difference(predictions.columns)
    if missing_cols:
        raise ValueError(f"Prediction file missing columns: {sorted(missing_cols)}")

    if not 0 <= threshold["positive"] <= 1:
        raise ValueError("Positive threshold must be between 0 and 1.")
    if not 0 <= threshold["negative"] <= 1:
        raise ValueError("Negative threshold must be between 0 and 1.")

    filtered = predictions.copy()
    filtered["product_id"] = filtered["product_id"].astype(int)
    filtered["concern"] = filtered["concern"].astype(str)
    filtered["pair"] = list(zip(filtered["product_id"], filtered["concern"]))

    mask_new_pair = filtered["pair"].apply(lambda pair: pair not in existing_pairs)
    filtered = filtered[mask_new_pair]

    pos_candidates = (
        filtered[filtered["probability"] >= threshold["positive"]]
        .drop_duplicates(subset=["product_id", "concern"], keep="first")
        .copy()
    )
    neg_candidates = (
        filtered[filtered["probability"] <= threshold["negative"]]
        .drop_duplicates(subset=["product_id", "concern"], keep="first")
        .copy()
    )

    for df in (pos_candidates, neg_candidates):
        if not df.empty:
            df.drop(columns=["pair"], inplace=True, errors="ignore")

    return pos_candidates, neg_candidates


def build_augmented_rows(
    candidates: pd.DataFrame,
    category_lookup: pd.Series,
    label_value: int,
    source_tag: str,
) -> pd.DataFrame:
    """Format pseudo-labelled rows to match the training CSV schema."""
    if candidates.empty:
        return pd.DataFrame(columns=["product_id", "concern", "label", "sources", "category"])

    categories = category_lookup.reindex(candidates["product_id"]).fillna("")

    rows = pd.DataFrame(
        {
            "product_id": candidates["product_id"].astype(int),
            "concern": candidates["concern"].astype(str),
            "label": label_value,
            "sources": source_tag,
            "category": categories.values,
        }
    )
    return rows


def augment_training_set(
    train_path: Path,
    predictions_path: Path,
    products_path: Path,
    pos_threshold: float,
    neg_threshold: float,
    output_path: Path,
) -> dict:
    """Create an augmented training CSV and return summary statistics."""
    train_df = pd.read_csv(train_path)
    predictions_df = pd.read_csv(predictions_path)
    category_lookup = load_products(products_path)

    existing_pairs = set(zip(train_df["product_id"], train_df["concern"]))
    pos_candidates, neg_candidates = select_candidates(
        predictions_df,
        {"positive": pos_threshold, "negative": neg_threshold},
        existing_pairs,
    )
    pos_rows = build_augmented_rows(
        pos_candidates,
        category_lookup,
        label_value=1,
        source_tag=f"pseudo:prob>={pos_threshold:.2f}",
    )
    neg_rows = build_augmented_rows(
        neg_candidates,
        category_lookup,
        label_value=0,
        source_tag=f"pseudo:prob<={neg_threshold:.2f}",
    )
    augment_rows = pd.concat([pos_rows, neg_rows], ignore_index=True)

    if augment_rows.empty:
        output_df = train_df.copy()
    else:
        output_df = pd.concat([train_df, augment_rows], ignore_index=True)
        output_df = output_df.drop_duplicates(subset=["product_id", "concern"], keep="first")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return {
        "train_rows_before": len(train_df),
        "train_rows_after": len(output_df),
        "added_positive": len(pos_rows),
        "added_negative": len(neg_rows),
        "positive_threshold": pos_threshold,
        "negative_threshold": neg_threshold,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment BT-BERT training data with high-confidence pseudo labels."
    )
    parser.add_argument("--train", required=True, help="Path to existing train.csv.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV with model predictions (requires product_id, concern, probability columns).",
    )
    parser.add_argument(
        "--products",
        required=True,
        help="Path to unified_products.csv (used to recover category information).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum probability to treat a prediction as positive (default: 0.8).",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=0.2,
        help="Maximum probability to treat a prediction as negative (default: 0.2).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV for the augmented training data.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="If set, replace the original train.csv after writing the augmented file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = augment_training_set(
        train_path=Path(args.train),
        predictions_path=Path(args.predictions),
        products_path=Path(args.products),
        pos_threshold=args.threshold,
        neg_threshold=args.negative_threshold,
        output_path=Path(args.output),
    )

    if args.replace:
        Path(args.output).replace(Path(args.train))

    print(
        "Augmentation complete:\n"
        f"  Added positives: {summary['added_positive']}\n"
        f"  Added negatives: {summary['added_negative']}\n"
        f"  Train rows before: {summary['train_rows_before']}\n"
        f"  Train rows after: {summary['train_rows_after']}\n"
        f"  Thresholds -> pos: {summary['positive_threshold']:.2f}, "
        f"neg: {summary['negative_threshold']:.2f}\n"
        f"  Output written to: {summary['output_path']}"
    )


if __name__ == "__main__":
    main()
