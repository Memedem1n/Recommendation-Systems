"""Data preparation utilities for the BlueSense recommendation pipeline."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


CATEGORY_MAP = {
    "antiaging": "anti aging",
    "anti aging": "anti aging",
    "aroundeyecream": "around eye cream",
    "around eye cream": "around eye cream",
    "bbcream": "bb cream",
    "bb cream": "bb cream",
    "cccream": "cc cream",
    "cc cream": "cc cream",
    "facialcleanser": "facial cleanser",
    "facial cleanser": "facial cleanser",
    "facialmoisturizertreatment": "facial moisturizer treatment",
    "facial moisturizer treatment": "facial moisturizer treatment",
    "mask": "mask",
    "makeupremover": "makeup remover",
    "makeup remover": "makeup remover",
    "serumsessences": "serums essences",
    "serums essences": "serums essences",
    "skinfadinglightener": "skin fading lightener",
    "skin fading lightener": "skin fading lightener",
    "tonersastringents": "toners astringents",
    "toners astringents": "toners astringents",
    "porestrips": "pore strips",
    "pore strips": "pore strips",
    "oilcontroller": "oil controller",
    "oil controller": "oil controller",
}

WATER_WORDS = {"water", "aqua", "agua", "eau"}

INGREDIENT_SYNONYM_MAP = {
    "glycerine": "glycerin",
    "d alpha tocopherol": "tocopherol",
    "dl alpha tocopherol": "tocopherol",
    "d alpha tocopherol acetate": "tocopheryl acetate",
    "dl alpha tocopherol acetate": "tocopheryl acetate",
    "vitamin e": "tocopherol",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file treating every column as string for consistency."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["source_file"] = path.name
    return df


def load_datasets(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files under the dataset directory."""
    csv_paths = sorted(dataset_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    dataframes: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_paths:
        dataframes[csv_path.stem] = _read_csv(csv_path)
    return dataframes


def harmonise_columns(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Ensure every dataframe shares the same column set."""
    all_columns = sorted({col for df in frames.values() for col in df.columns})
    aligned: Dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        missing = [col for col in all_columns if col not in df.columns]
        if missing:
            for col in missing:
                df[col] = ""
        aligned[name] = df[all_columns]
    return aligned


def combine_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames while resetting index."""
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.reindex(sorted(combined.columns), axis=1)
    return combined


def normalise_category(value: str) -> str:
    """Return a lowercase, punctuation-free category label."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    key = value.replace(" ", "")
    return CATEGORY_MAP.get(value, CATEGORY_MAP.get(key, value))


def rebuild_product_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with a fresh zero-based product_id column."""
    df = df.copy()
    df.insert(0, "product_id", range(len(df)))
    return df


SPLIT_PATTERN = re.compile(r"\s*[;,]\s*")
TAG_PATTERN = re.compile(r"\[[^\]]*\]")
NON_ALNUM_PATTERN = re.compile(r"[^0-9a-zA-Z\-\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
DIGIT_PATTERN = re.compile(r"\d+")


def split_ingredients(cell: str) -> List[str]:
    """Split a raw ingredient cell into distinct tokens."""
    if not cell:
        return []
    cell = cell.replace("\n", " ")
    parts = [part.strip() for part in SPLIT_PATTERN.split(cell) if part.strip()]
    return parts


def normalise_token(token: str) -> str:
    """Standardise a single ingredient token."""
    token = token.strip()
    token = TAG_PATTERN.sub("", token)
    token = token.replace("*", "")
    token = token.replace("&", " and ")
    token = unicodedata.normalize("NFKD", token)
    token = token.encode("ascii", "ignore").decode("ascii")
    token = token.lower()
    token = NON_ALNUM_PATTERN.sub(" ", token)
    token = MULTISPACE_PATTERN.sub(" ", token)
    token = token.strip()
    return token


def apply_manual_synonym(token: str) -> str:
    """Return a manual canonical form for strongly related ingredients."""
    if not token:
        return token
    words = token.split()
    if words and all(word in WATER_WORDS for word in words):
        return "water"
    if token in INGREDIENT_SYNONYM_MAP:
        return INGREDIENT_SYNONYM_MAP[token]
    return token


def bucket_key(value: str) -> str:
    """Create a short bucket key to limit fuzzy comparisons."""
    compact = "".join(ch for ch in value if ch.isalnum())
    length_band = len(compact) // 3
    return f"{compact[:3]}_{length_band}" if compact else f"{value[:3]}_{length_band}"


def numeric_signature(value: str) -> Tuple[str, ...]:
    """Extract the ordered digit sequence within a token."""
    return tuple(DIGIT_PATTERN.findall(value))


def fuzzy_match(a: str, b: str) -> float:
    """Return the similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def standardise_ingredient_map(
    mapping_df: pd.DataFrame, similarity_threshold: float = 0.85
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Group ingredients that are highly similar into canonical labels."""
    mapping_df = mapping_df.copy()
    token_counts = mapping_df['normalised'].value_counts()
    unique_tokens = sorted(mapping_df['normalised'].dropna().unique())

    manual_lookup: Dict[str, str] = {}
    auto_tokens: List[str] = []
    for token in unique_tokens:
        manual = apply_manual_synonym(token)
        if manual != token:
            manual_lookup[token] = manual
        else:
            auto_tokens.append(token)

    groups: Dict[str, List[Dict[str, List[str]]]] = defaultdict(list)
    for token in auto_tokens:
        key = bucket_key(token)
        bucket_groups = groups[key]
        matched_group = None
        for group in bucket_groups:
            representative = group['rep']
            if abs(len(token) - len(representative)) > 3:
                continue
            token_signature = numeric_signature(token)
            rep_signature = numeric_signature(representative)
            if (token_signature or rep_signature) and token_signature != rep_signature:
                continue
            if fuzzy_match(token, representative) >= similarity_threshold:
                matched_group = group
                break
        if matched_group is None:
            bucket_groups.append({'rep': token, 'tokens': [token]})
            continue
        matched_group['tokens'].append(token)
        rep = matched_group['rep']
        if token_counts[token] > token_counts[rep] or (token_counts[token] == token_counts[rep] and token < rep):
            matched_group['rep'] = token

    canonical_lookup: Dict[str, str] = {}
    for bucket_groups in groups.values():
        for group in bucket_groups:
            representative = group['rep']
            for token in group['tokens']:
                canonical_lookup[token] = representative

    canonical_lookup.update(manual_lookup)
    for token in unique_tokens:
        canonical_lookup.setdefault(token, token)

    mapping_df['canonical'] = mapping_df['normalised'].map(canonical_lookup).fillna(mapping_df['normalised'])
    mapping_df = mapping_df.drop_duplicates(subset=['product_id', 'original', 'canonical']).reset_index(drop=True)
    return mapping_df, canonical_lookup


def apply_canonical_to_products(
    products_df: pd.DataFrame, canonical_lookup: Dict[str, str]
) -> pd.DataFrame:
    """Replace normalised ingredient lists with their canonical forms."""
    products_df = products_df.copy()
    canonical_lists: List[List[str]] = []
    canonical_strings: List[str] = []
    counts: List[int] = []

    for tokens in products_df.get("ingredients_list_normalised", []):
        canonical_tokens = [canonical_lookup.get(token, token) for token in tokens]
        canonical_lists.append(canonical_tokens)
        canonical_strings.append(" | ".join(canonical_tokens))
        counts.append(len(canonical_tokens))

    products_df["ingredients_list_normalised"] = canonical_lists
    products_df["ingredients_normalised"] = canonical_strings
    products_df["ingredient_count"] = counts
    return products_df


def build_ingredient_lists(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create normalised ingredient lists and the mapping dataframe."""
    df = df.copy()
    raw_tokens_column: List[List[str]] = []
    norm_tokens_column: List[List[str]] = []
    mapping_records: List[Tuple[int, str, str]] = []

    for row in df.itertuples(index=False):
        raw_cell = getattr(row, "ingredients_raw", "")
        product_id = getattr(row, "product_id")
        tokens_raw = split_ingredients(raw_cell)
        tokens_norm: List[str] = []
        for raw in tokens_raw:
            normalised = normalise_token(raw)
            if normalised:
                tokens_norm.append(normalised)
                mapping_records.append((product_id, raw, normalised))
        raw_tokens_column.append(tokens_raw)
        norm_tokens_column.append(tokens_norm)

    df["ingredients_list_raw"] = raw_tokens_column
    df["ingredients_list_normalised"] = norm_tokens_column
    df["ingredients_normalised"] = [" | ".join(tokens) for tokens in norm_tokens_column]
    df["ingredient_count"] = [len(tokens) for tokens in norm_tokens_column]

    mapping_df = pd.DataFrame(mapping_records, columns=["product_id", "original", "normalised"])
    mapping_df = mapping_df.drop_duplicates().reset_index(drop=True)
    return df, mapping_df


@dataclass
class PipelineOutputs:
    products: pd.DataFrame
    ingredient_map: pd.DataFrame
    unique_ingredients: pd.DataFrame


def run_pipeline(dataset_dir: Path, output_dir: Path) -> PipelineOutputs:
    """Execute the full pipeline and persist outputs."""
    frames = load_datasets(dataset_dir)
    frames = harmonise_columns(frames)
    for frame in frames.values():
        if "category" in frame.columns:
            frame["category"] = frame["category"].apply(normalise_category)

    combined = combine_frames(frames.values())

    if "id" in combined.columns:
        combined = combined.drop(columns=["id"])

    combined = combined.rename(columns={"ingredients": "ingredients_raw"})

    if "category" in combined.columns:
        combined["category"] = combined["category"].apply(normalise_category)

    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined = rebuild_product_ids(combined)

    combined, mapping_df = build_ingredient_lists(combined)
    mapping_df, canonical_lookup = standardise_ingredient_map(mapping_df)
    combined = apply_canonical_to_products(combined, canonical_lookup)

    unique_ingredients = (
        mapping_df.groupby("canonical")
        .agg(
            product_frequency=("product_id", "nunique"),
            variant_count=("normalised", "nunique"),
        )
        .sort_values(["product_frequency", "canonical"], ascending=[False, True])
        .reset_index()
        .rename(columns={"canonical": "ingredient"})
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    products_path = output_dir / "unified_products.csv"
    mapping_path = output_dir / "ingredient_normalisation_map.csv"
    unique_path = output_dir / "unique_ingredients.csv"
    combined.to_csv(products_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)
    unique_ingredients.to_csv(unique_path, index=False)

    return PipelineOutputs(
        products=combined,
        ingredient_map=mapping_df,
        unique_ingredients=unique_ingredients,
    )
