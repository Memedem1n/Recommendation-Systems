#!/usr/bin/env python3
"""Offline feature generation pipeline for the hybrid concern experiments.

This script rebuilds the matrices that the notebook previously expected to find
as precomputed artifacts. It loads the dataset pipeline outputs, expands the
ingredient-to-concern mapping with the same heuristics used in the notebook,
trains the lightweight concern classifier, and finally emits the matrices and
product profiles required by the recommendation step.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset_Pipeline import data_pipeline

LOGGER = logging.getLogger("prepare_features")


def normalise_phrase(text: str) -> str:
    """Normalise free-form ingredient or keyword phrases."""
    if not isinstance(text, str):
        return ""
    return data_pipeline.normalise_token(text)


def extract_concerns_from_text(text: str, concerns: Sequence[str]) -> list[str]:
    """Return concern labels explicitly mentioned in free text."""
    text = (text or "").lower()
    matches = [concern for concern in concerns if concern in text]
    keyword_aliases = {
        "moisturize": "moisture",
        "moisturise": "moisture",
        "eye bag": "eyebag",
        "eye bags": "eyebag",
        "dry eye": "eyebag",
    }
    for alias, concern in keyword_aliases.items():
        if alias in text and concern not in matches:
            matches.append(concern)
    return matches


def split_ingredient_candidates(text: str) -> list[str]:
    """Split a label cell into normalised ingredient tokens."""
    if not isinstance(text, str):
        return []
    filtered = text.replace("/", ",").replace("\\", ",")
    filtered = filtered.replace(" and ", ",").replace(" or ", ",")
    filtered = filtered.replace("(", " ").replace(")", " ")
    parts = [normalise_phrase(part) for part in filtered.split(",")]
    return [part for part in parts if part]


def _normalise_weights(raw: np.ndarray) -> np.ndarray:
    """Normalise positional weight arrays."""
    if raw.size == 0:
        return raw
    total = raw.sum()
    if total <= 0.0:
        return np.zeros_like(raw)
    return raw / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments_update/Hybrid_Concern_Test/config.yaml"),
        help="Path to hybrid concern configuration YAML.",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("Dataset_Pipeline"),
        help="Directory holding unified_products.csv & companions.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Repository root containing ingredient_concern_map.xlsx etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments_update/artifacts/features"),
        help="Destination directory for generated artifacts.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


@dataclass
class ConfigBundle:
    concerns: list[str]
    params: dict
    hints: dict
    scoring: dict


def load_config(path: Path) -> ConfigBundle:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    concerns = cfg.get("concerns")
    if not concerns:
        raise ValueError("Config is missing 'concerns' list.")
    return ConfigBundle(
        concerns=list(concerns),
        params=cfg.get("config", {}),
        hints=cfg.get("hints", {}),
        scoring=cfg.get("scoring", {}),
    )


class FeatureBuilder:
    def __init__(self, cfg: ConfigBundle, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.random_seed = args.random_seed
        self.pipeline_root = args.pipeline_root.resolve()
        self.data_root = args.data_root.resolve()
        self.output_dir = args.output_dir.resolve()
        self.keyword_hints = cfg.hints.get("keyword_hints", {})
        self.category_hints = cfg.hints.get("category_concern_hints", {})
        self.name_hints = cfg.hints.get("name_hints", {})
        self.name_hint_tokens = cfg.hints.get("name_hint_tokens", {})
        self.name_hint_phrases = cfg.hints.get("name_hint_phrases", {})
        strategies = cfg.params.get("weight_strategies", ("harmonic", "log"))
        self.positional_strategies = tuple(strategies)
        self.name_hint_bonus = float(cfg.params.get("name_hint_bonus", 1.0))
        self.rare_signal_coef = float(cfg.scoring.get("rare_signal_coef", 0.35))
        self.concerns = cfg.concerns

    def run(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading pipeline artifacts from %s", self.pipeline_root)
        products, ingredient_map, unique_ingredients = self._load_pipeline_outputs()
        LOGGER.info(
            "Loaded %d products, %d mapping rows, %d unique ingredients",
            len(products),
            len(ingredient_map),
            len(unique_ingredients),
        )
        self.products = products  # type: ignore[attr-defined]

        LOGGER.info("Loading knowledge base assets from %s", self.data_root)
        ingredient_concern_df, about_lookup = self._load_knowledge_sources()

        LOGGER.info("Building ingredient concern candidate dictionary")
        ingredient_concern_candidates = self._build_concern_candidates(
            ingredient_concern_df, about_lookup, unique_ingredients
        )

        LOGGER.info("Evaluating positional weighting strategies: %s", self.positional_strategies)
        ingredient_concern_weights = self._derive_ingredient_weights(
            products, ingredient_concern_candidates
        )

        LOGGER.info("Computing rarity scores for %d ingredients", len(ingredient_concern_weights))
        ingredient_frequency = (
            ingredient_map.groupby("canonical")["product_id"].nunique().to_dict()
        )
        ingredient_rarity = {
            ingredient: float(1.0 / math.log1p(ingredient_frequency.get(ingredient, 1)))
            for ingredient in ingredient_concern_weights
        }

        LOGGER.info("Generating product-level feature table")
        product_profile_df = self._build_product_profiles(
            products, ingredient_concern_weights, ingredient_rarity
        )

        LOGGER.info("Training calibrated logistic classifier for concern prediction")
        self._attach_model_signals(product_profile_df)

        LOGGER.info("Deriving scoring matrices")
        matrices = self._build_matrices(product_profile_df)

        LOGGER.info("Writing artifacts to %s", self.output_dir)
        self._write_outputs(
            product_profile_df,
            ingredient_concern_weights,
            matrices,
            ingredient_concern_candidates,
        )

    # --------------------------------------------------------------------- load

    def _load_pipeline_outputs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        converters = {
            "ingredients_list_raw": ast.literal_eval,
            "ingredients_list_normalised": ast.literal_eval,
        }
        products = pd.read_csv(
            self.pipeline_root / "unified_products.csv",
            usecols=["product_id", "category", "name", "ingredients_list_normalised", "ingredients_list_raw"],
            converters=converters,
        )
        products["ingredients_list_normalised"] = products["ingredients_list_normalised"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        products["ingredients_list_raw"] = products["ingredients_list_raw"].apply(
            lambda value: value if isinstance(value, list) else []
        )

        ingredient_map = pd.read_csv(
            self.pipeline_root / "ingredient_normalisation_map.csv",
            usecols=["product_id", "canonical"],
        )
        unique_ingredients = pd.read_csv(self.pipeline_root / "unique_ingredients.csv")
        return products, ingredient_map, unique_ingredients

    def _load_knowledge_sources(self) -> tuple[pd.DataFrame, dict[str, str]]:
        ingredient_concern_df = pd.read_excel(self.data_root / "ingredient_concern_map.xlsx")
        about_path = self.data_root / "about_ingredients.json"
        about_data = json.loads(about_path.read_text(encoding="utf-8"))
        about_lookup = {
            normalise_phrase(item.get("ingredient")): item.get("description", "")
            for item in about_data
        }
        return ingredient_concern_df, about_lookup

    # -------------------------------------------------------- concern candidates

    def _build_concern_candidates(
        self,
        ingredient_concern_df: pd.DataFrame,
        about_lookup: dict[str, str],
        unique_ingredients: pd.DataFrame,
    ) -> dict[str, set[str]]:
        candidates: dict[str, set[str]] = defaultdict(set)
        for _, row in ingredient_concern_df.iterrows():
            concerns = extract_concerns_from_text(row.get("skin problem"), self.concerns)
            if not concerns:
                continue
            for candidate in split_ingredient_candidates(row.get("ingredients")):
                candidates[candidate].update(concerns)

        missing_after_descriptions = 0
        for canon_name in unique_ingredients["ingredient"]:
            norm = normalise_phrase(canon_name)
            if not norm or candidates.get(norm):
                continue
            description = about_lookup.get(norm, about_lookup.get(canon_name, ""))
            if not isinstance(description, str) or not description.strip():
                missing_after_descriptions += 1
                continue
            desc_norm = description.lower()
            matches = [
                concern
                for concern, keywords in self.keyword_hints.items()
                if any(keyword in desc_norm for keyword in keywords)
            ]
            if matches:
                candidates[norm].update(matches)
            else:
                missing_after_descriptions += 1

        LOGGER.debug(
            "Ingredient hints after description scan: %d (missing=%d)",
            len(candidates),
            missing_after_descriptions,
        )

        self._apply_category_first_rule(candidates)
        self._apply_description_model(candidates, about_lookup, unique_ingredients)
        return candidates

    def _apply_category_first_rule(self, candidates: dict[str, set[str]]) -> None:
        products = self.products  # type: ignore[attr-defined]
        if products is None:
            raise RuntimeError("Products dataframe must be available before applying rules.")
        concern_primary_categories: dict[str, list[str]] = {}
        for concern in self.concerns:
            primaries = [
                category
                for category, hints in self.category_hints.items()
                if hints and hints[0] == concern
            ]
            if not primaries:
                primaries = [
                    category for category, hints in self.category_hints.items() if concern in hints
                ]
            concern_primary_categories[concern] = primaries

        frames = {
            concern: products[products["category"].isin(categories)].copy()
            for concern, categories in concern_primary_categories.items()
        }
        counts = [len(df) for df in frames.values() if not df.empty]
        if counts:
            per_concern_limit = min(
                min(counts),
                int(self.cfg.params.get("first_ingredient_max_products", 300)),
            )
        else:
            per_concern_limit = 0

        min_hits = int(self.cfg.params.get("first_ingredient_min_hits", 4))
        accumulator: dict[str, Counter] = defaultdict(Counter)
        assignments = 0
        new_ingredients = 0

        for concern, frame in frames.items():
            if frame.empty or per_concern_limit == 0:
                continue
            sampled = frame.sort_values("product_id").head(per_concern_limit).reset_index(drop=True)
            for _, row in sampled.iterrows():
                tokens = row["ingredients_list_normalised"]
                primary = next((token for token in tokens if token not in data_pipeline.WATER_WORDS), None)
                if primary is None:
                    continue
                accumulator[primary][concern] += 1
                assignments += 1

        for ingredient, counter in accumulator.items():
            for concern, hits in counter.items():
                if hits >= min_hits:
                    before = len(candidates.get(ingredient, []))
                    candidates[ingredient].add(concern)
                    after = len(candidates.get(ingredient, []))
                    if before == 0 and after > 0:
                        new_ingredients += 1

        LOGGER.debug(
            "Category-first heuristic processed %d assignments, added %d new ingredients",
            assignments,
            new_ingredients,
        )

    def _apply_description_model(
        self,
        candidates: dict[str, set[str]],
        about_lookup: dict[str, str],
        unique_ingredients: pd.DataFrame,
    ) -> None:
        training_rows = []
        for ingredient, concerns in candidates.items():
            description = about_lookup.get(ingredient)
            if not concerns or not isinstance(description, str) or not description.strip():
                continue
            training_rows.append(
                {
                    "ingredient": ingredient,
                    "description": description,
                    "concerns": sorted(concerns),
                }
            )

        min_samples = int(self.cfg.params.get("description_model_min_samples", 120))
        if len(training_rows) < min_samples:
            LOGGER.info(
                "Skipping TF-IDF description model (insufficient samples: %d < %d)",
                len(training_rows),
                min_samples,
            )
            return

        training_df = pd.DataFrame(training_rows)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85)
        X_train = vectorizer.fit_transform(training_df["description"])
        mlb = MultiLabelBinarizer(classes=self.concerns)
        y_train = mlb.fit_transform(training_df["concerns"])

        classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=400, class_weight="balanced")
        )
        classifier.fit(X_train, y_train)

        max_labels = int(self.cfg.params.get("description_model_max_labels", 2))
        min_prob = float(self.cfg.params.get("description_model_min_probability", 0.42))

        added = 0
        added_labels = 0
        for ingredient in unique_ingredients["ingredient"]:
            norm = normalise_phrase(ingredient)
            if candidates.get(norm):
                continue
            description = about_lookup.get(norm, about_lookup.get(ingredient, ""))
            if not isinstance(description, str) or not description.strip():
                continue
            probs = classifier.predict_proba(vectorizer.transform([description]))[0]
            top_indices = np.argsort(probs)[::-1][:max_labels]
            accepted = [
                mlb.classes_[idx]
                for idx in top_indices
                if probs[idx] >= min_prob
            ]
            if accepted:
                candidates[norm].update(accepted)
                added += 1
                added_labels += len(accepted)

        LOGGER.info(
            "TF-IDF description model enriched %d ingredients with %d labels",
            added,
            added_labels,
        )

    # ------------------------------------------------------ ingredient weights

    def _derive_ingredient_weights(
        self,
        products: pd.DataFrame,
        candidates: dict[str, set[str]],
    ) -> dict[str, dict[str, float]]:
        positional_fns = {
            "harmonic": lambda n: np.array([1.0 / (idx + 1) for idx in range(n)], dtype=float),
            "log": lambda n: np.array([1.0 / math.log(idx + 2) for idx in range(n)], dtype=float),
        }

        concern_primary_categories = {}
        for concern in self.concerns:
            primaries = [
                category
                for category, hints in self.category_hints.items()
                if hints and hints[0] == concern
            ]
            if not primaries:
                primaries = [
                    category for category, hints in self.category_hints.items() if concern in hints
                ]
            concern_primary_categories[concern] = primaries

        concern_frames = {
            concern: products[products["category"].isin(categories)].copy()
            for concern, categories in concern_primary_categories.items()
        }

        coverage_scores: dict[str, float] = {}
        coverage_counts: dict[str, int] = {}
        weight_maps: dict[str, dict[str, dict[str, float]]] = {}

        for strategy in self.positional_strategies:
            if strategy not in positional_fns:
                LOGGER.warning("Unknown positional strategy '%s' skipped", strategy)
                continue
            positional_fn = positional_fns[strategy]
            accumulator: dict[str, Counter] = defaultdict(Counter)

            for _, row in products.iterrows():
                ingredients = row["ingredients_list_normalised"]
                if not ingredients:
                    continue
                weights = _normalise_weights(positional_fn(len(ingredients)))
                for idx, ingredient in enumerate(ingredients):
                    concerns = candidates.get(ingredient)
                    if not concerns:
                        continue
                    share = weights[idx] if idx < len(weights) else 0.0
                    portion = share / max(len(concerns), 1)
                    for concern in concerns:
                        accumulator[ingredient][concern] += portion

            weight_map: dict[str, dict[str, float]] = {}
            for ingredient, counter in accumulator.items():
                total = sum(counter.values())
                if total <= 0.0:
                    continue
                weight_map[ingredient] = {
                    concern: value / total for concern, value in counter.items()
                }

            score = self._evaluate_strategy(weight_map, concern_frames)
            coverage_scores[strategy] = score
            coverage_counts[strategy] = len(weight_map)
            weight_maps[strategy] = weight_map
            LOGGER.info(
                "Strategy %-8s | coverage=%5d | margin=%.4f",
                strategy,
                coverage_counts[strategy],
                coverage_scores[strategy],
            )

        if not weight_maps:
            raise RuntimeError("No ingredient concern weight map could be derived.")

        best_strategy = max(coverage_scores, key=lambda key: coverage_scores[key])
        LOGGER.info("Selected positional strategy: %s", best_strategy)
        return weight_maps[best_strategy]

    def _evaluate_strategy(
        self,
        weights_map: dict[str, dict[str, float]],
        concern_frames: dict[str, pd.DataFrame],
    ) -> float:
        margins: list[float] = []
        limit_per_concern = int(self.cfg.params.get("weight_eval_samples_per_concern", 150))
        for concern, frame in concern_frames.items():
            if frame.empty:
                continue
            limit = min(len(frame), limit_per_concern)
            sample = frame.head(limit)
            for ingredients in sample["ingredients_list_normalised"]:
                if not ingredients:
                    continue
                target_score = sum(weights_map.get(token, {}).get(concern, 0.0) for token in ingredients)
                other_scores = []
                for other_concern in self.concerns:
                    if other_concern == concern:
                        continue
                    other_scores.append(
                        sum(weights_map.get(token, {}).get(other_concern, 0.0) for token in ingredients)
                    )
                opponent = float(np.mean(other_scores)) if other_scores else 0.0
                margins.append(target_score - opponent)
        return float(np.mean(margins)) if margins else 0.0

    # -------------------------------------------------------- product profiles

    def _build_product_profiles(
        self,
        products: pd.DataFrame,
        ingredient_concern_weights: dict[str, dict[str, float]],
        ingredient_rarity: dict[str, float],
    ) -> pd.DataFrame:
        def build_signal_vector(seed: Iterable[str] | None) -> dict[str, float]:
            result = {concern: 0.0 for concern in self.concerns}
            if not seed:
                return result
            for concern in seed:
                if concern in result:
                    result[concern] += 1.0
            total = sum(result.values())
            if total > 0:
                result = {concern: value / total for concern, value in result.items()}
            return result

        def build_name_vector(name: str) -> dict[str, float]:
            vector = {concern: 0.0 for concern in self.concerns}
            if not isinstance(name, str) or not name:
                return vector
            normalized = name.lower()
            tokens = re.findall(r"[a-z0-9-]+", normalized)
            for token in tokens:
                mapping = self.name_hint_tokens.get(token)
                if not mapping:
                    continue
                weight = float(mapping.get("weight", 1.0))
                for concern in mapping.get("concerns", []):
                    if concern in vector:
                        vector[concern] += weight
            for phrase, concerns in self.name_hint_phrases.items():
                if phrase in normalized:
                    for concern in concerns:
                        if concern in vector:
                            vector[concern] += self.name_hint_bonus
            total = sum(vector.values())
            if total > 0:
                vector = {concern: value / total for concern, value in vector.items()}
            return vector

        positional_strategy = self.positional_strategies[0] if self.positional_strategies else "harmonic"
        positional_fn_map = {
            "harmonic": lambda n: np.array([1.0 / (idx + 1) for idx in range(n)], dtype=float),
            "log": lambda n: np.array([1.0 / math.log(idx + 2) for idx in range(n)], dtype=float),
        }
        positional_fn = positional_fn_map.get(positional_strategy, positional_fn_map["harmonic"])

        profiles = []
        for row in products.itertuples(index=False):
            ingredients: list[str] = getattr(row, "ingredients_list_normalised", [])
            weights = _normalise_weights(positional_fn(len(ingredients)))

            ingredient_vector = {concern: 0.0 for concern in self.concerns}
            rare_vector = {concern: 0.0 for concern in self.concerns}
            unknown_count = 0

            for idx, ingredient in enumerate(ingredients):
                concern_weights = ingredient_concern_weights.get(ingredient)
                if not concern_weights:
                    unknown_count += 1
                    continue
                share = weights[idx] if idx < len(weights) else 0.0
                rarity_multiplier = ingredient_rarity.get(ingredient, 1.0)
                for concern, value in concern_weights.items():
                    contribution = share * value
                    ingredient_vector[concern] += contribution
                    rare_vector[concern] += contribution * rarity_multiplier

            ingredient_total = sum(ingredient_vector.values())
            if ingredient_total > 0:
                ingredient_vector = {
                    concern: value / ingredient_total for concern, value in ingredient_vector.items()
                }
            rare_total = sum(rare_vector.values())
            if rare_total > 0:
                rare_vector = {
                    concern: value / rare_total for concern, value in rare_vector.items()
                }

            category_vector = build_signal_vector(self.category_hints.get(getattr(row, "category"), []))
            name_vector = build_name_vector(getattr(row, "name"))

            aggregate_vector = {concern: 0.0 for concern in self.concerns}
            for concern in self.concerns:
                aggregate_vector[concern] = (
                    ingredient_vector.get(concern, 0.0)
                    + category_vector.get(concern, 0.0)
                    + name_vector.get(concern, 0.0)
                )
            aggregate_total = sum(aggregate_vector.values())
            if aggregate_total > 0:
                aggregate_vector = {
                    concern: value / aggregate_total for concern, value in aggregate_vector.items()
                }

            dominant = max(aggregate_vector, key=aggregate_vector.get)

            profile = {
                "product_id": getattr(row, "product_id"),
                "category": getattr(row, "category"),
                "name": getattr(row, "name"),
                **{f"ing_{c}": ingredient_vector.get(c, 0.0) for c in self.concerns},
                **{f"rare_{c}": rare_vector.get(c, 0.0) for c in self.concerns},
                **{f"cat_{c}": category_vector.get(c, 0.0) for c in self.concerns},
                **{f"name_{c}": name_vector.get(c, 0.0) for c in self.concerns},
                **{f"total_{c}": aggregate_vector.get(c, 0.0) for c in self.concerns},
                "unknown_ratio": unknown_count / max(len(ingredients), 1),
                "dominant_concern": dominant,
            }
            profiles.append(profile)

        profile_df = pd.DataFrame(profiles)
        return profile_df

    # --------------------------------------------------------------- classifier

    def _attach_model_signals(self, profiles: pd.DataFrame) -> None:
        valid_profiles = profiles[profiles["dominant_concern"].isin(self.concerns)].copy()
        if valid_profiles.empty:
            LOGGER.warning("No valid profiles with dominant concern; skipping classifier.")
            for concern in self.concerns:
                profiles[f"model_{concern}"] = 0.0
            profiles["model_predicted_concern"] = ""
            profiles["model_confidence"] = 0.0
            return

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_seed)
        train_indices, test_indices = next(
            splitter.split(valid_profiles, valid_profiles["dominant_concern"])
        )
        train_df = valid_profiles.iloc[train_indices].reset_index(drop=True)
        test_df = valid_profiles.iloc[test_indices].reset_index(drop=True)

        missing_in_test = set(self.concerns) - set(test_df["dominant_concern"].unique())
        for concern in missing_in_test:
            candidates = train_df[train_df["dominant_concern"] == concern]
            if candidates.empty:
                continue
            take_row = candidates.iloc[[0]]
            train_df = train_df.drop(take_row.index).reset_index(drop=True)
            test_df = pd.concat([test_df, take_row], ignore_index=True)

        feature_cols = [
            *[f"ing_{c}" for c in self.concerns],
            *[f"rare_{c}" for c in self.concerns],
            *[f"cat_{c}" for c in self.concerns],
            *[f"name_{c}" for c in self.concerns],
            *[f"total_{c}" for c in self.concerns],
            "unknown_ratio",
        ]

        logistic_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=600, class_weight="balanced")),
            ]
        )
        param_grid = {
            "clf__C": [0.5, 1.0, 1.5],
            "clf__solver": ["lbfgs"],
        }

        grid = GridSearchCV(
            logistic_pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=3,
            n_jobs=1,
            refit=True,
        )
        grid.fit(train_df[feature_cols], train_df["dominant_concern"])
        LOGGER.info("Best logistic params: %s", grid.best_params_)

        calibrated = CalibratedClassifierCV(grid.best_estimator_, method="isotonic", cv=3)
        calibrated.fit(train_df[feature_cols], train_df["dominant_concern"])

        y_pred = calibrated.predict(test_df[feature_cols])
        report = classification_report(
            test_df["dominant_concern"], y_pred, digits=3, zero_division=0
        )
        LOGGER.info("Validation report:\n%s", report)

        full_calibrated = CalibratedClassifierCV(grid.best_estimator_, method="isotonic", cv=3)
        full_calibrated.fit(valid_profiles[feature_cols], valid_profiles["dominant_concern"])

        probabilities = full_calibrated.predict_proba(profiles[feature_cols])
        classes = list(full_calibrated.classes_)
        prob_df = pd.DataFrame(probabilities, columns=classes, index=profiles.index)

        for concern in self.concerns:
            profiles[f"model_{concern}"] = prob_df.get(concern, pd.Series(0.0, index=profiles.index))
        profiles["model_predicted_concern"] = full_calibrated.predict(profiles[feature_cols])
        profiles["model_confidence"] = profiles[[f"model_{c}" for c in self.concerns]].max(axis=1)

        baseline_models = {
            "random_forest": LogisticRegression(max_iter=400),
        }
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier

            baseline_models = {
                "random_forest": RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    random_state=self.random_seed,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
                "gradient_boosting": GradientBoostingClassifier(random_state=self.random_seed),
                "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
            }
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Could not import optional baseline models: %s", exc)

        baseline_rows = []
        for name, model in baseline_models.items():
            try:
                scores = cross_val_score(
                    model,
                    train_df[feature_cols],
                    train_df["dominant_concern"],
                    cv=3,
                    scoring="f1_macro",
                    n_jobs=1,
                )
                baseline_rows.append(
                    {"model": name, "mean_f1_macro": float(scores.mean()), "std_f1_macro": float(scores.std())}
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Baseline model %s failed: %s", name, exc)

        if baseline_rows:
            baseline_df = pd.DataFrame(baseline_rows).sort_values("mean_f1_macro", ascending=False)
            LOGGER.info("Baseline models (3-fold F1_macro):\n%s", baseline_df)

    # --------------------------------------------------------------- matrices

    def _build_matrices(self, profiles: pd.DataFrame) -> dict[str, np.ndarray]:
        product_ids = profiles["product_id"].to_numpy(dtype=int)
        product_categories = profiles["category"].to_numpy()
        product_names = profiles["name"].to_numpy()
        dominant_concerns = profiles["dominant_concern"].to_numpy()

        ingredient_matrix = profiles[[f"ing_{c}" for c in self.concerns]].to_numpy(dtype=float)
        rare_matrix = profiles[[f"rare_{c}" for c in self.concerns]].to_numpy(dtype=float)
        name_matrix = profiles[[f"name_{c}" for c in self.concerns]].to_numpy(dtype=float)
        total_matrix = profiles[[f"total_{c}" for c in self.concerns]].to_numpy(dtype=float)
        model_columns = [f"model_{c}" for c in self.concerns]
        for column in model_columns:
            if column not in profiles.columns:
                profiles[column] = 0.0
        model_matrix = profiles[model_columns].to_numpy(dtype=float)
        unknown_array = profiles["unknown_ratio"].to_numpy(dtype=float)

        category_focus_matrix = np.array(
            [
                [
                    1.0 if concern in self.category_hints.get(category, []) else 0.0
                    for concern in self.concerns
                ]
                for category in product_categories
            ],
            dtype=float,
        )

        combined_ingredient_matrix = ingredient_matrix + self.rare_signal_coef * rare_matrix

        return {
            "product_ids": product_ids,
            "product_categories": product_categories,
            "product_names": product_names,
            "dominant_concerns": dominant_concerns,
            "ingredient_matrix": ingredient_matrix,
            "rare_matrix": rare_matrix,
            "combined_ingredient_matrix": combined_ingredient_matrix,
            "name_matrix": name_matrix,
            "total_matrix": total_matrix,
            "model_matrix": model_matrix,
            "category_focus_matrix": category_focus_matrix,
            "unknown_ratio": unknown_array,
        }

    # ---------------------------------------------------------------- outputs

    def _write_outputs(
        self,
        profiles: pd.DataFrame,
        ingredient_concern_weights: dict[str, dict[str, float]],
        matrices: dict[str, np.ndarray],
        candidates: dict[str, set[str]],
    ) -> None:
        profiles_target = self.output_dir / "product_profiles.parquet"
        matrices_output = self.output_dir / "feature_matrices.npz"
        weights_target = self.output_dir / "ingredient_concern_weights.parquet"
        candidate_output = self.output_dir / "ingredient_concern_candidates.json"
        summary_output = self.output_dir / "summary.json"

        def _write_dataframe(df: pd.DataFrame, target: Path) -> Path:
            try:
                df.to_parquet(target, index=False)
                return target
            except (ImportError, ValueError):  # pragma: no cover - depends on optional deps
                fallback = target.with_suffix(".csv")
                df.to_csv(fallback, index=False)
                LOGGER.warning("Parquet support unavailable; wrote CSV to %s", fallback)
                return fallback

        profiles_path = _write_dataframe(profiles, profiles_target)

        np.savez_compressed(matrices_output, **matrices)

        weights_df = (
            pd.DataFrame(
                (
                    {"ingredient": ingredient, "concern": concern, "weight": weight}
                    for ingredient, mapping in ingredient_concern_weights.items()
                    for concern, weight in mapping.items()
                )
            )
            .sort_values(["ingredient", "concern"])
            .reset_index(drop=True)
        )
        weights_path = _write_dataframe(weights_df, weights_target)

        candidate_payload = {
            ingredient: sorted(concern_set) for ingredient, concern_set in candidates.items()
        }
        candidate_output.write_text(
            json.dumps(candidate_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = {
            "product_count": int(len(profiles)),
            "matrix_shape": list(matrices["ingredient_matrix"].shape),
            "concern_labels": self.concerns,
            "output_files": {
                "product_profiles": str(profiles_path),
                "feature_matrices": str(matrices_output),
                "ingredient_concern_weights": str(weights_path),
                "ingredient_concern_candidates": str(candidate_output),
            },
        }
        summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        LOGGER.info("Artifacts written: %s", json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cfg = load_config(args.config)
    builder = FeatureBuilder(cfg, args)
    builder.run()


if __name__ == "__main__":
    main()
