"""Hybrid concern-based recommender built on precomputed product weights.

This module provides a lightweight bridge between the hybrid experimentation
outputs and the downstream service layer.  It loads the consolidated
`product_concern_weights.csv` file produced by the Hybrid Concern notebook and
supports profile-driven recommendation ranking without requiring a live
BT-BERT checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_WEIGHTS_PATH = (
    Path("Experiments")
    / "Hybrid_Concern_Test"
    / "product_concern_weights.csv"
)


@dataclass
class HybridRecommendation:
    """Container for a recommendation row returned by the hybrid engine."""

    product_id: int
    name: str
    category: str
    score: float
    concern_scores: Dict[str, float]
    model_probabilities: Dict[str, float]
    ingredient_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "score": self.score,
            "concern_scores": self.concern_scores,
            "model_probabilities": self.model_probabilities,
            "ingredient_count": self.ingredient_count,
        }


class HybridRecommender:
    """Simple weighted recommender based on hybrid concern scores."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Hybrid weights file not found: {weights_path}. "
                "Run the Hybrid Concern notebook to regenerate it."
            )
        self.weights_path = weights_path
        self.data = pd.read_csv(weights_path)
        self.concerns = self._detect_concerns()

    def _detect_concerns(self) -> List[str]:
        prefix = "model_prob_"
        return sorted(
            col[len(prefix) :]
            for col in self.data.columns
            if col.startswith(prefix)
        )

    def recommend_for_profile(
        self,
        profile: Dict[str, float],
        *,
        top_k: int = 5,
        concern_filter: Optional[Iterable[str]] = None,
        min_score: float = 0.0,
    ) -> List[HybridRecommendation]:
        """Return top products for a concern preference profile."""

        if not profile:
            raise ValueError("Profile must provide at least one concern weight.")

        active_concerns = (
            {concern for concern in concern_filter or self.concerns}
            & set(self.concerns)
        )
        if not active_concerns:
            raise ValueError(
                "Concern filter eliminates all available concerns. "
                f"Available concerns: {', '.join(self.concerns)}"
            )

        # Normalise profile weights over the active concerns.
        weights = {
            concern: float(profile.get(concern, 0.0))
            for concern in active_concerns
        }
        total = sum(weights.values())
        if total == 0:
            raise ValueError(
                "Profile weights sum to zero for the selected concerns."
            )
        weights = {concern: value / total for concern, value in weights.items()}

        score_columns = [f"model_prob_{concern}" for concern in active_concerns]
        missing = [col for col in score_columns if col not in self.data.columns]
        if missing:
            raise KeyError(
                f"Missing model probability columns in weights CSV: {missing}"
            )

        scores = self.data[score_columns].fillna(0.0).copy()
        for concern in active_concerns:
            scores[f"weighted_{concern}"] = (
                scores[f"model_prob_{concern}"] * weights[concern]
            )

        aggregated = scores[[f"weighted_{c}" for c in active_concerns]].sum(axis=1)
        ranked = self.data.assign(
            hybrid_score=aggregated,
        ).sort_values("hybrid_score", ascending=False)

        ranked = ranked[ranked["hybrid_score"] >= min_score]
        top = ranked.head(top_k)

        recommendations: List[HybridRecommendation] = []
        for row in top.itertuples():
            concern_scores = {}
            model_probs = {}
            ingredient_count = getattr(row, "ingredient_count", 0)
            for concern in self.concerns:
                prob_col = f"model_prob_{concern}"
                weight_col = f"weight_{concern}"
                if prob_col in self.data.columns:
                    model_probs[concern] = float(getattr(row, prob_col, 0.0))
                if weight_col in self.data.columns:
                    concern_scores[concern] = float(getattr(row, weight_col, 0.0))

            recommendations.append(
                HybridRecommendation(
                    product_id=int(getattr(row, "product_id")),
                    name=str(getattr(row, "name")),
                    category=str(getattr(row, "category")),
                    score=float(getattr(row, "hybrid_score")),
                    concern_scores=concern_scores,
                    model_probabilities=model_probs,
                    ingredient_count=int(ingredient_count) if ingredient_count == ingredient_count else 0,
                )
            )
        return recommendations


def load_demo_profiles() -> Dict[str, Dict[str, float]]:
    """Return the built-in demo user concern vectors."""

    return {
        "general_1": {
            "redness": 0.817,
            "eyebag": 0.875,
            "acne": 0.909,
            "oiliness": 0.289,
            "wrinkle": 0.325,
            "age": 0.193,
            "moisture": 0.802,
        },
        "general_2": {
            "redness": 0.896,
            "eyebag": 0.466,
            "acne": 0.659,
            "oiliness": 0.239,
            "wrinkle": 0.937,
            "age": 0.878,
            "moisture": 0.979,
        },
        "general_3": {
            "redness": 0.83,
            "eyebag": 0.893,
            "acne": 0.122,
            "oiliness": 0.763,
            "wrinkle": 0.399,
            "age": 0.938,
            "moisture": 0.822,
        },
        "general_4": {
            "redness": 0.878,
            "eyebag": 0.83,
            "acne": 0.34,
            "oiliness": 0.809,
            "wrinkle": 0.197,
            "age": 0.885,
            "moisture": 0.873,
        },
        "general_5": {
            "redness": 0.3,
            "eyebag": 0.835,
            "acne": 0.514,
            "oiliness": 0.375,
            "wrinkle": 0.816,
            "age": 0.305,
            "moisture": 0.121,
        },
        "general_6": {
            "redness": 0.274,
            "eyebag": 0.395,
            "acne": 0.878,
            "oiliness": 0.97,
            "wrinkle": 0.351,
            "age": 0.677,
            "moisture": 0.46,
        },
        "general_7": {
            "redness": 0.983,
            "eyebag": 0.583,
            "acne": 0.945,
            "oiliness": 0.204,
            "wrinkle": 0.973,
            "age": 0.261,
            "moisture": 0.966,
        },
        "general_8": {
            "redness": 0.339,
            "eyebag": 0.198,
            "acne": 0.491,
            "oiliness": 0.756,
            "wrinkle": 0.382,
            "age": 0.646,
            "moisture": 0.56,
        },
        "general_9": {
            "redness": 0.447,
            "eyebag": 0.619,
            "acne": 0.329,
            "oiliness": 0.738,
            "wrinkle": 0.102,
            "age": 0.933,
            "moisture": 0.585,
        },
        "general_10": {
            "redness": 0.747,
            "eyebag": 0.768,
            "acne": 0.704,
            "oiliness": 0.428,
            "wrinkle": 0.163,
            "age": 0.698,
            "moisture": 0.397,
        },
        "general_11": {
            "redness": 0.383,
            "eyebag": 0.863,
            "acne": 0.748,
            "oiliness": 0.37,
            "wrinkle": 0.378,
            "age": 0.468,
            "moisture": 0.462,
        },
        "general_12": {
            "redness": 0.366,
            "eyebag": 0.215,
            "acne": 0.478,
            "oiliness": 0.946,
            "wrinkle": 0.71,
            "age": 0.913,
            "moisture": 0.654,
        },
        "general_13": {
            "redness": 0.371,
            "eyebag": 0.593,
            "acne": 0.1,
            "oiliness": 0.358,
            "wrinkle": 0.487,
            "age": 0.622,
            "moisture": 0.689,
        },
        "general_14": {
            "redness": 0.518,
            "eyebag": 0.498,
            "acne": 0.292,
            "oiliness": 0.526,
            "wrinkle": 0.911,
            "age": 0.816,
            "moisture": 0.253,
        },
        "focused_redness": {
            "redness": 0.884,
            "eyebag": 0.127,
            "acne": 0.145,
            "oiliness": 0.1,
            "wrinkle": 0.173,
            "age": 0.163,
            "moisture": 0.151,
        },
        "focused_eyebag": {
            "redness": 0.08,
            "eyebag": 0.944,
            "acne": 0.087,
            "oiliness": 0.121,
            "wrinkle": 0.177,
            "age": 0.061,
            "moisture": 0.112,
        },
        "focused_acne": {
            "redness": 0.079,
            "eyebag": 0.154,
            "acne": 0.966,
            "oiliness": 0.087,
            "wrinkle": 0.148,
            "age": 0.051,
            "moisture": 0.163,
        },
        "focused_oiliness": {
            "redness": 0.066,
            "eyebag": 0.114,
            "acne": 0.076,
            "oiliness": 0.977,
            "wrinkle": 0.128,
            "age": 0.058,
            "moisture": 0.087,
        },
        "focused_wrinkle": {
            "redness": 0.118,
            "eyebag": 0.17,
            "acne": 0.15,
            "oiliness": 0.198,
            "wrinkle": 0.942,
            "age": 0.193,
            "moisture": 0.184,
        },
        "focused_age": {
            "redness": 0.158,
            "eyebag": 0.126,
            "acne": 0.175,
            "oiliness": 0.132,
            "wrinkle": 0.185,
            "age": 0.889,
            "moisture": 0.121,
        },
        "focused_moisture": {
            "redness": 0.087,
            "eyebag": 0.146,
            "acne": 0.165,
            "oiliness": 0.128,
            "wrinkle": 0.144,
            "age": 0.091,
            "moisture": 0.893,
        },
    }

