from __future__ import annotations

import argparse
from textwrap import indent

from .hybrid_recommender import HybridRecommender, load_demo_profiles


def format_recommendation(rec, rank):
    lines = [
        f"{rank}. {rec.name} (#{rec.product_id}, {rec.category})",
        f"   Hybrid score: {rec.score:.4f}",
    ]
    if rec.model_probabilities:
        probs = ", ".join(
            f"{conc}:{prob:.2f}"
            for conc, prob in rec.model_probabilities.items()
        )
        lines.append(f"   Model probs: {probs}")
    return "\n".join(lines)


def run_demo(user_ids: list[str], top_k: int, min_score: float) -> None:
    recommender = HybridRecommender()
    profiles = load_demo_profiles()

    if not user_ids:
        user_ids = sorted(profiles.keys())

    for user_id in user_ids:
        profile = profiles.get(user_id)
        if not profile:
            print(f"[WARN] Profile '{user_id}' not found, skipping.")
            continue
        print(f"\n=== Recommendations for {user_id} ===")
        recommendations = recommender.recommend_for_profile(
            profile,
            top_k=top_k,
            min_score=min_score,
        )
        if not recommendations:
            print("No recommendations above the requested score threshold.")
            continue

        for rank, rec in enumerate(recommendations, start=1):
            print(indent(format_recommendation(rec, rank), "  "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print demo recommendations using hybrid concern weights."
    )
    parser.add_argument(
        "--users",
        nargs="*",
        default=None,
        help="Subset of demo user IDs to evaluate (default: all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of items to show per user profile.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum hybrid score threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(
        user_ids=args.users or [],
        top_k=args.top_k,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()

