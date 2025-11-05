from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

from .recommender import RecommendationResult

_LOG_LOCK = Lock()


def append_recommendation_log(
    path: Path,
    *,
    user_id: Optional[str],
    concern: str,
    top_k: int,
    category: Optional[str],
    include_only: Optional[Iterable[int]],
    results: Iterable[RecommendationResult],
) -> None:
    """Append a JSONL record describing served recommendations."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "concern": concern,
        "top_k": top_k,
        "category_filter": category,
        "include_only": list(include_only) if include_only else None,
        "results": [
            {
                "product_id": result.product_id,
                "probability": result.probability,
                "ingredient_count": result.ingredient_count,
            }
            for result in results
        ],
    }

    payload = json.dumps(record, ensure_ascii=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")

