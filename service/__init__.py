"""Service package for BlueSense recommendation microservice."""

from __future__ import annotations

try:  # Optional re-export; skip if optional deps are missing.
    from .config import ServiceSettings  # noqa: F401
    from .recommender import BTBertRecommender  # noqa: F401
except Exception:  # pragma: no cover - best effort for lightweight scripts.
    ServiceSettings = None  # type: ignore
    BTBertRecommender = None  # type: ignore
