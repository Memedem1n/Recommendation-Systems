from __future__ import annotations

from functools import lru_cache

from .config import ServiceSettings
from .pipeline_runner import PipelineRunner
from .recommender import BTBertRecommender


@lru_cache(maxsize=1)
def get_settings() -> ServiceSettings:
    return ServiceSettings()


@lru_cache(maxsize=1)
def get_recommender() -> BTBertRecommender:
    settings = get_settings()
    return BTBertRecommender(settings)


@lru_cache(maxsize=1)
def get_pipeline_runner() -> PipelineRunner:
    settings = get_settings()
    return PipelineRunner(settings)

