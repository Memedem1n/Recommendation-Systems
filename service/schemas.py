from __future__ import annotations

from typing import List, Optional, Sequence

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    product_id: int = Field(..., description="Unique product identifier.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted relevance score.")
    name: str = Field("", description="Product display name.")
    category: str = Field("", description="Original product category.")
    category_norm: str = Field("", description="Normalised category label.")
    product_url: Optional[str] = Field(None, description="Source URL for the product.")
    ingredients: List[str] = Field(default_factory=list, description="Normalised ingredient tokens.")
    ingredient_count: int = Field(0, ge=0, description="Number of ingredients after normalisation.")


class RecommendationResponse(BaseModel):
    concern: str
    top_k: int
    user_id: Optional[str] = Field(default=None, description="Caller identifier if provided.")
    results: List[RecommendationItem]


class ScoreRequest(BaseModel):
    concern: str = Field(..., description="Concern label defined in the configuration.")
    product_ids: Sequence[int] = Field(..., description="Product identifiers to score.")


class ScoreResponse(BaseModel):
    concern: str
    results: List[RecommendationItem]


class RecommendationRequest(BaseModel):
    concern: str = Field(..., description="Concern label defined in the configuration.")
    top_k: int = Field(10, ge=1, le=100, description="Maximum number of recommendations to return.")
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter; matched using the normalisation routine.",
    )
    include_only: Optional[Sequence[int]] = Field(
        default=None,
        description="Optional subset of product IDs to score and rank.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="İsteği yapan kullanıcıyı tanımlamak için opsiyonel alan.",
    )


class PipelineRunRequest(BaseModel):
    dataset_dir: Optional[str] = Field(
        default=None,
        description="Override dataset directory containing scraped CSV files.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Override output directory for harmonised datasets.",
    )


class PipelineRunResponse(BaseModel):
    products_csv: str
    ingredient_map_csv: str
    unique_ingredients_csv: str
    records_processed: int
