from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Path
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_pipeline_runner, get_recommender, get_settings
from .logging_utils import append_recommendation_log
from .schemas import (
    PipelineRunRequest,
    PipelineRunResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    ScoreRequest,
    ScoreResponse,
)

app = FastAPI(
    title="BlueSense Recommendation Service",
    description="Serves BT-BERT concern recommendations via REST endpoints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    # Instantiate heavy dependencies on startup to surface issues early.
    await run_in_threadpool(get_recommender)


@app.get("/health", tags=["system"])
async def healthcheck() -> Dict[str, Any]:
    settings = get_settings()
    recommender = get_recommender()
    return {
        "status": "ok",
        "concerns": recommender.available_concerns(),
        "concern_count": len(recommender.available_concerns()),
        "products_count": recommender.products_count(),
        "device": str(recommender.device),
        "checkpoint_path": str(settings.resolve_paths().checkpoint_path),
        "products_csv": str(settings.resolve_paths().products_csv),
    }


@app.get("/concerns", tags=["metadata"])
async def list_concerns() -> Dict[str, List[str]]:
    recommender = get_recommender()
    return {"concerns": recommender.available_concerns()}


@app.get("/products/{product_id}", tags=["metadata"])
async def get_product(
    product_id: int = Path(..., description="Product identifier."),
) -> Dict[str, Any]:
    recommender = get_recommender()
    try:
        product_row = recommender.products.loc[product_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.") from exc
    return {
        "product_id": int(product_id),
        "name": str(product_row.get("title_text") or product_row.get("name") or ""),
        "category": str(product_row.get("category") or ""),
        "category_norm": str(product_row.get("category_norm") or ""),
        "product_url": str(product_row.get("product_url") or "") or None,
        "ingredients": list(product_row.get("ingredients_list") or []),
        "ingredient_count": int(product_row.get("ingredient_count") or 0),
    }


@app.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
async def create_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    recommender = get_recommender()
    try:
        results = await run_in_threadpool(
            recommender.recommend_for_concern,
            request.concern,
            request.top_k,
            request.category,
            request.include_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    canonical = recommender.canonicalise_concern(request.concern)
    settings = get_settings()
    paths = settings.resolve_paths()
    await run_in_threadpool(
        append_recommendation_log,
        paths.recommendation_log_path,
        user_id=request.user_id,
        concern=canonical,
        top_k=request.top_k,
        category=request.category,
        include_only=request.include_only,
        results=results,
    )

    return RecommendationResponse(
        concern=canonical,
        top_k=request.top_k,
        user_id=request.user_id,
        results=[RecommendationItem(**result.to_dict()) for result in results],
    )


@app.post("/scores", response_model=ScoreResponse, tags=["recommendations"])
async def score_products(request: ScoreRequest) -> ScoreResponse:
    recommender = get_recommender()
    try:
        results = await run_in_threadpool(
            recommender.score_products,
            request.concern,
            list(request.product_ids),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    concern = recommender.canonicalise_concern(request.concern)
    return ScoreResponse(
        concern=concern,
        results=[RecommendationItem(**result.to_dict()) for result in results],
    )


@app.post("/pipeline/run", response_model=PipelineRunResponse, tags=["pipeline"])
async def run_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
    pipeline_runner = get_pipeline_runner()
    recommender = get_recommender()
    dataset_override = Path(request.dataset_dir).expanduser().resolve() if request.dataset_dir else None
    output_override = Path(request.output_dir).expanduser().resolve() if request.output_dir else None
    result = await run_in_threadpool(
        pipeline_runner.run,
        dataset_override,
        output_override,
    )
    await run_in_threadpool(recommender.refresh_products, result.products_csv)
    return PipelineRunResponse(
        products_csv=str(result.products_csv),
        ingredient_map_csv=str(result.ingredient_map_csv),
        unique_ingredients_csv=str(result.unique_ingredients_csv),
        records_processed=result.processed_count,
    )
