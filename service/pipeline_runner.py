from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ServiceSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "bt_bert_model" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Dataset_Pipeline import data_pipeline  # type: ignore[import]  # noqa: E402


@dataclass
class PipelineResult:
    products_csv: Path
    ingredient_map_csv: Path
    unique_ingredients_csv: Path
    processed_count: int

    def to_response(self) -> dict:
        return {
            "products_csv": str(self.products_csv),
            "ingredient_map_csv": str(self.ingredient_map_csv),
            "unique_ingredients_csv": str(self.unique_ingredients_csv),
            "records_processed": self.processed_count,
        }


class PipelineRunner:
    """Thin wrapper around Dataset_Pipeline.run_pipeline for service usage."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings

    def run(
        self,
        dataset_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> PipelineResult:
        dataset_path = Path(dataset_dir) if dataset_dir else self.settings.dataset_dir
        output_path = Path(output_dir) if output_dir else self.settings.pipeline_output_dir
        dataset_path = dataset_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        outputs = data_pipeline.run_pipeline(dataset_path, output_path)
        products_path = output_path / "unified_products.csv"
        ingredient_map_path = output_path / "ingredient_normalisation_map.csv"
        unique_ingredients_path = output_path / "unique_ingredients.csv"
        if not products_path.exists():
            raise FileNotFoundError(f"Expected products CSV not found at {products_path}")
        records_processed = len(outputs.products)
        return PipelineResult(
            products_csv=products_path,
            ingredient_map_csv=ingredient_map_path,
            unique_ingredients_csv=unique_ingredients_path,
            processed_count=records_processed,
        )

