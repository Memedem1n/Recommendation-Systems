from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    # Prefer Pydantic v1 compatibility imports when running on Pydantic v2+
    from pydantic.v1 import BaseModel, BaseSettings, Field, validator
except ImportError:  # pragma: no cover - fallback for genuine v1 installations
    from pydantic import BaseModel, BaseSettings, Field, validator


class ServicePaths(BaseModel):
    """Resolved filesystem paths used by the service."""

    config_path: Path
    checkpoint_path: Path
    products_csv: Path
    dataset_dir: Path
    pipeline_output_dir: Path
    recommendation_log_path: Path


class ServiceSettings(BaseSettings):
    """Configuration values for the recommendation service."""

    config_path: Path = Field(
        default=Path("bt_bert_model/config.yaml"),
        description="YAML configuration file that defines data/model settings.",
    )
    checkpoint_path: Path = Field(
        default=Path("bt_bert_model/outputs/checkpoints/model_state.pt"),
        description="Serialized PyTorch checkpoint for the fine-tuned BT-BERT model.",
    )
    products_csv: Optional[Path] = Field(
        default=None,
        description="Optional override path for the harmonised products CSV.",
    )
    dataset_dir: Path = Field(
        default=Path("Dataset"),
        description="Source directory containing raw scraped CSV files.",
    )
    pipeline_output_dir: Path = Field(
        default=Path("bt_bert_model/data/raw"),
        description="Directory where the dataset pipeline writes normalised outputs.",
    )
    device: str = Field(
        default="auto",
        description="Torch device specifier: 'auto', 'cpu', 'cuda', or explicit device id.",
    )
    inference_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size used when generating concern-level scores.",
    )
    preload_scores: bool = Field(
        default=False,
        description="Whether to compute scores for all concerns at startup.",
    )
    recommendation_log_path: Path = Field(
        default=Path("bt_bert_model/outputs/recommendation_logs.jsonl"),
        description="File path where served recommendations are appended as JSONL records.",
    )

    class Config:
        env_prefix = "BLUESENSE_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator(
        "config_path",
        "checkpoint_path",
        "dataset_dir",
        "pipeline_output_dir",
        "recommendation_log_path",
        pre=True,
    )
    def _expand_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value)).expanduser().resolve()

    @validator("products_csv", pre=True)
    def _expand_optional_path(cls, value: object) -> Optional[Path]:
        if value in (None, "", "null", "None"):
            return None
        return Path(str(value)).expanduser().resolve()

    @validator("device")
    def _validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value in {"auto", "cpu", "cuda"} or value.startswith("cuda:") or value.startswith("mps"):
            return value
        raise ValueError("device must be 'auto', 'cpu', 'cuda', 'cuda:<index>', or 'mps'")

    def resolve_paths(self) -> ServicePaths:
        """Return a helper object with fully resolved paths."""
        products_csv = self.products_csv
        if products_csv is None:
            products_csv = self._derived_products_path()
        return ServicePaths(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            products_csv=products_csv,
            dataset_dir=self.dataset_dir,
            pipeline_output_dir=self.pipeline_output_dir,
            recommendation_log_path=self.recommendation_log_path,
        )

    def _derived_products_path(self) -> Path:
        from yaml import safe_load

        config_path = self.config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            cfg = safe_load(stream)
        project_cfg = cfg.get("project", {}) or {}
        base_dir = Path(project_cfg.get("base_dir", "."))
        if not base_dir.is_absolute():
            base_dir = (config_path.parent / base_dir).resolve()
        data_cfg = cfg.get("data", {}) or {}
        products_rel = data_cfg.get("unified_products_path")
        if not products_rel:
            raise ValueError(
                "unified_products_path missing under data section in config file"
            )
        products_path = Path(products_rel)
        if not products_path.is_absolute():
            products_path = (base_dir / products_path).resolve()
        return products_path
