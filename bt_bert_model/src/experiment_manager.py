"""Experiment manager for iterative BT-BERT training on the cluster."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from train import load_yaml as load_training_config  # noqa: E402


EXPERIMENT_SEARCH_SPACES: Dict[str, List[Dict[str, Any]]] = {
    "scenario_baseline": [
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 1.0e-5,
            "max": 5.0e-5,
        },
        {
            "name": "attention_scale",
            "path": "model.attention_scale",
            "type": "float",
            "scale": "add",
            "step": 2.0,
            "min": 10.0,
            "max": 30.0,
        },
        {
            "name": "weight_decay",
            "path": "training.weight_decay",
            "type": "float",
            "scale": "add",
            "step": 0.005,
            "min": 0.0,
            "max": 0.04,
        },
    ],
    "scenario_rank_tokens": [
        {
            "name": "ingredient_top_k",
            "path": "text_options.ingredient_top_k",
            "type": "int",
            "scale": "add",
            "step": 10,
            "min": 20,
            "max": 80,
        },
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 1.0e-5,
            "max": 4.0e-5,
        },
        {
            "name": "repeat_factor",
            "path": "text_options.repeat_factor",
            "type": "int",
            "scale": "add",
            "step": 1,
            "min": 1,
            "max": 4,
            "default": 1,
        },
        {
            "name": "attention_scale",
            "path": "model.attention_scale",
            "type": "float",
            "scale": "add",
            "step": 2.0,
            "min": 12.0,
            "max": 28.0,
        },
    ],
    "scenario_repeat_emphasis": [
        {
            "name": "repeat_factor",
            "path": "text_options.repeat_factor",
            "type": "int",
            "scale": "add",
            "step": 1,
            "min": 2,
            "max": 5,
            "default": 3,
        },
        {
            "name": "repeat_top_n",
            "path": "text_options.repeat_top_n",
            "type": "int",
            "scale": "add",
            "step": 2,
            "min": 4,
            "max": 12,
            "default": 6,
        },
        {
            "name": "num_epochs",
            "path": "training.num_epochs",
            "type": "int",
            "scale": "add",
            "step": 1,
            "min": 4,
            "max": 8,
        },
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 1.2e-5,
            "max": 3.5e-5,
        },
    ],
    "scenario_category_prompt": [
        {
            "name": "ingredient_top_k",
            "path": "text_options.ingredient_top_k",
            "type": "int",
            "scale": "add",
            "step": 10,
            "min": 30,
            "max": 80,
        },
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 1.5e-5,
            "max": 4.0e-5,
        },
        {
            "name": "weight_decay",
            "path": "training.weight_decay",
            "type": "float",
            "scale": "add",
            "step": 0.005,
            "min": 0.005,
            "max": 0.03,
        },
        {
            "name": "warmup_steps",
            "path": "training.warmup_steps",
            "type": "int",
            "scale": "add",
            "step": 100,
            "min": 0,
            "max": 800,
        },
    ],
    "scenario_loss_balanced": [
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.15,
            "min": 1.2e-5,
            "max": 3.2e-5,
        },
        {
            "name": "loss_positive_weight",
            "path": "training.loss_positive_weight",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 0.8,
            "max": 6.0,
            "default": "auto",
        },
        {
            "name": "concern_wrinkle_weight",
            "path": "training.concern_loss_weights.wrinkle",
            "type": "float",
            "scale": "add",
            "step": 0.1,
            "min": 0.8,
            "max": 1.8,
            "default": 1.25,
        },
        {
            "name": "concern_moisture_weight",
            "path": "training.concern_loss_weights.moisture",
            "type": "float",
            "scale": "add",
            "step": 0.05,
            "min": 0.6,
            "max": 1.2,
            "default": 0.95,
        },
    ],
    "scenario_stopwords": [
        {
            "name": "ingredient_top_k",
            "path": "text_options.ingredient_top_k",
            "type": "int",
            "scale": "add",
            "step": 5,
            "min": 30,
            "max": 70,
        },
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.15,
            "min": 1.5e-5,
            "max": 3.0e-5,
        },
    ],
    "scenario_keyword_boost": [
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 1.5e-5,
            "max": 3.5e-5,
        },
        {
            "name": "attention_scale",
            "path": "model.attention_scale",
            "type": "float",
            "scale": "add",
            "step": 2.0,
            "min": 12.0,
            "max": 28.0,
        },
    ],
    "scenario_alt_backbone": [
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.15,
            "min": 1.2e-5,
            "max": 3.5e-5,
        },
        {
            "name": "attention_scale",
            "path": "model.attention_scale",
            "type": "float",
            "scale": "add",
            "step": 2.0,
            "min": 12.0,
            "max": 28.0,
        },
        {
            "name": "loss_positive_weight",
            "path": "training.loss_positive_weight",
            "type": "float",
            "scale": "mul",
            "factor": 1.2,
            "min": 0.8,
            "max": 6.0,
            "default": "auto",
        },
        {
            "name": "gradient_accumulation",
            "path": "training.gradient_accumulation",
            "type": "int",
            "scale": "add",
            "step": 1,
            "min": 1,
            "max": 4,
        },
    ],
    "scenario_pseudo_augment": [
        {
            "name": "positive_threshold",
            "path": "augmentation.positive_threshold",
            "type": "float",
            "scale": "add",
            "step": 0.02,
            "min": 0.75,
            "max": 0.95,
        },
        {
            "name": "negative_threshold",
            "path": "augmentation.negative_threshold",
            "type": "float",
            "scale": "add",
            "step": 0.02,
            "min": 0.05,
            "max": 0.25,
        },
        {
            "name": "learning_rate",
            "path": "training.learning_rate",
            "type": "float",
            "scale": "mul",
            "factor": 1.15,
            "min": 1.5e-5,
            "max": 3.5e-5,
        },
    ],
}

STATE_FILENAME = "manager_state.json"
GENERATED_CONFIG_DIR = Path("configs/generated")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "status": "ready",
        "history": [],
        "best_val_f1": None,
        "best_config": None,
        "current_config": None,
        "change_index": 0,
        "direction": 1,
        "pending": None,
        "best_checkpoint": None,
    }


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _split_path(path: str) -> List[str]:
    return path.split(".")


def _get_path(config: Dict[str, Any], path: str) -> Any:
    node: Any = config
    for part in _split_path(path):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = _split_path(path)
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _clamp(value: float, minimum: Optional[float], maximum: Optional[float]) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _apply_change(
    config: Dict[str, Any],
    change: Dict[str, Any],
    direction: int,
) -> Tuple[bool, Any, Any]:
    """Apply change in-place. Returns success flag, previous value, new value."""
    current_value = _get_path(config, change["path"])
    change_type = change["type"]
    prev_value = current_value

    if current_value is None:
        current_value = change.get("default")
        if current_value is None:
            if change_type in {"float", "int"}:
                current_value = 0
            elif change_type == "bool":
                current_value = False

    new_value = current_value

    if change_type == "float":
        if change.get("scale") == "mul":
            factor = float(change.get("factor", 1.1))
            new_value = float(current_value) * (factor if direction >= 0 else 1.0 / factor)
        else:
            step = float(change.get("step", 0.1))
            new_value = float(current_value) + (step * direction)
        new_value = _clamp(
            new_value,
            change.get("min"),
            change.get("max"),
        )
    elif change_type == "int":
        step = int(change.get("step", 1))
        if change.get("scale") == "mul":
            factor = change.get("factor", 1.1)
            new_value = int(round(float(current_value) * (factor if direction >= 0 else 1.0 / factor)))
        else:
            new_value = int(current_value) + (step * direction)
        new_value = int(
            _clamp(
                float(new_value),
                change.get("min"),
                change.get("max"),
            )
        )
    elif change_type == "bool":
        new_value = not bool(current_value)
    else:
        raise ValueError(f"Unsupported change type: {change_type}")

    if new_value == prev_value:
        return False, prev_value, new_value

    if change_type == "float":
        new_value = float(new_value)
    elif change_type == "int":
        new_value = int(new_value)
    elif change_type == "bool":
        new_value = bool(new_value)

    _set_path(config, change["path"], new_value)
    return True, prev_value, new_value


def _generate_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"run_{timestamp}"


def _resolve_logging_paths(config: Dict[str, Any], run_dir: Path) -> None:
    logging_cfg = config.setdefault("logging", {})
    logging_cfg["metrics_path"] = str(run_dir / "metrics.json")
    logging_cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    logging_cfg["attention_report_dir"] = str(run_dir / "attention_reports")


def _prepare_state_for_experiment(
    state: Dict[str, Any],
    experiment_id: str,
    base_config_path: Path,
) -> None:
    state["experiment_id"] = experiment_id
    state["base_config_path"] = str(base_config_path)


def _load_config_for_state(state: Dict[str, Any]) -> Dict[str, Any]:
    base_path = Path(state["base_config_path"])
    cfg = load_training_config(base_path)
    return cfg


def _choose_next_change(
    state: Dict[str, Any],
    experiment_id: str,
) -> Tuple[Optional[Dict[str, Any]], int]:
    search_space = EXPERIMENT_SEARCH_SPACES.get(experiment_id, [])
    if not search_space:
        return None, 0

    idx = state.get("change_index", 0) % len(search_space)
    direction = state.get("direction", 1)
    return search_space[idx], direction


def _cycle_to_next_change(state: Dict[str, Any], num_changes: int) -> None:
    state["change_index"] = (state.get("change_index", 0) + 1) % max(num_changes, 1)
    state["direction"] = 1


def prepare_run(args: argparse.Namespace) -> None:
    experiment_id = args.experiment
    base_config_path = Path(args.base_config).resolve()
    base_dir = (base_config_path.parent / "..").resolve()

    experiment_dir = _ensure_dir(base_dir / "outputs" / "experiments" / experiment_id)
    runs_dir = _ensure_dir(experiment_dir / "runs")
    state_path = experiment_dir / STATE_FILENAME

    state = _load_state(state_path)
    _prepare_state_for_experiment(state, experiment_id, base_config_path)

    if state.get("status") == "awaiting_results":
        raise RuntimeError(
            f"Experiment {experiment_id} is waiting for results from "
            f"run {state.get('pending', {}).get('run_id')}. Complete the run before preparing a new one."
        )

    if state.get("best_config") is None:
        config = load_training_config(base_config_path)
        state["best_config"] = copy.deepcopy(config)
        state["current_config"] = copy.deepcopy(config)
        pending_change: Optional[Dict[str, Any]] = None
    else:
        config = copy.deepcopy(state["best_config"])
        state["current_config"] = copy.deepcopy(config)
        change, direction = _choose_next_change(state, experiment_id)
        pending_change = None
        if change:
            success, prev_value, new_value = _apply_change(config, change, direction)
            attempts = 0
            while not success and attempts < len(EXPERIMENT_SEARCH_SPACES.get(experiment_id, [change])):
                # Unable to apply change (likely due to limits); cycle to next change.
                _cycle_to_next_change(state, len(EXPERIMENT_SEARCH_SPACES.get(experiment_id, [])))
                change, direction = _choose_next_change(state, experiment_id)
                if not change:
                    break
                success, prev_value, new_value = _apply_change(config, change, direction)
                attempts += 1
            if success:
                pending_change = {
                    "name": change["name"],
                    "path": change["path"],
                    "prev": prev_value,
                    "new": new_value,
                    "direction": direction,
                    "type": change["type"],
                }
                state["current_config"] = copy.deepcopy(config)
            else:
                pending_change = None
                state["direction"] = 1
                state["change_index"] = 0

    run_id = _generate_run_id()
    run_dir = _ensure_dir(runs_dir / run_id)
    _resolve_logging_paths(config, run_dir)

    generated_dir = _ensure_dir(GENERATED_CONFIG_DIR / experiment_id)
    generated_config_path = generated_dir / f"{experiment_id}_{run_id}.yaml"
    generated_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    generated_config_path_resolved = generated_config_path.resolve()

    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    state["status"] = "awaiting_results"
    state["pending"] = {
        "run_id": run_id,
        "change": pending_change,
        "config_path": str(generated_config_path_resolved),
        "metrics_path": str(Path(config["logging"]["metrics_path"]).resolve()),
        "checkpoint_dir": str(Path(config["logging"]["checkpoint_dir"]).resolve()),
        "run_dir": str(run_dir.resolve()),
    }
    state.setdefault("history", []).append(
        {
            "run_id": run_id,
            "config_path": str(generated_config_path),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    _save_state(state_path, state)

    data_dir = (base_dir / config["data"]["label_output_dir"]).resolve()

    print(str(generated_config_path_resolved))
    print(str(Path(config["logging"]["metrics_path"]).resolve()))
    print(str(Path(config["logging"]["checkpoint_dir"]).resolve()))
    print(str(data_dir))
    print(str(run_dir.resolve()))
    print(str(state.get("best_checkpoint", "") or ""))


def _read_metrics(metrics_path: Path) -> Tuple[float, float, Dict[str, Any]]:
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not history:
        raise RuntimeError(f"No metrics recorded in {metrics_path}")
    best_entry = max(history, key=lambda item: item.get("f1", float("-inf")))
    last_entry = history[-1]
    return float(best_entry.get("f1", 0.0)), float(last_entry.get("f1", 0.0)), last_entry


def _handle_overfitting(state: Dict[str, Any]) -> bool:
    config = state["current_config"]
    training_cfg = config.setdefault("training", {})
    current_epochs = int(training_cfg.get("num_epochs", 5))
    adjusted = False

    if current_epochs > 3:
        training_cfg["num_epochs"] = current_epochs - 1
        adjusted = True

    patience = training_cfg.get("early_stopping_patience")
    if patience is None or patience > 2:
        training_cfg["early_stopping_patience"] = 2
        adjusted = True

    if adjusted:
        state["current_config"] = copy.deepcopy(config)
        state["best_config"] = copy.deepcopy(config)

    return adjusted


def complete_run(args: argparse.Namespace) -> None:
    experiment_id = args.experiment
    base_config_path = Path(args.base_config).resolve()
    base_dir = (base_config_path.parent / "..").resolve()
    experiment_dir = _ensure_dir(base_dir / "outputs" / "experiments" / experiment_id)
    state_path = experiment_dir / STATE_FILENAME

    state = _load_state(state_path)
    pending = state.get("pending")
    if not pending:
        raise RuntimeError(f"No pending run found for experiment {experiment_id}.")

    run_dir = Path(args.run_dir).resolve()
    metrics_path = Path(pending["metrics_path"]).resolve()
    if args.metrics:
        metrics_path = Path(args.metrics).resolve()

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        checkpoint_dir = Path(pending.get("checkpoint_dir", ""))
        if checkpoint_dir.exists():
            candidates = sorted(checkpoint_dir.glob("*.pt"))
            if candidates:
                checkpoint_path = candidates[-1].resolve()

    best_f1, last_f1, last_entry = _read_metrics(metrics_path)
    prev_best = state.get("best_val_f1")
    improved = prev_best is None or best_f1 > prev_best + 1e-4

    history = state.get("history", [])
    for entry in history:
        if entry.get("run_id") == pending.get("run_id"):
            entry["val_f1"] = best_f1
            entry["last_val_f1"] = last_f1
            entry["improved"] = improved
            break

    state["status"] = "ready"

    pending_change = pending.get("change")
    search_space = EXPERIMENT_SEARCH_SPACES.get(experiment_id, [])

    if improved:
        state["best_val_f1"] = best_f1
        state["best_config"] = copy.deepcopy(state["current_config"])
        state["current_config"] = copy.deepcopy(state["best_config"])
        if checkpoint_path is not None:
            state["best_checkpoint"] = str(checkpoint_path)
        if pending_change and pending_change.get("type") == "bool":
            _cycle_to_next_change(state, len(search_space))
        elif pending_change and pending_change.get("type") != "bool":
            # keep same change to see if more improvement is possible
            pass
        else:
            _cycle_to_next_change(state, len(search_space))
    else:
        if pending_change:
            # revert to previous configuration
            _set_path(state["current_config"], pending_change["path"], pending_change["prev"])
            if state.get("best_config") is None:
                state["best_config"] = copy.deepcopy(state["current_config"])
            state["current_config"] = copy.deepcopy(state["best_config"])
            if pending_change.get("type") == "bool":
                _cycle_to_next_change(state, len(search_space))
            else:
                if pending_change["direction"] == 1:
                    state["direction"] = -1
                else:
                    _cycle_to_next_change(state, len(search_space))
        else:
            _cycle_to_next_change(state, len(search_space))

    overfit = (state.get("best_val_f1") is not None) and ((state["best_val_f1"] - last_f1) > 0.02)
    if overfit:
        adjusted = _handle_overfitting(state)
        if adjusted:
            print(
                "Overfitting detected (val F1 dropped by {:.3f}); applying early-stopping adjustments.".format(
                    state["best_val_f1"] - last_f1
                )
            )

    state["pending"] = None
    _save_state(state_path, state)

    summary = {
        "experiment": experiment_id,
        "run_id": pending.get("run_id"),
        "metrics_path": str(metrics_path),
        "best_val_f1": state.get("best_val_f1"),
        "improved": improved,
        "last_epoch_f1": last_f1,
        "train_loss_last": last_entry.get("train_loss"),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
    }
    print(json.dumps(summary, indent=2))


def download_cache(args: argparse.Namespace) -> None:
    """Helper to download models on the cluster if cache is absent."""
    from transformers import AutoModel, AutoTokenizer

    model_name = args.model or "bert-base-uncased"
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    print(f"Downloading {model_name} to {cache_dir} ...")
    AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    print("Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage BT-BERT experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-run", help="Generate a config for the next run.")
    prepare.add_argument("--experiment", required=True, help="Experiment identifier (e.g., scenario_rank_tokens).")
    prepare.add_argument("--base-config", required=True, help="Path to the base configuration YAML.")

    complete = subparsers.add_parser("complete-run", help="Record metrics after a run finishes.")
    complete.add_argument("--experiment", required=True, help="Experiment identifier.")
    complete.add_argument("--base-config", required=True, help="Path to the base configuration YAML.")
    complete.add_argument("--run-dir", required=True, help="Run directory produced by prepare-run.")
    complete.add_argument("--metrics", help="Optional explicit metrics path.")
    complete.add_argument("--checkpoint", help="Path to the checkpoint saved for this run.")

    download = subparsers.add_parser("download-cache", help="Download Hugging Face models to the cache.")
    download.add_argument("--model", default="bert-base-uncased", help="Model name or path.")
    download.add_argument("--cache-dir", default="~/hf_cache", help="Target cache directory.")

    args = parser.parse_args()

    if args.command == "prepare-run":
        prepare_run(args)
    elif args.command == "complete-run":
        complete_run(args)
    elif args.command == "download-cache":
        download_cache(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
