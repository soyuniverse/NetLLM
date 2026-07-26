#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(
    os.environ.get("NETLLM_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from netllm_litevlm.evaluation.runtime_benchmark import benchmark_selector
from netllm_litevlm.selectors import IdentitySelector, RecentKSelector


CSV_COLUMNS = [
    "model",
    "checkpoint",
    "selector",
    "selected_tokens",
    "keep_ratio",
    "mae",
    "rmse",
    "mean_angular_error",
    "test_loss",
    "latency_median_ms",
    "latency_p95_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "metric_valid",
]


def _selector_from_config(selector_config: Dict[str, Any]):
    selector_type = selector_config["type"]
    if selector_type == "original":
        return None
    if selector_type == "identity":
        return IdentitySelector()
    if selector_type == "recent_k":
        return RecentKSelector(int(selector_config["keep"]))
    raise ValueError(f"unsupported selector type: {selector_type}")


def prepare_dry_run(config: Dict[str, Any]) -> Dict[str, Any]:
    history_tokens = int(config["dataset"]["history_tokens"])
    runtime = config["runtime"]
    embeddings = torch.zeros(1, history_tokens, 1024, dtype=torch.float32)
    attention_mask = torch.ones(1, history_tokens, dtype=torch.long)

    rows: List[Dict[str, Any]] = []
    configurations: List[Dict[str, Any]] = []
    for selector_config in config["selectors"]:
        keep = int(selector_config["keep"])
        if keep <= 0 or keep > history_tokens:
            raise ValueError(
                f"invalid keep={keep} for history_tokens={history_tokens}"
            )
        selector = _selector_from_config(selector_config)
        selected_indices = list(range(history_tokens - keep, history_tokens))
        selector_latency = None
        if selector is not None:
            output = selector(embeddings, attention_mask)
            if output.selected_indices.tolist() != selected_indices:
                raise RuntimeError(
                    f"selector indices mismatch for {selector_config['name']}"
                )
            selector_latency = benchmark_selector(
                selector,
                embeddings,
                attention_mask,
                repetitions=int(runtime["selector_repetitions"]),
                warmup_repetitions=int(
                    runtime["selector_warmup_repetitions"]
                ),
            ).to_dict()

        configurations.append(
            {
                "name": selector_config["name"],
                "type": selector_config["type"],
                "original_token_count": history_tokens,
                "selected_token_count": keep,
                "keep_ratio": keep / history_tokens,
                "selected_indices": selected_indices,
                "selector_latency": selector_latency,
                "total_inference_latency": None,
                "gpu_peak_allocated_mib": None,
                "gpu_peak_reserved_mib": None,
                "prediction_shape": None,
                "runtime_measured": False,
                "metric_valid": False,
                "metric_invalid_reason": "trained checkpoint unavailable",
            }
        )
        rows.append(
            {
                "model": config["model"],
                "checkpoint": "",
                "selector": selector_config["name"],
                "selected_tokens": keep,
                "keep_ratio": keep / history_tokens,
                "mae": "",
                "rmse": "",
                "mean_angular_error": "",
                "test_loss": "",
                "latency_median_ms": "",
                "latency_p95_ms": "",
                "peak_allocated_mib": "",
                "peak_reserved_mib": "",
                "metric_valid": False,
            }
        )

    return {
        "mode": "dry-run",
        "schema_version": config["schema_version"],
        "model": config["model"],
        "artifact_revision": config["artifact_revision"],
        "checkpoint": config["checkpoint"],
        "metric_valid": False,
        "metric_invalid_reason": "trained checkpoint unavailable",
        "random_head_metrics_reported": False,
        "training_executed": False,
        "llama_executed": False,
        "speculative_decoding_executed": False,
        "csv_columns": CSV_COLUMNS,
        "configurations": configurations,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/vp_benchmark.json",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare contracts only; no model inference or accuracy metrics.",
    )
    args = parser.parse_args()
    if not args.dry_run:
        raise RuntimeError(
            "Only --dry-run is enabled in Phase 3A benchmark preparation"
        )
    if args.output_csv.exists() or args.metadata_json.exists():
        raise RuntimeError("output exists; benchmark preparation will not overwrite it")

    config = json.loads(args.config.read_text())
    result = prepare_dry_run(config)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result.pop("rows"))
    args.metadata_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "success": True,
                "mode": "dry-run",
                "output_csv": str(args.output_csv),
                "metadata_json": str(args.metadata_json),
                "configurations": len(result["configurations"]),
                "metric_valid": result["metric_valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
