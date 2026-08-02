#!/usr/bin/env python3
"""Speculative block-verification benchmark harness.

Compares SpeculativeBlockVerifyPipeline against the unmodified
LlamaOldSelectablePipeline baseline across a (threshold, gamma) grid, on
real Jin2022 test samples with the real fine-tuned VP checkpoint. Reuses
this project's own metric/latency utilities
(netllm_litevlm.evaluation.vp_metrics, .runtime_benchmark) rather than
reimplementing them, so results stay numerically consistent with existing
benchmark tables (e.g. experiments/vp/llama_benchmark).

Requires --checkpoint-path and --dataset-path to exist; exits immediately
with a clear error otherwise (checkpoint/dataset are not currently present
in this instance -- see docs/experiment_phase/speculative/PHASE_A_DESIGN.md).
Pass --dry-run to self-test the harness's own machinery (CLI parsing,
model assembly, measurement loop, CSV/summary writing) with the same
random-head assembly and synthetic inputs used in
docs/experiment_phase/speculative/PHASE_B_7B_SMOKE.md -- it produces
metric_valid=False rows by design and must never be mistaken for a real
result.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(
    os.environ.get("NETLLM_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from netllm_litevlm.evaluation.runtime_benchmark import benchmark_callable
from netllm_litevlm.evaluation.vp_metrics import evaluate_vp_metrics
from netllm_litevlm.speculative import SpeculativeBlockVerifyPipeline
from netllm_litevlm.vp.checkpoint_era_runtime import (
    DEFAULT_BASE_MODEL_PATH,
    UPSTREAM_VP_ROOT,
    load_checkpoint_era_model,
)
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline

FUT_WINDOW = 20
HIS_WINDOW = 10

# forward-count reduction AND latency reduction must both be measured for
# a speedup claim; MAE degrading by more than this fraction relative to
# the baseline means accuracy was not preserved. Centralized here so the
# summary's judgment fields are reproducible from one place.
ACCURACY_DEGRADATION_TOLERANCE = 0.05

CSV_COLUMNS = [
    "config",
    "threshold",
    "gamma",
    "num_samples",
    "mae",
    "corrected_rmse",
    "test_loss",
    "latency_median_ms",
    "target_forward_avg",
    "accepted_per_iteration_avg",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "metric_valid",
]


def _synthetic_sample(seed: int):
    import numpy as np

    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(HIS_WINDOW + FUT_WINDOW).float()
    roll = 30.0 * torch.sin(t / 3.0) + torch.randn(t.shape[0], generator=generator) * 2.0
    pitch = 15.0 * torch.cos(t / 4.0) + torch.randn(t.shape[0], generator=generator) * 1.0
    yaw = 5.0 * t + torch.randn(t.shape[0], generator=generator) * 2.0
    trace = torch.stack([roll, pitch, yaw], dim=-1)
    history = trace[:HIS_WINDOW].numpy().astype(np.float32)
    future = trace[HIS_WINDOW:].numpy().astype(np.float32)
    return history, future, (seed, seed, seed)


def _load_samples(args, dry_run: bool) -> List[Any]:
    if dry_run:
        return [_synthetic_sample(seed) for seed in range(args.num_samples)]

    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from config import cfg
    from dataset.load_dataset import create_dataset

    cfg.dataset["Jin2022"] = str(args.dataset_path)
    test_dataset = create_dataset(
        "Jin2022",
        his_window=HIS_WINDOW,
        fut_window=FUT_WINDOW,
        trim_head=30,
        trim_tail=60,
        frequency=5,
        step=15,
        include=["test"],
    )[0]
    count = min(args.num_samples, len(test_dataset))
    return [test_dataset[i] for i in range(count)]


def _normalize(history_np, future_np, dry_run: bool, device: str, dtype):
    # dry_run's synthetic samples are already generated in real-angle units
    # (see _synthetic_sample) specifically so this same normalization path
    # applies to both -- no separate unnormalized code path to drift from
    # the real one.
    del dry_run
    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from utils.normalize import normalize_data

    history_raw = torch.from_numpy(history_np).unsqueeze(0)
    future_raw = torch.from_numpy(future_np).unsqueeze(0)
    history = normalize_data(history_raw, "Jin2022").to(device, dtype=dtype)
    return history, future_raw


def _run_pipeline_over_samples(
    pipeline, samples, dry_run: bool, device: str, dtype
) -> Dict[str, Any]:
    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from utils.normalize import denormalize_data

    predictions = []
    targets = []
    latencies_ms = []
    forward_counts = []
    accepted_counts = []
    per_sample: List[Dict[str, Any]] = []
    is_cuda = device.startswith("cuda")

    for sample_id, (history_np, future_np, info) in enumerate(samples):
        history, future = _normalize(history_np, future_np, dry_run, device, dtype)

        def _call():
            return pipeline.inference(history, future, info)

        if is_cuda:
            torch.cuda.synchronize()
            started = time.perf_counter()
        with torch.inference_mode():
            prediction, target = _call()
        if is_cuda:
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
        else:
            latency_ms = float("nan")
        latencies_ms.append(latency_ms)

        # pipeline.inference returns the model's raw normalized-space
        # ([-1,1]-ish, Tanh-bounded) prediction; target is already
        # raw-degree (see _normalize). Denormalize before comparing, same
        # as run_llama_selector_benchmark.py's `prediction_norm * scale`.
        prediction_degrees = denormalize_data(prediction.float(), "Jin2022")
        target_float = target.float()
        predictions.append(prediction_degrees.cpu())
        targets.append(target_float.cpu())

        if hasattr(pipeline, "target_forward_count"):
            forward_count = pipeline.target_forward_count
            accepted_this_sample = list(pipeline.accepted_per_iteration)
            forward_counts.append(forward_count)
            accepted_counts.extend(accepted_this_sample)
        else:
            forward_count = pipeline.last_trace["plm_forward_count"]
            accepted_this_sample = None
            forward_counts.append(forward_count)

        sample_metrics = evaluate_vp_metrics(
            prediction_degrees, target_float, checkpoint_available=True
        )
        video, user, timestep = int(info[0]), int(info[1]), int(info[2])
        per_sample.append(
            {
                "sample_id": sample_id,
                "video": video,
                "user": user,
                "timestep": timestep,
                "mae": sample_metrics.mae,
                "corrected_rmse": sample_metrics.corrected_rotation_aware_rmse,
                "mean_angular_error": sample_metrics.mean_angular_error,
                "latency_ms": latency_ms,
                "target_forward_count": forward_count,
                "accepted_sum": (
                    sum(accepted_this_sample) if accepted_this_sample is not None else None
                ),
                "finite": bool(torch.isfinite(prediction_degrees).all().item()),
            }
        )

    return {
        "predictions": torch.cat(predictions, dim=0),
        "targets": torch.cat(targets, dim=0),
        "latencies_ms": latencies_ms,
        "forward_counts": forward_counts,
        "per_sample": per_sample,
        "accepted_counts": accepted_counts,
    }


def _peak_memory(device: str) -> Dict[str, Optional[float]]:
    if not device.startswith("cuda"):
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576.0,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1048576.0,
    }


def _reset_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _build_row(name: str, threshold, gamma, run: Dict[str, Any], metric_available: bool) -> Dict[str, Any]:
    metrics = evaluate_vp_metrics(
        run["predictions"], run["targets"], checkpoint_available=metric_available
    )
    accepted_avg = (
        sum(run["accepted_counts"]) / len(run["accepted_counts"])
        if run["accepted_counts"]
        else None
    )
    return {
        "config": name,
        "threshold": threshold,
        "gamma": gamma,
        "num_samples": run["predictions"].shape[0],
        "mae": metrics.mae,
        "corrected_rmse": metrics.corrected_rotation_aware_rmse,
        "test_loss": metrics.test_loss,
        "latency_median_ms": _median(run["latencies_ms"]),
        "target_forward_avg": sum(run["forward_counts"]) / len(run["forward_counts"]),
        "accepted_per_iteration_avg": accepted_avg,
        **_peak_memory(run.get("device", "cpu")),
        "metric_valid": metrics.metric_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--base-model-path", type=Path, default=None)
    # acceptance_threshold lives in SpeculativeBlockVerifyPipeline's own
    # comparison space: normalized (Tanh-bounded, ~[-1,1] per channel)
    # coordinates, not denormalized degrees -- see
    # docs/experiment_phase/speculative/PHASE_A_DESIGN.md /
    # PHASE_B_REAL_RESULTS.md for the empirical calibration (10 real
    # samples' draft-vs-target normalized L2 disagreement: median 0.174,
    # most mass in 0.01-0.7, with rare fast-yaw outliers reaching 3-9 that
    # should stay rejected). This sweep spans strict to generous within
    # that real range.
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.2,0.35,0.7")
    parser.add_argument("--gammas", type=str, default="2,4,8")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "speculative")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Self-test the harness with a random-head assembly and synthetic "
        "inputs (see docs/experiment_phase/speculative/PHASE_B_7B_SMOKE.md). "
        "Produces metric_valid=False rows; never a real result.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        if args.checkpoint_path is None or not Path(args.checkpoint_path).exists():
            print(
                f"ERROR: --checkpoint-path is required and must exist "
                f"(got {args.checkpoint_path}); the fine-tuned VP checkpoint "
                "is not currently available in this instance -- see "
                "docs/experiment_phase/speculative/PHASE_A_DESIGN.md. "
                "Pass --dry-run to self-test the harness without it.",
                file=sys.stderr,
            )
            return 2
        if args.dataset_path is None or not Path(args.dataset_path).exists():
            print(
                f"ERROR: --dataset-path is required and must exist "
                f"(got {args.dataset_path}); the Jin2022 dataset is not "
                "currently available in this instance. Pass --dry-run to "
                "self-test the harness without it.",
                file=sys.stderr,
            )
            return 2

    thresholds = [float(value) for value in args.thresholds.split(",") if value]
    gammas = [int(value) for value in args.gammas.split(",") if value]
    dtype = torch.float16
    device = args.device

    torch.manual_seed(args.seed)
    model, checkpoint_loaded = load_checkpoint_era_model(
        base_model_path=args.base_model_path or DEFAULT_BASE_MODEL_PATH,
        device=device,
        dtype=dtype,
        rank=32,
        fut_window=FUT_WINDOW,
        checkpoint_path=None if args.dry_run else args.checkpoint_path,
        seed=args.seed,
    )
    if args.dry_run and checkpoint_loaded:
        raise RuntimeError("dry-run must not load a real checkpoint")
    if not args.dry_run and not checkpoint_loaded:
        raise RuntimeError("checkpoint load requested but did not happen")

    samples = _load_samples(args, args.dry_run)
    if not samples:
        print("ERROR: no samples loaded", file=sys.stderr)
        return 2

    metric_available = checkpoint_loaded

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write_per_sample_csv(name: str, per_sample: List[Dict[str, Any]]) -> None:
        path = output_dir / f"per_sample_{name}.csv"
        columns = [
            "sample_id", "video", "user", "timestep", "mae", "corrected_rmse",
            "mean_angular_error", "latency_ms", "target_forward_count",
            "accepted_sum", "finite",
        ]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(per_sample)

    rows: List[Dict[str, Any]] = []

    baseline_pipeline = LlamaOldSelectablePipeline(model)
    _reset_memory(device)
    baseline_run = _run_pipeline_over_samples(baseline_pipeline, samples, args.dry_run, device, dtype)
    baseline_run["device"] = device
    baseline_row = _build_row("baseline", None, None, baseline_run, metric_available)
    rows.append(baseline_row)
    _write_per_sample_csv("baseline", baseline_run["per_sample"])

    configs = []
    for threshold in thresholds:
        for gamma in gammas:
            spec_pipeline = SpeculativeBlockVerifyPipeline(
                model, gamma=gamma, acceptance_threshold=threshold
            )
            _reset_memory(device)
            run = _run_pipeline_over_samples(spec_pipeline, samples, args.dry_run, device, dtype)
            run["device"] = device
            config_name = f"threshold={threshold}_gamma={gamma}"
            row = _build_row(config_name, threshold, gamma, run, metric_available)
            speedup_claim_valid = (
                row["target_forward_avg"] < baseline_row["target_forward_avg"]
                and row["latency_median_ms"] < baseline_row["latency_median_ms"]
            )
            accuracy_preserved = (
                metric_available
                and baseline_row["mae"] is not None
                and row["mae"] is not None
                and row["mae"] <= baseline_row["mae"] * (1.0 + ACCURACY_DEGRADATION_TOLERANCE)
            )
            row["speedup_claim_valid"] = speedup_claim_valid
            row["accuracy_preserved"] = accuracy_preserved
            rows.append(row)
            configs.append(row)
            _write_per_sample_csv(config_name, run["per_sample"])

    baseline_row["speedup_claim_valid"] = False
    baseline_row["accuracy_preserved"] = True

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})

    summary = {
        "dry_run": args.dry_run,
        "checkpoint_loaded": checkpoint_loaded,
        "metric_available": metric_available,
        "num_samples": len(samples),
        "thresholds": thresholds,
        "gammas": gammas,
        "accuracy_degradation_tolerance": ACCURACY_DEGRADATION_TOLERANCE,
        "speedup_claim_criterion": "target_forward_avg < baseline AND latency_median_ms < baseline (both measured)",
        "accuracy_preserved_criterion": f"mae <= baseline_mae * (1 + {ACCURACY_DEGRADATION_TOLERANCE})",
        "baseline": baseline_row,
        "configs": configs,
    }
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    summary_md_lines = [
        "# Speculative Block Verification Benchmark",
        "",
        f"dry_run={args.dry_run}, checkpoint_loaded={checkpoint_loaded}, "
        f"metric_available={metric_available}, num_samples={len(samples)}",
        "",
        "| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | "
        "avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        summary_md_lines.append(
            "| {config} | {threshold} | {gamma} | {mae} | {corrected_rmse} | "
            "{latency_median_ms:.3f} | {target_forward_avg:.2f} | {accepted} | "
            "{speedup} | {accuracy} |".format(
                config=row["config"],
                threshold=row["threshold"],
                gamma=row["gamma"],
                mae=row["mae"],
                corrected_rmse=row["corrected_rmse"],
                latency_median_ms=row["latency_median_ms"],
                target_forward_avg=row["target_forward_avg"],
                accepted=row.get("accepted_per_iteration_avg"),
                speedup=row.get("speedup_claim_valid"),
                accuracy=row.get("accuracy_preserved"),
            )
        )
    summary_md_path = output_dir / "summary.md"
    summary_md_path.write_text("\n".join(summary_md_lines) + "\n")

    print(
        json.dumps(
            {
                "success": True,
                "dry_run": args.dry_run,
                "output_dir": str(output_dir),
                "csv": str(csv_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "num_configs": len(configs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
