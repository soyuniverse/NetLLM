#!/usr/bin/env python3
"""Generalization spot-check: does config D's accuracy improvement +
latency reduction hold on Wu2017 (unseen during this checkpoint's Jin2022
fine-tuning), not just Jin2022?

This mirrors the generalization-evaluation framing NetLLM itself uses
(cross-dataset splits) -- it is a spot-check on 200 evenly-spaced samples
from Wu2017's 1,395-sample test split, not a full benchmark, and the
checkpoint was fine-tuned on Jin2022, not Wu2017, so this measures
distribution-shift robustness, not an apples-to-apples comparison with
the Jin2022 numbers elsewhere in this session.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(
    os.environ.get("NETLLM_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from netllm_litevlm.evaluation.vp_metrics import evaluate_vp_metrics
from netllm_litevlm.selectors import RecentKSelector
from netllm_litevlm.speculative import SpeculativeBlockVerifyPipeline
from netllm_litevlm.vp.checkpoint_era_runtime import UPSTREAM_VP_ROOT, load_checkpoint_era_model
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline

CHECKPOINT = Path("/root/NetLLM-assets/checkpoints/try_llama2_7b")
DATASET_ROOT = Path("/root/NetLLM-source/viewport_prediction/data/viewports/Wu2017")
OUTPUT = PROJECT_ROOT / "experiments/vp/wu2017_generalization_spotcheck"
NUM_SAMPLES = 200
DEVICE = "cuda:0"
DTYPE = torch.float16


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(str(tmp), str(path))


def run_pipeline(pipeline, samples, normalize_data, denormalize_data):
    predictions, targets, latencies_ms = [], [], []
    for history_np, future_np, info in samples:
        history = normalize_data(
            torch.from_numpy(history_np).unsqueeze(0), "Wu2017"
        ).to(DEVICE, dtype=DTYPE)
        future_raw = torch.from_numpy(future_np).unsqueeze(0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, target = pipeline.inference(history, future_raw, info)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        predictions.append(denormalize_data(prediction.float(), "Wu2017").cpu())
        targets.append(target.float().cpu())
    metrics = evaluate_vp_metrics(
        torch.cat(predictions, dim=0), torch.cat(targets, dim=0), checkpoint_available=True
    )
    return {
        "mae": metrics.mae,
        "corrected_rmse": metrics.corrected_rotation_aware_rmse,
        "latency_median_ms": sorted(latencies_ms)[len(latencies_ms) // 2],
    }


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "spotcheck_result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {result_path}")

    model, checkpoint_loaded = load_checkpoint_era_model(
        device=DEVICE, dtype=DTYPE, rank=32, checkpoint_path=CHECKPOINT, seed=0,
    )
    if not checkpoint_loaded:
        raise RuntimeError("expected the real checkpoint to load")

    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from config import cfg
    from dataset.load_dataset import create_dataset
    from utils.normalize import denormalize_data, normalize_data

    cfg.dataset["Wu2017"] = str(DATASET_ROOT)
    test_dataset = create_dataset(
        "Wu2017", his_window=10, fut_window=20, trim_head=30, trim_tail=60,
        frequency=5, step=15, include=["test"],
    )[0]
    total = len(test_dataset)
    stride = max(1, total // NUM_SAMPLES)
    indices = list(range(0, total, stride))[:NUM_SAMPLES]
    samples = [test_dataset[i] for i in indices]

    results = {
        "dataset": "Wu2017",
        "note": "checkpoint fine-tuned on Jin2022, not Wu2017 -- this is a "
        "distribution-shift generalization spot-check, not an apples-to-apples "
        "comparison with the Jin2022 numbers elsewhere in this session",
        "test_split_total": total,
        "num_samples": len(samples),
        "sample_stride": stride,
    }

    baseline_pipeline = LlamaOldSelectablePipeline(model)
    a_result = run_pipeline(baseline_pipeline, samples, normalize_data, denormalize_data)
    results["A_baseline"] = {**a_result, "target_forward_count_avg": 20.0}

    combined_pipeline = SpeculativeBlockVerifyPipeline(
        model, selector=RecentKSelector(2), gamma=8, acceptance_threshold=0.35
    )
    forward_counts = []
    predictions, targets, latencies_ms = [], [], []
    for history_np, future_np, info in samples:
        history = normalize_data(
            torch.from_numpy(history_np).unsqueeze(0), "Wu2017"
        ).to(DEVICE, dtype=DTYPE)
        future_raw = torch.from_numpy(future_np).unsqueeze(0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, target = combined_pipeline.inference(history, future_raw, info)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        forward_counts.append(combined_pipeline.target_forward_count)
        predictions.append(denormalize_data(prediction.float(), "Wu2017").cpu())
        targets.append(target.float().cpu())
    d_metrics = evaluate_vp_metrics(
        torch.cat(predictions, dim=0), torch.cat(targets, dim=0), checkpoint_available=True
    )
    results["D_recentk2_plus_speculative"] = {
        "mae": d_metrics.mae,
        "corrected_rmse": d_metrics.corrected_rotation_aware_rmse,
        "latency_median_ms": sorted(latencies_ms)[len(latencies_ms) // 2],
        "target_forward_count_avg": sum(forward_counts) / len(forward_counts),
    }

    results["mae_improvement_holds"] = (
        results["D_recentk2_plus_speculative"]["mae"] <= results["A_baseline"]["mae"]
    )
    results["latency_reduction_holds"] = (
        results["D_recentk2_plus_speculative"]["latency_median_ms"]
        < results["A_baseline"]["latency_median_ms"]
    )

    write_json(result_path, results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
