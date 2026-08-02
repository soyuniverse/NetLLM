#!/usr/bin/env python3
"""Real 7B smoke for AttentionTopKSelector, K in {8,6,4,2}, vs RecentKSelector.

Uses the real recovered checkpoint (unlike the earlier
run_llama_7b_speculative_smoke.py, which used a random head) and the real
Jin2022 test split, so this produces a real preliminary MAE comparison,
not just a structural/control-flow check. Whichever selector wins on MAE
at a given K, the number is recorded as measured -- no claim either way
is asserted in this script.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(
    os.environ.get("NETLLM_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from netllm_litevlm.evaluation.vp_metrics import evaluate_vp_metrics
from netllm_litevlm.selectors import AttentionTopKSelector, RecentKSelector
from netllm_litevlm.vp.checkpoint_era_runtime import UPSTREAM_VP_ROOT, load_checkpoint_era_model
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline

CHECKPOINT = Path("/root/NetLLM-assets/checkpoints/try_llama2_7b")
DATASET_ROOT = Path("/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022")
OUTPUT = PROJECT_ROOT / "experiments/vp/attention_topk_7b_smoke"
NUM_SAMPLES = 50
K_VALUES = (8, 6, 4, 2)
DEVICE = "cuda:0"
DTYPE = torch.float16


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(str(tmp), str(path))


def run_selector(model, selector, samples, normalize_data, denormalize_data):
    pipeline = LlamaOldSelectablePipeline(model, selector=selector)
    predictions, targets, latencies_ms = [], [], []
    for history_np, future_np, info in samples:
        history = normalize_data(
            torch.from_numpy(history_np).unsqueeze(0), "Jin2022"
        ).to(DEVICE, dtype=DTYPE)
        future_raw = torch.from_numpy(future_np).unsqueeze(0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, target = pipeline.inference(history, future_raw, info)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        predictions.append(denormalize_data(prediction.float(), "Jin2022").cpu())
        targets.append(target.float().cpu())
    metrics = evaluate_vp_metrics(
        torch.cat(predictions, dim=0), torch.cat(targets, dim=0), checkpoint_available=True
    )
    return {
        "mae": metrics.mae,
        "corrected_rmse": metrics.corrected_rotation_aware_rmse,
        "latency_median_ms": sorted(latencies_ms)[len(latencies_ms) // 2],
        "selector_forward_count": getattr(selector, "selector_forward_count", None),
    }


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required for this smoke"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "smoke_result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {result_path}")

    model, checkpoint_loaded = load_checkpoint_era_model(
        device=DEVICE, dtype=DTYPE, rank=32, checkpoint_path=CHECKPOINT, seed=0,
    )
    if not checkpoint_loaded:
        raise RuntimeError("expected the real checkpoint to load")
    llama_model = model.embedding_model.plm.model.model

    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from config import cfg
    from dataset.load_dataset import create_dataset
    from utils.normalize import denormalize_data, normalize_data

    cfg.dataset["Jin2022"] = str(DATASET_ROOT)
    test_dataset = create_dataset(
        "Jin2022", his_window=10, fut_window=20, trim_head=30, trim_tail=60,
        frequency=5, step=15, include=["test"],
    )[0]
    samples = [test_dataset[i] for i in range(min(NUM_SAMPLES, len(test_dataset)))]

    baseline_result = run_selector(model, None, samples, normalize_data, denormalize_data)

    results = {
        "checkpoint_loaded": checkpoint_loaded,
        "num_samples": len(samples),
        "baseline": baseline_result,
        "k_comparison": {},
    }

    for k in K_VALUES:
        attention_selector = AttentionTopKSelector(k=k, llama_model=llama_model)
        attention_result = run_selector(
            model, attention_selector, samples, normalize_data, denormalize_data
        )

        recent_k_selector = RecentKSelector(k=k)
        recent_k_result = run_selector(
            model, recent_k_selector, samples, normalize_data, denormalize_data
        )

        results["k_comparison"][str(k)] = {
            "attention_top_k": attention_result,
            "recent_k": recent_k_result,
            "attention_mae_minus_recent_mae": attention_result["mae"] - recent_k_result["mae"],
        }

    write_json(result_path, results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
