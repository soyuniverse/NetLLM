#!/usr/bin/env python3
"""Real 7B integration smoke for SpeculativeBlockVerifyPipeline.

Uses the base Llama2-7b weights + peft_model(base, "llama", 32) (PEFT's own
default init -- LoRA B=0, zero effect until trained) + a freshly
random-initialized SimpleLinearTaskHead, NOT the fine-tuned VP checkpoint
(unavailable in this instance -- see
docs/experiment_phase/speculative/PHASE_A_DESIGN.md). Prediction quality
(MAE etc.) is therefore meaningless and is not measured here. This smoke
only checks structural/control-flow correctness (does block verification
behave the same as the tiny-model gate tests at real 7B scale?) and
resource usage (memory, wall-clock as a reference number only -- no
speedup claim).
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

from netllm_litevlm.speculative import SpeculativeBlockVerifyPipeline
from netllm_litevlm.vp.checkpoint_era_runtime import (
    DEFAULT_BASE_MODEL_PATH,
    load_checkpoint_era_model,
)
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline

OUTPUT = PROJECT_ROOT / "experiments/vp/llama_7b_speculative_smoke"
FUT_WINDOW = 20
HIS_WINDOW = 10
DEVICE = "cuda:0"
DTYPE = torch.float16


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(str(tmp), str(path))


def synthetic_history(seed: int) -> torch.Tensor:
    """Real-angle-unit synthetic trajectory (roll/yaw in [-180,180],
    pitch in [-90,90]) with a smooth, realistic-scale velocity -- not
    real Jin2022 data (unavailable), just a plausible input scale."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(HIS_WINDOW).float()
    roll = 30.0 * torch.sin(t / 3.0) + torch.randn(HIS_WINDOW, generator=generator) * 2.0
    pitch = 15.0 * torch.cos(t / 4.0) + torch.randn(HIS_WINDOW, generator=generator) * 1.0
    yaw = 5.0 * t + torch.randn(HIS_WINDOW, generator=generator) * 2.0
    return torch.stack([roll, pitch, yaw], dim=-1).unsqueeze(0)


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)


def peak_mib():
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(DEVICE) / 1048576.0,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(DEVICE) / 1048576.0,
    }


def main():
    assert torch.cuda.is_available(), "CUDA required for this smoke"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "smoke_result.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {result_path}")

    load_started = time.perf_counter()
    model, checkpoint_loaded = load_checkpoint_era_model(
        device=DEVICE, dtype=DTYPE, rank=32, fut_window=FUT_WINDOW,
        checkpoint_path=None, seed=0,
    )
    load_seconds = time.perf_counter() - load_started
    if checkpoint_loaded:
        raise RuntimeError("expected random-head assembly, got a loaded checkpoint")

    sys.path.insert(0, str(PROJECT_ROOT / "third_party/netllm_upstream/viewport_prediction"))
    from utils.normalize import normalize_data

    histories = []
    for seed in range(5):
        raw = synthetic_history(seed)
        histories.append(normalize_data(raw, "Jin2022").to(DEVICE, dtype=DTYPE))

    result = {
        "scope": "structural/resource smoke only -- base Llama2-7b weights + "
        "PEFT-default-init LoRA(rank=32) + randomly-initialized "
        "SimpleLinearTaskHead, NOT the fine-tuned VP checkpoint",
        "quality_claim_valid": False,
        "speedup_claim_valid": False,
        "checkpoint_loaded": checkpoint_loaded,
        "base_model_path": str(DEFAULT_BASE_MODEL_PATH),
        "model_load_seconds": load_seconds,
        "device": DEVICE,
        "dtype": "float16",
        "fut_window": FUT_WINDOW,
        "num_samples": len(histories),
    }

    # --- Gate 1: threshold=0 equivalence, and the real 7B floating-point
    # reassociation noise floor for the chained KV-cache loop (tiny-model
    # floor measured in PHASE_A_DESIGN.md / test_block_verify.py: ~1e-7
    # fp32, ~1e-3 fp16 -- this measures the *same* quantity at 7B scale).
    baseline = LlamaOldSelectablePipeline(model)
    spec_zero = SpeculativeBlockVerifyPipeline(model, gamma=4, acceptance_threshold=0.0)
    zero_threshold_diffs = []
    zero_threshold_forward_counts = []
    for history in histories:
        with torch.inference_mode():
            expected = baseline.auto_regressive(history, None)
            actual = spec_zero.auto_regressive(history, None)
        zero_threshold_diffs.append((actual.float() - expected.float()).abs().max().item())
        zero_threshold_forward_counts.append(spec_zero.target_forward_count)
    result["gate_threshold_zero"] = {
        "max_abs_diff_per_sample": zero_threshold_diffs,
        "max_abs_diff": max(zero_threshold_diffs),
        "target_forward_count_per_sample": zero_threshold_forward_counts,
        "target_forward_count_matches_baseline": all(
            count == FUT_WINDOW for count in zero_threshold_forward_counts
        ),
        "draft_forward_count": spec_zero.draft_forward_count,
    }

    # --- Gate 2: a generously large threshold (task head ends in Tanh, so
    # normalized-space outputs are bounded to [-1,1]^3; a naive
    # velocity-extrapolation draft on this synthetic trajectory stays
    # within a few units of that, so this threshold is chosen to
    # deterministically force acceptance regardless of the specific
    # (untrained) random head weights, exactly as in the tiny-model gate
    # test) to confirm forward-count reduction reproduces at 7B scale.
    large_threshold = 3.0
    spec_large = SpeculativeBlockVerifyPipeline(model, gamma=4, acceptance_threshold=large_threshold)
    large_threshold_forward_counts = []
    large_threshold_accept_counts = []
    non_finite = False
    for history in histories:
        with torch.inference_mode():
            prediction = spec_large.auto_regressive(history, None)
        if not torch.isfinite(prediction).all():
            non_finite = True
        large_threshold_forward_counts.append(spec_large.target_forward_count)
        large_threshold_accept_counts.append(list(spec_large.accepted_per_iteration))
    result["gate_threshold_large"] = {
        "threshold": large_threshold,
        "target_forward_count_per_sample": large_threshold_forward_counts,
        "reduced_vs_baseline": all(
            count < FUT_WINDOW for count in large_threshold_forward_counts
        ),
        "accepted_per_iteration_per_sample": large_threshold_accept_counts,
        "non_finite_output_observed": non_finite,
        "draft_forward_count": spec_large.draft_forward_count,
    }

    # --- gamma sweep: peak GPU memory + OOM check, one 20-step inference each.
    gamma_sweep = {}
    for gamma in (2, 4, 8):
        entry = {"gamma": gamma}
        try:
            pipeline = SpeculativeBlockVerifyPipeline(model, gamma=gamma, acceptance_threshold=large_threshold)
            reset_peak()
            with torch.inference_mode():
                prediction = pipeline.auto_regressive(histories[0], None)
            entry.update(peak_mib())
            entry["target_forward_count"] = pipeline.target_forward_count
            entry["oom"] = False
            entry["finite"] = bool(torch.isfinite(prediction).all().item())
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            entry["oom"] = True
            entry["error"] = str(error)
        gamma_sweep[str(gamma)] = entry
    result["gamma_memory_sweep"] = gamma_sweep

    # --- baseline peak memory, for reference alongside the sweep above.
    reset_peak()
    with torch.inference_mode():
        baseline.auto_regressive(histories[0], None)
    result["baseline_peak_memory"] = peak_mib()

    # --- wall-clock: one 20-step inference, reference numbers only.
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        baseline.auto_regressive(histories[0], None)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - started) * 1000.0

    spec_reference = SpeculativeBlockVerifyPipeline(model, gamma=4, acceptance_threshold=large_threshold)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        spec_reference.auto_regressive(histories[0], None)
    torch.cuda.synchronize()
    speculative_ms = (time.perf_counter() - started) * 1000.0
    result["latency_reference_ms_no_speedup_claim"] = {
        "baseline_one_inference_ms": baseline_ms,
        "speculative_gamma4_large_threshold_one_inference_ms": speculative_ms,
        "speculative_target_forward_count": spec_reference.target_forward_count,
    }

    write_json(result_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
