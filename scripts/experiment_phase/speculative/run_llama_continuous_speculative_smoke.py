#!/usr/bin/env python3
"""Single-sample continuous VP draft-and-verify smoke."""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path("/root/NetLLM")
BASE = Path("/root/NetLLM-assets/llama/base")
CHECKPOINT = Path("/root/NetLLM-assets/checkpoints/try_llama2_7b")
ERA_VP = Path("/root/NetLLM-source-checkpoint-era/viewport_prediction")
DATASET_ROOT = Path(
    "/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022"
)
OUTPUT = PROJECT / "experiments/vp/llama_speculative_smoke"
THRESHOLD = 0.1


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def configure():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    sys.path.insert(0, str(PROJECT / "src"))
    sys.path.insert(0, str(ERA_VP))


def load_runtime():
    from config import cfg
    from dataset.load_dataset import create_dataset
    from models.low_rank import peft_model
    from models.old.llama import LlamaTaskHeadModel2
    from models.old.networking_head import SimpleLinearTaskHead
    from models.old.pipeline import EmbeddingForViewportPrediction
    from transformers import LlamaConfig

    started = time.perf_counter()
    config = LlamaConfig.from_pretrained(str(BASE), local_files_only=True)
    base = LlamaTaskHeadModel2.from_pretrained(
        str(BASE), config=config, local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    plm = peft_model(base, "llama", 32)
    plm.set_task_head(SimpleLinearTaskHead(4096, 3, 20).to("cuda:0"))
    model = EmbeddingForViewportPrediction(
        plm, fut_window=20, device="cuda:0", embed_size=4096, frequency=5,
        using_teaching_forcing=False, using_multimodal=False,
        dataset="Jin2022",
    )
    model.plm.load_adapter(str(CHECKPOINT), adapter_name="default")
    model.plm.set_adapter("default")
    state = torch.load(
        CHECKPOINT / "modules_except_plm.bin",
        map_location="cpu", weights_only=True,
    )
    incompatible = model.embedding_model.modules_except_plm.load_state_dict(
        state, strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict non-PLM load mismatch")
    model.half().eval()
    load_seconds = time.perf_counter() - started

    cfg.dataset["Jin2022"] = str(DATASET_ROOT)
    dataset = create_dataset(
        "Jin2022", his_window=10, fut_window=20, trim_head=30, trim_tail=60,
        frequency=5, step=15, include=["test"],
    )[0]
    return model, dataset, load_seconds


def main():
    configure()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "speculative_smoke.json"
    status_path = OUTPUT / "run_status.txt"
    if result_path.exists() or status_path.exists():
        raise RuntimeError("refusing to overwrite an existing speculative smoke")

    from netllm_litevlm.speculative import (
        ContinuousDraftVerify,
        RecentVelocityDraft,
        TargetOutput,
    )
    from utils.normalize import normalize_data

    model, dataset, load_seconds = load_runtime()
    history_np, future_np, info = dataset[0]
    history_raw = torch.from_numpy(history_np).unsqueeze(0)
    future_raw = torch.from_numpy(future_np).unsqueeze(0)
    history = normalize_data(history_raw, "Jin2022").to(
        "cuda:0", dtype=torch.float16
    )

    torch.cuda.synchronize()
    with torch.inference_mode():
        warmup, _ = model.inference(history, future_raw, info)
    if not torch.isfinite(warmup).all():
        raise RuntimeError("non-finite target warmup")

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        baseline, _ = model.inference(history, future_raw, info)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - started) * 1000.0

    def target_predictor(history_value, steps, context):
        del steps, context
        prediction, _ = model.inference(history_value, future_raw, info)
        return TargetOutput(
            coordinates=prediction,
            forward_count=20,
            metadata={"model": "checkpoint-era Llama2 NetLLM"},
        )

    prototype = ContinuousDraftVerify(
        draft_model=RecentVelocityDraft(),
        target_predictor=target_predictor,
        threshold=THRESHOLD,
        baseline_target_forward_count=20,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = prototype.run(
            history, steps=20,
            context={"dataset": "Jin2022", "sample_index": 0},
        )
    torch.cuda.synchronize()
    prototype_ms = (time.perf_counter() - started) * 1000.0

    output = result.output
    success = (
        result.control_flow_valid
        and list(output.shape) == [1, 20, 3]
        and bool(torch.isfinite(output).all().item())
        and result.target.forward_count == 20
        and result.draft.forward_count == 1
    )
    payload = {
        "prototype": "Continuous VP Draft-and-Verify Prototype",
        "success": success,
        "dataset": "Jin2022",
        "sample_index": 0,
        "sample_identity": {
            "video": int(info[0]), "user": int(info[1]),
            "timestep": int(info[2]),
        },
        "history_shape": list(history.shape),
        "future_shape": list(future_raw.shape),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "threshold_normalized_max_abs": THRESHOLD,
        "accepted_prefix_length": (
            result.verification.accepted_prefix_length
        ),
        "first_rejected_index": result.verification.first_rejected_index,
        "per_step_max_absolute_error": (
            result.verification.per_step_max_absolute_error
        ),
        "draft": "deterministic recent-velocity extrapolation",
        "draft_forward_count": result.draft.forward_count,
        "target_forward_count": result.target.forward_count,
        "baseline_target_forward_count": (
            result.baseline_target_forward_count
        ),
        "model_load_latency_seconds": load_seconds,
        "baseline_target_latency_ms": baseline_ms,
        "prototype_latency_ms": prototype_ms,
        "latency_ratio_baseline_over_prototype": (
            baseline_ms / prototype_ms
        ),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1048576.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 1048576.0
        ),
        "control_flow_valid": result.control_flow_valid,
        "speedup_claim_valid": result.speedup_claim_valid,
        "quality_claim_valid": False,
        "technical_smoke_only": True,
        "target_block_verification_optimized": False,
    }
    write_json(result_path, payload)
    with status_path.open("x", encoding="utf-8") as stream:
        stream.write("success={}\n".format(str(success).lower()))
        stream.write("control_flow_valid={}\n".format(
            str(result.control_flow_valid).lower()
        ))
        stream.write("speedup_claim_valid=false\n")
    print(json.dumps(payload, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
