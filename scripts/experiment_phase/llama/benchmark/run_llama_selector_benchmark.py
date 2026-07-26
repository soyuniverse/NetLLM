#!/usr/bin/env python3
"""Recovered-artifact controlled selector benchmark for old Llama NetLLM."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
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
SOURCE_COMMIT = "ee4d8726898610e4ae7df08bdd26728cafb4701f"
CONFIGS = [
    "original",
    "identity_keep10",
    "recent_k8",
    "recent_k6",
    "recent_k4",
    "recent_k2",
]
SUMMARY_FIELDS = [
    "model", "source_commit", "checkpoint", "selector", "selected_tokens",
    "keep_ratio", "mae", "upstream_rmse", "corrected_rmse",
    "mean_angular_error", "evaluation_loss", "latency_median_ms",
    "latency_p95_ms", "selector_latency_median_ms", "peak_allocated_mib",
    "peak_reserved_mib", "plm_forward_count",
    "processed_sequence_length_sum", "sample_count",
    "comparative_quality_valid", "paper_reproduction_valid",
]


def configure_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=str(path.parent), delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = stream.name
    os.replace(temporary, path)


def percentile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return float(values[lo] * (hi - rank) + values[hi] * (rank - lo))


def tensor_sha256(tensor):
    value = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_runtime():
    configure_offline()
    sys.path.insert(0, str(PROJECT / "src"))
    sys.path.insert(0, str(ERA_VP))
    from config import cfg
    from dataset.load_dataset import create_dataset
    from models.low_rank import peft_model
    from models.old.llama import LlamaTaskHeadModel2
    from models.old.networking_head import SimpleLinearTaskHead
    from models.old.pipeline import EmbeddingForViewportPrediction
    from transformers import LlamaConfig

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
        using_teaching_forcing=False, using_multimodal=False, dataset="Jin2022",
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

    cfg.dataset["Jin2022"] = str(DATASET_ROOT)
    dataset = create_dataset(
        "Jin2022", his_window=10, fut_window=20, trim_head=30, trim_tail=60,
        frequency=5, step=15, include=["test"],
    )[0]
    return model, dataset


def prepare_sample(dataset, index):
    sys.path.insert(0, str(ERA_VP))
    from utils.normalize import normalize_data
    history_np, future_np, info = dataset[index]
    history = torch.from_numpy(history_np).unsqueeze(0)
    future_raw = torch.from_numpy(future_np).unsqueeze(0)
    history = normalize_data(history, "Jin2022").to(
        "cuda:0", dtype=torch.float16
    )
    future_normalized = normalize_data(future_raw, "Jin2022").to(
        "cuda:0", dtype=torch.float16
    )
    return history, future_raw, future_normalized, info


def make_path(model, name):
    from netllm_litevlm.selectors import IdentitySelector, RecentKSelector
    from netllm_litevlm.vp.llama_old_selectable_pipeline import (
        LlamaOldSelectablePipeline,
    )
    if name == "original":
        return model
    if name == "identity_keep10":
        return LlamaOldSelectablePipeline(model, IdentitySelector()).half().eval()
    return LlamaOldSelectablePipeline(
        model, RecentKSelector(int(name.rsplit("k", 1)[1]))
    ).half().eval()


def execute(path, name, history, future_raw, info):
    if name == "original":
        prediction, _ = path.inference(history, future_raw, info)
        selector_ms = 0.0
    else:
        prediction, _ = path.inference(history, future_raw, info)
        selector_ms = path.selector_elapsed_ms()
    return prediction, selector_ms


def metric_row(index, info, name, prediction_norm, future_raw,
               future_normalized, latency_ms, selector_ms):
    scale = torch.tensor([180.0, 90.0, 180.0], device="cuda:0")
    prediction = prediction_norm.float() * scale
    target = future_raw.to("cuda:0", dtype=torch.float32)
    raw_error = torch.abs(prediction - target)
    circular_error = torch.minimum(
        torch.remainder(raw_error, 360.0),
        360.0 - torch.remainder(raw_error, 360.0),
    )
    elements = prediction.numel()
    evaluation_loss = torch.nn.functional.mse_loss(
        prediction_norm, future_normalized
    )
    return {
        "sample_id": index,
        "video": int(info[0]), "user": int(info[1]), "timestep": int(info[2]),
        "selector": name,
        "mae": float(circular_error.mean().item()),
        "upstream_mse": float(raw_error.square().mean().item()),
        "corrected_mse": float(circular_error.square().mean().item()),
        "mean_angular_error": float(circular_error.mean().item()),
        "evaluation_loss": float(evaluation_loss.item()),
        "absolute_error_sum": float(circular_error.sum().item()),
        "upstream_squared_error_sum": float(raw_error.square().sum().item()),
        "corrected_squared_error_sum": float(circular_error.square().sum().item()),
        "element_count": elements,
        "latency_ms": latency_ms,
        "selector_latency_ms": selector_ms,
        "prediction_sha256": tensor_sha256(prediction_norm),
        "finite": bool(torch.isfinite(prediction_norm).all().item()),
    }


def summarize(rows, name, peak_allocated, peak_reserved):
    k = 10 if name in ("original", "identity_keep10") else int(name.rsplit("k", 1)[1])
    n = len(rows)
    elements = sum(int(row["element_count"]) for row in rows)
    mae_value = sum(float(row["absolute_error_sum"]) for row in rows) / elements
    upstream = math.sqrt(
        sum(float(row["upstream_squared_error_sum"]) for row in rows) / elements
    )
    corrected = math.sqrt(
        sum(float(row["corrected_squared_error_sum"]) for row in rows) / elements
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    selector_latencies = [float(row["selector_latency_ms"]) for row in rows]
    per_sample_processed = sum(range(k, k + 20))
    return {
        "model": "Llama-2-7b-hf NetLLM recovered artifact",
        "source_commit": SOURCE_COMMIT,
        "checkpoint": str(CHECKPOINT),
        "selector": name,
        "selected_tokens": k,
        "keep_ratio": k / 10.0,
        "mae": mae_value,
        "upstream_rmse": upstream,
        "corrected_rmse": corrected,
        "mean_angular_error": mae_value,
        "evaluation_loss": statistics.fmean(
            float(row["evaluation_loss"]) for row in rows
        ),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "selector_latency_median_ms": statistics.median(selector_latencies),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "plm_forward_count": 20 * n,
        "processed_sequence_length_sum": per_sample_processed * n,
        "sample_count": n,
        "comparative_quality_valid": True,
        "paper_reproduction_valid": False,
    }


def run_configuration(model, dataset, name, count, output_dir, resume):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "per_sample_metrics.csv"
    rows = []
    if resume and rows_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    completed = {int(row["sample_id"]) for row in rows}
    path = make_path(model, name)

    for warmup_index in range(min(5, count)):
        history, future_raw, _, info = prepare_sample(dataset, warmup_index)
        with torch.inference_mode():
            execute(path, name, history, future_raw, info)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fields = [
        "sample_id", "video", "user", "timestep", "selector", "mae",
        "upstream_mse", "corrected_mse", "mean_angular_error",
        "evaluation_loss", "absolute_error_sum",
        "upstream_squared_error_sum", "corrected_squared_error_sum",
        "element_count", "latency_ms", "selector_latency_ms",
        "prediction_sha256", "finite",
    ]
    for index in range(count):
        if index in completed:
            continue
        history, future_raw, future_normalized, info = prepare_sample(
            dataset, index
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, selector_ms = execute(
                path, name, history, future_raw, info
            )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        row = metric_row(
            index, info, name, prediction, future_raw, future_normalized,
            latency_ms, selector_ms,
        )
        if not row["finite"]:
            raise RuntimeError("non-finite output at {} {}".format(name, index))
        rows.append(row)
        if (index + 1) % 50 == 0 or index + 1 == count:
            rows.sort(key=lambda item: int(item["sample_id"]))
            atomic_csv(rows_path, rows, fields)
            atomic_json(output_dir / "progress.json", {
                "selector": name, "completed_sample_ids": [
                    int(item["sample_id"]) for item in rows
                ], "complete": len(rows) == count,
            })
            print("{} {}/{}".format(name, len(rows), count), flush=True)

    peak_allocated = torch.cuda.max_memory_allocated() / 1048576.0
    peak_reserved = torch.cuda.max_memory_reserved() / 1048576.0
    summary = summarize(rows, name, peak_allocated, peak_reserved)
    atomic_json(output_dir / "summary.json", summary)
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    model, dataset = load_runtime()
    count = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    root = PROJECT / "experiments/vp/llama_benchmark" / args.mode
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_status.txt").exists() and not args.resume:
        raise RuntimeError("refusing to overwrite existing benchmark runtime")

    started = time.perf_counter()
    summaries, all_rows = [], []
    for name in CONFIGS:
        summary, rows = run_configuration(
            model, dataset, name, count, root / name, args.resume
        )
        summaries.append(summary); all_rows.extend(rows)
        atomic_json(root / "benchmark_summary.json", {
            "mode": args.mode, "sample_count_per_configuration": count,
            "complete_configurations": [x["selector"] for x in summaries],
            "summaries": summaries, "partial": len(summaries) != len(CONFIGS),
        })
        atomic_csv(root / "benchmark_summary.csv", summaries, SUMMARY_FIELDS)

    identity = next(x for x in summaries if x["selector"] == "identity_keep10")
    original = next(x for x in summaries if x["selector"] == "original")
    identity_diffs = {
        key: abs(float(original[key]) - float(identity[key]))
        for key in ["mae", "upstream_rmse", "corrected_rmse",
                    "mean_angular_error", "evaluation_loss"]
    }
    original_sha = [
        x["prediction_sha256"] for x in all_rows if x["selector"] == "original"
    ]
    identity_sha = [
        x["prediction_sha256"] for x in all_rows
        if x["selector"] == "identity_keep10"
    ]
    success = (
        count > 0 and all(int(x["sample_count"]) == count for x in summaries)
        and max(identity_diffs.values()) <= 1e-6
        and original_sha == identity_sha
    )
    all_fields = list(all_rows[0].keys())
    atomic_csv(root / "per_sample_metrics.csv", all_rows, all_fields)
    elapsed = time.perf_counter() - started
    final = {
        "mode": args.mode, "success": success, "partial": False,
        "sample_count_per_configuration": count,
        "configuration_count": len(CONFIGS), "elapsed_seconds": elapsed,
        "identity_metric_differences": identity_diffs,
        "identity_prediction_sha256_all_equal": original_sha == identity_sha,
        "summaries": summaries,
    }
    atomic_json(root / "benchmark_summary.json", final)
    with (root / "run_status.txt").open("w", encoding="utf-8") as stream:
        stream.write("success={}\npartial=false\nelapsed_seconds={}\n".format(
            str(success).lower(), elapsed
        ))
    print(json.dumps(final, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
