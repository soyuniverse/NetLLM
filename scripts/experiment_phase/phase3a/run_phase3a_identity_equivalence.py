#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path("/workspace/NetLLM")
SOURCE_ROOT = Path("/workspace/NetLLM-source")
VP_ROOT = SOURCE_ROOT / "viewport_prediction"
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACT_ROOT = Path("/workspace/NetLLM-artifacts/plms")
ARTIFACT_PATH = ARTIFACT_ROOT / "gpt2/base"
TOLERANCE = 1e-7


def tensor_summary(tensor):
    import torch

    detached = tensor.detach()
    numeric = detached.to(dtype=torch.float64)
    payload = detached.contiguous().cpu().numpy().tobytes()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": bool(torch.isfinite(numeric).all().item()),
        "requires_grad": bool(tensor.requires_grad),
        "min": float(numeric.min().item()),
        "max": float(numeric.max().item()),
        "mean": float(numeric.mean().item()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class PathMonitor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.sequence_lengths = []
        self.past_key_values_passed = []
        self.cache_returned = []
        self.initial_layernorm_output_shape = None
        self.handles = []

    def layernorm_hook(self, module, inputs, output):
        self.initial_layernorm_output_shape = list(output.shape)

    def plm_pre_hook(self, module, args, kwargs):
        self.sequence_lengths.append(int(kwargs["inputs_embeds"].shape[1]))
        self.past_key_values_passed.append(kwargs.get("past_key_values") is not None)

    def plm_hook(self, module, inputs, output):
        self.cache_returned.append(output.past_key_values is not None)

    def __enter__(self):
        self.handles = [
            self.pipeline.embed_ln.register_forward_hook(self.layernorm_hook),
            self.pipeline.plm.register_forward_pre_hook(self.plm_pre_hook, with_kwargs=True),
            self.pipeline.plm.register_forward_hook(self.plm_hook),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()

    def report(self):
        return {
            "initial_layernorm_output_shape": self.initial_layernorm_output_shape,
            "sequence_lengths": self.sequence_lengths,
            "plm_forward_count": len(self.sequence_lengths),
            "past_key_values_passed": self.past_key_values_passed,
            "cache_returned": self.cache_returned,
            "cache_reused": any(self.past_key_values_passed),
        }


def configure_offline_environment():
    values = {
        "HF_HOME": "/workspace/NetLLM-artifacts/hf_cache",
        "TRANSFORMERS_CACHE": "/workspace/NetLLM-artifacts/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "socks5://127.0.0.1:9",
    }
    os.environ.update(values)
    sys.dont_write_bytecode = True


def max_difference(left, right):
    import torch

    difference = (left - right).abs()
    flat_index = int(difference.argmax().item())
    index = []
    remaining = flat_index
    for size in reversed(tuple(int(value) for value in difference.shape)):
        index.append(remaining % size)
        remaining //= size
    index.reverse()
    return {
        "max_absolute_difference": float(difference.max().item()),
        "max_difference_index": index,
        "left_value_at_max": float(left[tuple(index)].item()),
        "right_value_at_max": float(right[tuple(index)].item()),
        "exact_equal": bool(torch.equal(left, right)),
        "within_tolerance": bool(
            torch.allclose(left, right, rtol=0.0, atol=TOLERANCE)
        ),
        "rtol": 0.0,
        "atol": TOLERANCE,
    }


def run_path(label, callable_path, pipeline):
    import torch

    monitor = PathMonitor(pipeline)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with monitor, torch.inference_mode():
        prediction, ground_truth = callable_path()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "output": tensor_summary(prediction),
        "trace": monitor.report(),
        "elapsed_seconds": elapsed,
    }


def run_equivalence(output_path: Optional[Path] = None) -> Dict[str, Any]:
    configure_offline_environment()
    os.chdir(VP_ROOT)
    sys.path.insert(0, str(VP_ROOT))
    sys.path.insert(0, str(SRC_ROOT))

    import torch
    from torch.utils.data import DataLoader

    from config import cfg
    from dataset.load_dataset import create_dataset
    from models.networking_head import NetworkingHead
    from models.pipeline import Pipeline
    from netllm_litevlm.selectors import IdentitySelector
    from netllm_litevlm.vp.selectable_pipeline import SelectablePipeline
    from utils.normalize import normalize_data
    from utils.plms_utils import load_plm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not ARTIFACT_PATH.is_dir():
        raise RuntimeError(f"GPT-2 artifact is missing: {ARTIFACT_PATH}")

    device = torch.device("cuda:0")
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.set_grad_enabled(False)

    cfg.plms_dir = str(ARTIFACT_ROOT)
    model_path = Path(cfg.plms_dir) / "gpt2/base"
    load_start = time.perf_counter()
    plm, tokenizer, model_config = load_plm("gpt2", str(model_path), plm_size="base")
    model_load_seconds = time.perf_counter() - load_start
    plm = plm.to(device)
    networking_head = NetworkingHead(
        input_dim=plm.hidden_size,
        output_dim=3,
        fut_window=20,
    ).to(device)
    plm.set_networking_head(networking_head)
    original_pipeline = Pipeline(
        plm=plm,
        loss_func=None,
        fut_window=20,
        device=str(device),
        embed_size=1024,
        frequency=5,
        using_multimodal=False,
        dataset="Jin2022",
    )
    selectable_pipeline = SelectablePipeline(original_pipeline, selector=None)
    original_pipeline.eval()
    selectable_pipeline.eval()

    dataset_test = create_dataset(
        "Jin2022",
        his_window=10,
        fut_window=20,
        trim_head=30,
        trim_tail=60,
        frequency=5,
        step=15,
        include=["test"],
        for_track=False,
    )[0]
    history, future, video_user_info = next(
        iter(DataLoader(dataset_test, batch_size=1, shuffle=False))
    )
    video, user, timestep = [int(item.item()) for item in video_user_info]
    if (video, user, timestep) != (4, 83, 30):
        raise RuntimeError(f"Unexpected sample: {(video, user, timestep)}")
    csv_relative = Path(
        "viewport_prediction/data/viewports/Jin2022/video4/5Hz/simple_5Hz_user83.csv"
    )
    tracked = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "ls-files", "--error-unmatch", str(csv_relative)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).returncode == 0
    if not tracked:
        raise RuntimeError(f"Sample CSV is not Git tracked: {csv_relative}")

    normalized_history = normalize_data(history, "Jin2022").to(device)
    future = future.to(device)

    original = run_path(
        "original",
        lambda: original_pipeline.inference(
            normalized_history, future, video_user_info
        ),
        original_pipeline,
    )
    selectable_pipeline.set_selector(None)
    disabled = run_path(
        "selectable_disabled",
        lambda: selectable_pipeline.inference(
            normalized_history, future, video_user_info
        ),
        original_pipeline,
    )
    disabled_extension_trace = dict(selectable_pipeline.last_trace)

    identity_selector = IdentitySelector()
    selectable_pipeline.set_selector(identity_selector)
    identity = run_path(
        "selectable_identity",
        lambda: selectable_pipeline.inference(
            normalized_history, future, video_user_info
        ),
        original_pipeline,
    )
    identity_extension_trace = dict(selectable_pipeline.last_trace)
    selection = selectable_pipeline.last_selection_output
    if selection is None:
        raise RuntimeError("IdentitySelector did not produce SelectionOutput")

    comparisons = {
        "original_vs_disabled": max_difference(
            original["prediction"], disabled["prediction"]
        ),
        "original_vs_identity": max_difference(
            original["prediction"], identity["prediction"]
        ),
        "disabled_vs_identity": max_difference(
            disabled["prediction"], identity["prediction"]
        ),
    }

    expected_lengths = list(range(10, 30))
    path_reports = {}
    for path_result in (original, disabled, identity):
        path_reports[path_result["label"]] = {
            "output": path_result["output"],
            "trace": path_result["trace"],
            "elapsed_seconds": path_result["elapsed_seconds"],
        }

    success = all(
        comparison["max_absolute_difference"] <= TOLERANCE
        for comparison in comparisons.values()
    )
    success = success and all(
        path["output"]["shape"] == [1, 20, 3]
        and path["output"]["finite"]
        and path["trace"]["sequence_lengths"] == expected_lengths
        and path["trace"]["plm_forward_count"] == 20
        and not any(path["trace"]["past_key_values_passed"])
        for path in path_reports.values()
    )
    success = success and (
        selection.original_length == 10
        and selection.selected_length == 10
        and selection.selected_indices.tolist() == list(range(10))
        and selection.scores is None
        and selection.embeddings.dtype == normalized_history.dtype
        and selection.embeddings.device == normalized_history.device
        and identity_extension_trace["selector_call_count"] == 1
        and identity_extension_trace["selector_applied_to_feedback"] is False
    )

    report = {
        "phase": "3A",
        "success": bool(success),
        "tolerance": {"atol": TOLERANCE, "rtol": 0.0},
        "sample": {
            "dataset": "Jin2022",
            "split": "test",
            "dataset_index": 0,
            "dataset_length": len(dataset_test),
            "batch_size": 1,
            "video": video,
            "user": user,
            "timestep": timestep,
            "history_shape": list(history.shape),
            "future_shape": list(future.shape),
            "csv_path": str(SOURCE_ROOT / csv_relative),
            "csv_git_tracked": tracked,
        },
        "model": {
            "artifact_path": str(ARTIFACT_PATH),
            "loader": "utils.plms_utils.load_plm",
            "model_class": type(plm).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "hidden_size": plm.hidden_size,
            "layers": model_config.n_layer,
            "attention_heads": model_config.n_head,
            "dtype": str(next(plm.parameters()).dtype),
            "device": str(next(plm.parameters()).device),
            "shared_module_instance": True,
            "parameter_count": sum(parameter.numel() for parameter in plm.parameters()),
            "model_load_seconds": model_load_seconds,
        },
        "extension": {
            "class": type(selectable_pipeline).__name__,
            "composition": True,
            "upstream_pipeline_class": type(original_pipeline).__name__,
            "selector_insertion": "after embed_ln, before autoregressive PLM loop",
            "feedback_selection": False,
            "disabled_trace": disabled_extension_trace,
            "identity_trace": identity_extension_trace,
        },
        "identity_selection_output": {
            "embeddings_shape": list(selection.embeddings.shape),
            "attention_mask_shape": None
            if selection.attention_mask is None
            else list(selection.attention_mask.shape),
            "selected_indices": selection.selected_indices.tolist(),
            "scores": None,
            "original_length": selection.original_length,
            "selected_length": selection.selected_length,
            "metadata": selection.metadata,
            "dtype": str(selection.embeddings.dtype),
            "device": str(selection.embeddings.device),
            "requires_grad": bool(selection.embeddings.requires_grad),
            "embeddings_same_object": identity_extension_trace[
                "embeddings_same_object"
            ],
            "attention_mask_same_object": identity_extension_trace[
                "attention_mask_same_object"
            ],
        },
        "paths": path_reports,
        "comparisons": comparisons,
        "execution": {
            "training": False,
            "backward": False,
            "optimizer": False,
            "scheduler": False,
            "lora": False,
            "adaptation": False,
            "multimodal": False,
            "token_pruning": False,
            "recent_k": False,
            "learned_scorer": False,
            "cache_optimization": False,
            "feedback_layernorm_added": False,
            "final_unused_feedback_removed": False,
        },
    }

    if output_path is not None:
        if output_path.exists():
            raise RuntimeError(f"Output already exists: {output_path}")
        if not output_path.parent.is_dir():
            raise RuntimeError(f"Output directory is missing: {output_path.parent}")
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not success:
        raise RuntimeError(
            "Phase 3A equivalence failed: "
            + json.dumps(comparisons, sort_keys=True)
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_equivalence(Path(args.output))
    print(json.dumps({
        "success": report["success"],
        "sample": report["sample"],
        "comparisons": report["comparisons"],
        "identity_selection_output": report["identity_selection_output"],
        "path_contracts": {
            name: {
                "shape": value["output"]["shape"],
                "sequence_lengths": value["trace"]["sequence_lengths"],
                "plm_forward_count": value["trace"]["plm_forward_count"],
            }
            for name, value in report["paths"].items()
        },
        "output": args.output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
