#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llama_selector_benchmark import (
    PROJECT, load_runtime, prepare_sample, tensor_sha256,
)

sys.path.insert(0, str(PROJECT / "src"))
from netllm_litevlm.selectors import IdentitySelector
from netllm_litevlm.vp.llama_old_selectable_pipeline import (
    LlamaOldSelectablePipeline,
)


def original_trace(model, history, future, info):
    lengths = []
    def hook(module, args, kwargs):
        lengths.append(int(kwargs["inputs_embeds"].shape[1]))
    handle = model.plm.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.inference_mode():
        output, _ = model.inference(history, future, info)
    handle.remove()
    return output, lengths


def main():
    root = PROJECT / "experiments/vp/llama_selector_equivalence"
    for name in ("identity_equivalence.json", "run_status.txt"):
        if (root / name).exists():
            raise RuntimeError("refusing overwrite: {}".format(root / name))
    model, dataset = load_runtime()
    history, future_raw, _, info = prepare_sample(dataset, 0)
    original, original_lengths = original_trace(
        model, history, future_raw, info
    )
    disabled_path = LlamaOldSelectablePipeline(model, None).half().eval()
    identity_path = LlamaOldSelectablePipeline(
        model, IdentitySelector()
    ).half().eval()
    with torch.inference_mode():
        disabled, _ = disabled_path.inference(history, future_raw, info)
        identity, _ = identity_path.inference(history, future_raw, info)
    pairs = {
        "original_vs_disabled": float((original - disabled).abs().max().item()),
        "original_vs_identity": float((original - identity).abs().max().item()),
        "disabled_vs_identity": float((disabled - identity).abs().max().item()),
    }
    result = {
        "success": max(pairs.values()) <= 1e-6,
        "tolerance": {"atol": 1e-6, "rtol": 0},
        "sample": {"dataset": "Jin2022", "index": 0,
                   "video": int(info[0]), "user": int(info[1]),
                   "timestep": int(info[2])},
        "output_shape": list(original.shape),
        "finite": bool(torch.isfinite(original).all().item()),
        "dtype": str(original.dtype), "device": str(original.device),
        "max_absolute_differences": pairs,
        "exact_equal": {
            "original_vs_disabled": bool(torch.equal(original, disabled)),
            "original_vs_identity": bool(torch.equal(original, identity)),
            "disabled_vs_identity": bool(torch.equal(disabled, identity)),
        },
        "prediction_sha256": {
            "original": tensor_sha256(original),
            "disabled": tensor_sha256(disabled),
            "identity": tensor_sha256(identity),
        },
        "original_sequence_lengths": original_lengths,
        "disabled_trace": disabled_path.last_trace,
        "identity_trace": identity_path.last_trace,
        "identity_selection": {
            "original_length": identity_path.last_selection_output.original_length,
            "selected_length": identity_path.last_selection_output.selected_length,
            "selected_indices": identity_path.last_selection_output.selected_indices.tolist(),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "identity_equivalence.json").open("w") as f:
        json.dump(result, f, indent=2); f.write("\n")
    with (root / "run_status.txt").open("w") as f:
        f.write("success={}\n".format(str(result["success"]).lower()))
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
