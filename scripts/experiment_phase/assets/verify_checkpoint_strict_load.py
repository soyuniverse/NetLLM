#!/usr/bin/env python3
"""Strict-load re-verification for the recovered try_llama2_7b checkpoint.

Replicates the exact procedure
scripts/experiment_phase/llama/smoke/run_llama_vp_technical_smoke.py used
the first time this checkpoint was strict-loaded (no
docs/RUNBOOK_ASSETS_ARRIVAL.md exists in this repo to follow instead -- see
docs/experiment_phase/assets/ASSET_RECOVERY_VERIFICATION.md for that
finding): adapter missing/unexpected/value-mismatch keys via
get_peft_model_state_dict, and non-PLM module missing/unexpected keys via
a strict state_dict load, both checked independently of
load_checkpoint_era_model's own internal assertion so this script's output
is a standalone, inspectable record.
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

CHECKPOINT = Path("/root/NetLLM-assets/checkpoints/try_llama2_7b")
OUTPUT = PROJECT_ROOT / "experiments/vp/asset_recovery"


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(str(tmp), str(path))


def main() -> int:
    from netllm_litevlm.vp.checkpoint_era_runtime import UPSTREAM_VP_ROOT

    sys.path.insert(0, str(UPSTREAM_VP_ROOT))
    from peft.utils.save_and_load import get_peft_model_state_dict

    from netllm_litevlm.vp.checkpoint_era_runtime import load_checkpoint_era_model

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "checkpoint_strict_load.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {result_path}")

    started = time.perf_counter()
    model, checkpoint_loaded = load_checkpoint_era_model(
        device="cuda:0", dtype=torch.float16, rank=32, checkpoint_path=CHECKPOINT, seed=0,
    )
    load_seconds = time.perf_counter() - started
    if not checkpoint_loaded:
        raise RuntimeError("load_checkpoint_era_model did not report a loaded checkpoint")

    # Adapter: compare the raw saved state dict against what's actually
    # active in the model after load_adapter, key-for-key and value-for-value.
    source_adapter = torch.load(
        CHECKPOINT / "adapter_model.bin", map_location="cpu", weights_only=True
    )
    loaded_adapter = get_peft_model_state_dict(model.plm, adapter_name="default")
    source_keys = set(source_adapter)
    loaded_keys = set(loaded_adapter)
    adapter_missing = sorted(source_keys - loaded_keys)
    adapter_unexpected = sorted(loaded_keys - source_keys)
    adapter_value_mismatches = []
    for key in sorted(source_keys & loaded_keys):
        source_tensor = source_adapter[key]
        loaded_tensor = loaded_adapter[key].detach().cpu()
        if not torch.equal(source_tensor.to(dtype=loaded_tensor.dtype), loaded_tensor):
            adapter_value_mismatches.append(key)

    # Non-PLM modules (conv1d1, linear_layer, embed_ln, task_head): strict
    # state_dict load against the model's own live modules_except_plm list.
    non_plm_source = torch.load(
        CHECKPOINT / "modules_except_plm.bin", map_location="cpu", weights_only=True
    )
    incompatible = model.embedding_model.modules_except_plm.load_state_dict(
        non_plm_source, strict=True
    )

    result = {
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_loaded": checkpoint_loaded,
        "load_seconds": load_seconds,
        "adapter_missing_keys": adapter_missing,
        "adapter_unexpected_keys": adapter_unexpected,
        "adapter_value_mismatches": adapter_value_mismatches,
        "adapter_missing_count": len(adapter_missing),
        "adapter_unexpected_count": len(adapter_unexpected),
        "adapter_value_mismatch_count": len(adapter_value_mismatches),
        "non_plm_missing_keys": list(incompatible.missing_keys),
        "non_plm_unexpected_keys": list(incompatible.unexpected_keys),
        "non_plm_missing_count": len(incompatible.missing_keys),
        "non_plm_unexpected_count": len(incompatible.unexpected_keys),
        "strict_load_pass": (
            len(adapter_missing) == 0
            and len(adapter_unexpected) == 0
            and len(adapter_value_mismatches) == 0
            and len(incompatible.missing_keys) == 0
            and len(incompatible.unexpected_keys) == 0
        ),
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2))
    return 0 if result["strict_load_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
