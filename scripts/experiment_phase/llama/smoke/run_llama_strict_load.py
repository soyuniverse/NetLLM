#!/usr/bin/env python3
"""Local-only strict load of the published checkpoint-era Llama VP model."""

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import psutil
import torch


BASE = Path("/root/NetLLM-assets/llama/base")
CHECKPOINT = Path("/root/NetLLM-assets/checkpoints/try_llama2_7b")
ERA_VP = Path("/root/NetLLM-source-checkpoint-era/viewport_prediction")
RESULT_DIR = Path("/root/NetLLM/experiments/vp/llama_strict_load")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail_result(stage, error, started, extra=None):
    result = {
        "success": False,
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
    }
    if extra:
        result.update(extra)
    return result


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for output_name in ("strict_load_result.json", "run_status.txt"):
        if (RESULT_DIR / output_name).exists():
            raise RuntimeError(
                "refusing to overwrite existing runtime: {}".format(
                    RESULT_DIR / output_name
                )
            )
    started = time.perf_counter()
    stage = "preflight"
    result = None

    module_path = CHECKPOINT / "modules_except_plm.bin"
    adapter_path = CHECKPOINT / "adapter_model.bin"
    module_sha_before = sha256_file(module_path)
    adapter_sha_before = sha256_file(adapter_path)

    try:
        assert torch.cuda.is_available(), "CUDA available=False"
        for required in (
            BASE / "config.json",
            BASE / "tokenizer.model",
            CHECKPOINT / "adapter_config.json",
            adapter_path,
            module_path,
        ):
            assert required.is_file(), "missing required file: {}".format(required)

        sys.path.insert(0, str(ERA_VP))
        from models.low_rank import peft_model
        from models.old.llama import LlamaTaskHeadModel2
        from models.old.networking_head import SimpleLinearTaskHead
        from models.old.pipeline import EmbeddingForViewportPrediction
        from transformers import LlamaConfig, LlamaTokenizer

        stage = "cpu_config"
        config = LlamaConfig.from_pretrained(str(BASE), local_files_only=True)
        stage = "cpu_tokenizer"
        tokenizer = LlamaTokenizer.from_pretrained(
            str(BASE), local_files_only=True, use_fast=False
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        stage = "base_model"
        load_started = time.perf_counter()
        base_model = LlamaTaskHeadModel2.from_pretrained(
            str(BASE),
            config=config,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )

        stage = "lora_structure"
        plm = peft_model(base_model, "llama", 32)

        stage = "task_head_and_pipeline"
        task_head = SimpleLinearTaskHead(
            input_dim=plm.hidden_size, output_dim=3, fut_window=20
        ).to("cuda:0")
        plm.set_task_head(task_head)
        model = EmbeddingForViewportPrediction(
            plm,
            fut_window=20,
            device="cuda:0",
            embed_size=4096,
            frequency=5,
            using_teaching_forcing=False,
            using_multimodal=False,
            dataset="Jin2022",
        )

        stage = "lora_adapter"
        model.plm.load_adapter(str(CHECKPOINT), adapter_name="default")
        model.plm.set_adapter("default")

        stage = "strict_non_plm"
        state = torch.load(module_path, map_location="cpu", weights_only=True)
        incompatible = model.embedding_model.modules_except_plm.load_state_dict(
            state, strict=True
        )
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        assert not missing_keys, "missing keys: {}".format(missing_keys)
        assert not unexpected_keys, "unexpected keys: {}".format(unexpected_keys)

        # The checkpoint was stored in FP32. Convert the complete inference
        # pipeline after strict restoration to match the FP16 base model.
        model.half()
        model.eval()
        load_latency = time.perf_counter() - load_started

        stage = "post_load_validation"
        adapter_sha_after = sha256_file(adapter_path)
        module_sha_after = sha256_file(module_path)
        assert adapter_sha_after == adapter_sha_before
        assert module_sha_after == module_sha_before

        lora_parameters = [
            (name, parameter)
            for name, parameter in model.plm.named_parameters()
            if "lora_" in name
        ]
        active_adapter = getattr(model.plm, "active_adapter", None)
        if callable(active_adapter):
            active_adapter = active_adapter()

        non_plm_state = model.embedding_model.modules_except_plm.state_dict()
        result = {
            "success": True,
            "load_mode": "checkpoint-era native model plus strict non-PLM state load",
            "offline": True,
            "local_files_only": True,
            "model_class": type(base_model).__name__,
            "peft_model_class": type(model.plm).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "base_path": str(BASE),
            "base_revision": "01c7f73d771dfac7d292323805ebc428287df4f9",
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_era_commit": "ee4d8726898610e4ae7df08bdd26728cafb4701f",
            "torch_dtype": str(next(base_model.parameters()).dtype),
            "device_map": {"entire_model": "cuda:0"},
            "lora_active": bool(lora_parameters) and active_adapter == "default",
            "active_adapter": active_adapter,
            "lora_tensor_count": len(lora_parameters),
            "lora_parameter_count": sum(p.numel() for _, p in lora_parameters),
            "restored_non_plm_parameter_count": sum(
                tensor.numel() for tensor in non_plm_state.values()
            ),
            "networking_task_head_restored": (
                "4.task_head.0.weight" in non_plm_state
                and "4.task_head.0.bias" in non_plm_state
            ),
            "viewport_embedding_restored": (
                "0.weight" in non_plm_state and "0.bias" in non_plm_state
            ),
            "multimodal_projection_restored": (
                "1.weight" in non_plm_state and "1.bias" in non_plm_state
            ),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "random_initialization_remaining": False,
            "unused_tokenizer_pad_row_added": False,
            "module_checkpoint_sha256_before": module_sha_before,
            "module_checkpoint_sha256_after": module_sha_after,
            "adapter_checkpoint_sha256_before": adapter_sha_before,
            "adapter_checkpoint_sha256_after": adapter_sha_after,
            "model_load_latency_seconds": load_latency,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "system_ram_total_bytes": psutil.virtual_memory().total,
            "system_ram_available_bytes": psutil.virtual_memory().available,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
    except Exception as error:
        result = fail_result(
            stage,
            error,
            started,
            {
                "module_checkpoint_sha256_before": module_sha_before,
                "module_checkpoint_sha256_after": sha256_file(module_path),
                "adapter_checkpoint_sha256_before": adapter_sha_before,
                "adapter_checkpoint_sha256_after": sha256_file(adapter_path),
            },
        )

    with open(RESULT_DIR / "strict_load_result.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with open(RESULT_DIR / "run_status.txt", "w", encoding="utf-8") as stream:
        stream.write("success={}\n".format(str(result["success"]).lower()))
        if not result["success"]:
            stream.write("failed_stage={}\n".format(result["failed_stage"]))

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
