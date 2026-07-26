#!/usr/bin/env python3
"""One-sample, non-multimodal VP smoke using the exact old NetLLM path."""

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
DATASET_ROOT = Path(
    "/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022"
)
RESULT_DIR = Path(
    os.environ.get(
        "NETLLM_VP_SMOKE_RESULT_DIR",
        "/root/NetLLM/experiments/vp/llama_vp_technical_smoke",
    )
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("technical_smoke_result.json", "run_status.txt"):
        if (RESULT_DIR / name).exists():
            raise RuntimeError(
                "refusing to overwrite existing runtime: {}".format(
                    RESULT_DIR / name
                )
            )

    started = time.perf_counter()
    load_started = started
    stage = "preflight"
    result = None
    module_path = CHECKPOINT / "modules_except_plm.bin"
    adapter_path = CHECKPOINT / "adapter_model.bin"
    module_sha_before = sha256_file(module_path)
    adapter_sha_before = sha256_file(adapter_path)

    try:
        assert torch.cuda.is_available(), "CUDA available=False"
        assert DATASET_ROOT.is_dir(), "dataset root missing"
        sys.path.insert(0, str(ERA_VP))

        from config import cfg
        from dataset.load_dataset import create_dataset
        from models.low_rank import peft_model
        from models.old.llama import LlamaTaskHeadModel2
        from models.old.networking_head import SimpleLinearTaskHead
        from models.old.pipeline import EmbeddingForViewportPrediction
        from peft.utils.save_and_load import get_peft_model_state_dict
        from transformers import LlamaConfig, LlamaTokenizer
        from utils.normalize import normalize_data

        stage = "cpu_config_tokenizer"
        config = LlamaConfig.from_pretrained(str(BASE), local_files_only=True)
        tokenizer = LlamaTokenizer.from_pretrained(
            str(BASE), local_files_only=True, use_fast=False
        )

        stage = "dataset"
        cfg.dataset["Jin2022"] = str(DATASET_ROOT)
        test_dataset = create_dataset(
            "Jin2022",
            his_window=10,
            fut_window=20,
            trim_head=30,
            trim_tail=60,
            frequency=5,
            step=15,
            include=["test"],
        )[0]
        history_np, future_np, sample_info = test_dataset[0]
        history = torch.from_numpy(history_np).unsqueeze(0)
        future = torch.from_numpy(future_np).unsqueeze(0)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        stage = "base_model"
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
        stage = "task_head_pipeline"
        plm.set_task_head(
            SimpleLinearTaskHead(4096, 3, 20).to("cuda:0")
        )
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

        stage = "adapter_load"
        adapter_load_result = model.plm.load_adapter(
            str(CHECKPOINT), adapter_name="default"
        )
        model.plm.set_adapter("default")

        stage = "adapter_value_validation"
        source_adapter = torch.load(
            adapter_path, map_location="cpu", weights_only=True
        )
        loaded_adapter = get_peft_model_state_dict(
            model.plm, adapter_name="default"
        )
        source_adapter_keys = set(source_adapter)
        loaded_adapter_keys = set(loaded_adapter)
        adapter_missing = sorted(source_adapter_keys - loaded_adapter_keys)
        adapter_unexpected = sorted(loaded_adapter_keys - source_adapter_keys)
        adapter_value_mismatches = []
        for key in sorted(source_adapter_keys & loaded_adapter_keys):
            source_tensor = source_adapter[key]
            loaded_tensor = loaded_adapter[key].detach().cpu()
            source_cast = source_tensor.to(dtype=loaded_tensor.dtype)
            if not torch.equal(source_cast, loaded_tensor):
                adapter_value_mismatches.append(key)
        assert not adapter_missing, "adapter missing keys: {}".format(
            adapter_missing
        )
        assert not adapter_unexpected, "adapter unexpected keys: {}".format(
            adapter_unexpected
        )
        assert not adapter_value_mismatches, (
            "adapter value mismatches: {}".format(adapter_value_mismatches)
        )

        stage = "strict_non_plm"
        non_plm_source = torch.load(
            module_path, map_location="cpu", weights_only=True
        )
        incompatible = model.embedding_model.modules_except_plm.load_state_dict(
            non_plm_source, strict=True
        )
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        assert not missing_keys
        assert not unexpected_keys

        model.half()
        model.eval()
        load_latency = time.perf_counter() - load_started

        sequence_lengths = []
        cache_supplied = []

        def forward_pre_hook(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            sequence_lengths.append(
                int(embeds.shape[1]) if embeds is not None else None
            )
            cache_supplied.append(kwargs.get("past_key_values") is not None)

        # The PEFT wrapper is the callable invoked by the old pipeline.
        # Hooking the pre-PEFT object does not observe these calls.
        hook = model.plm.register_forward_pre_hook(
            forward_pre_hook, with_kwargs=True
        )

        stage = "vp_inference"
        history = normalize_data(history, "Jin2022").to(
            device="cuda:0", dtype=torch.float16
        )
        future = normalize_data(future, "Jin2022").to(
            device="cuda:0", dtype=torch.float16
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            prediction, ground_truth = model.inference(
                history, future, sample_info
            )
        torch.cuda.synchronize()
        inference_latency = time.perf_counter() - inference_started
        hook.remove()

        stage = "post_validation"
        finite = bool(torch.isfinite(prediction).all().item())
        assert list(prediction.shape) == [1, 20, 3], (
            "prediction shape {} != [1, 20, 3]".format(
                list(prediction.shape)
            )
        )
        assert finite, "prediction contains non-finite values"
        assert sequence_lengths == list(range(10, 30)), (
            "sequence lengths {} != 10..29".format(sequence_lengths)
        )
        assert not any(cache_supplied), (
            "past_key_values was supplied: {}".format(cache_supplied)
        )
        assert sha256_file(module_path) == module_sha_before
        assert sha256_file(adapter_path) == adapter_sha_before

        active_adapter = getattr(model.plm, "active_adapter", None)
        if callable(active_adapter):
            active_adapter = active_adapter()
        non_plm_state = model.embedding_model.modules_except_plm.state_dict()
        random_initialization_remaining = False

        result = {
            "success": True,
            "technical_smoke_valid": True,
            "quality_metric_valid": False,
            "reproduction_claim_valid": False,
            "quality_metric_reason": (
                "single-sample technical smoke; checkpoint selection "
                "epoch/criterion and exact training base revision remain absent"
            ),
            "using_multimodal": False,
            "using_multimodal_evidence": "proven_non_multimodal",
            "dataset_root": str(DATASET_ROOT),
            "dataset_split": "test",
            "dataset_index": 0,
            "sample_identity": {
                "video": int(sample_info[0]),
                "user": int(sample_info[1]),
                "timestep": int(sample_info[2]),
            },
            "history_shape": list(history.shape),
            "future_shape": list(future.shape),
            "prediction_shape": list(prediction.shape),
            "output_finite": finite,
            "prediction_dtype": str(prediction.dtype),
            "prediction_device": str(prediction.device),
            "model_dtype": str(next(base_model.parameters()).dtype),
            "model_device": str(next(base_model.parameters()).device),
            "lora_active": active_adapter == "default",
            "active_adapter": active_adapter,
            "adapter_source_key_count": len(source_adapter),
            "adapter_loaded_key_count": len(loaded_adapter),
            "adapter_missing_keys": adapter_missing,
            "adapter_unexpected_keys": adapter_unexpected,
            "adapter_value_mismatch_count_after_dtype_cast": len(
                adapter_value_mismatches
            ),
            "adapter_loader_missing_key_count": len(
                adapter_load_result.missing_keys
            ),
            "adapter_loader_unexpected_keys": list(
                adapter_load_result.unexpected_keys
            ),
            "networking_task_head_restored": (
                "4.task_head.0.weight" in non_plm_state
                and "4.task_head.0.bias" in non_plm_state
            ),
            "viewport_projection_restored": (
                "0.weight" in non_plm_state and "0.bias" in non_plm_state
            ),
            "non_plm_missing_keys": missing_keys,
            "non_plm_unexpected_keys": unexpected_keys,
            "random_initialization_remaining": random_initialization_remaining,
            "model_load_latency_seconds": load_latency,
            "inference_latency_seconds": inference_latency,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "full_sequence_forward_count": len(sequence_lengths),
            "sequence_lengths": sequence_lengths,
            "cache_reuse": any(cache_supplied),
            "module_checkpoint_sha256_before": module_sha_before,
            "module_checkpoint_sha256_after": sha256_file(module_path),
            "adapter_checkpoint_sha256_before": adapter_sha_before,
            "adapter_checkpoint_sha256_after": sha256_file(adapter_path),
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "training_performed": False,
        }
    except Exception as error:
        result = {
            "success": False,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved()
                if torch.cuda.is_available()
                else 0
            ),
            "module_checkpoint_sha256_before": module_sha_before,
            "module_checkpoint_sha256_after": sha256_file(module_path),
            "adapter_checkpoint_sha256_before": adapter_sha_before,
            "adapter_checkpoint_sha256_after": sha256_file(adapter_path),
            "observed_prediction_shape": (
                list(prediction.shape)
                if "prediction" in locals()
                else None
            ),
            "observed_output_finite": (
                bool(torch.isfinite(prediction).all().item())
                if "prediction" in locals()
                else None
            ),
            "observed_sequence_lengths": (
                sequence_lengths
                if "sequence_lengths" in locals()
                else None
            ),
            "observed_cache_supplied": (
                cache_supplied if "cache_supplied" in locals() else None
            ),
        }

    with open(
        RESULT_DIR / "technical_smoke_result.json", "w", encoding="utf-8"
    ) as stream:
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
