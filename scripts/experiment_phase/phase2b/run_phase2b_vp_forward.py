#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path("/workspace/NetLLM")
SOURCE_ROOT = Path("/workspace/NetLLM-source")
VP_ROOT = SOURCE_ROOT / "viewport_prediction"
ARTIFACT_ROOT = Path("/workspace/NetLLM-artifacts/plms")
ARTIFACT_PATH = ARTIFACT_ROOT / "gpt2/base"
SELECTED_STEPS = {0, 1, 19}


def tensor_stats(tensor):
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor, got {type(tensor)!r}")
    detached = tensor.detach()
    if detached.numel() == 0:
        minimum = maximum = mean = None
        finite = True
    else:
        numeric = detached.to(dtype=torch.float64)
        minimum = float(numeric.min().item())
        maximum = float(numeric.max().item())
        mean = float(numeric.mean().item())
        finite = bool(torch.isfinite(numeric).all().item())
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "finite": finite,
        "requires_grad": bool(tensor.requires_grad),
    }


class RuntimeTrace:
    def __init__(self):
        self.tensors = {}
        self.steps = {str(step): {} for step in sorted(SELECTED_STEPS)}
        self.conv_call = -1
        self.embed_call = -1
        self.plm_step = -1
        self.sequence_lengths = []
        self.cache_input_present = []
        self.cache_output_present = []
        self.cache_layer_counts = []

    def add_tensor(self, name, tensor):
        self.tensors[name] = tensor_stats(tensor)

    def add_step_tensor(self, step, name, tensor):
        if step in SELECTED_STEPS:
            self.steps[str(step)][name] = tensor_stats(tensor)

    def conv_layer_hook(self, module, inputs, output):
        self.conv_call += 1
        call = self.conv_call
        if call == 0:
            self.add_tensor("single_timestep_input", inputs[0])
            self.add_tensor("conv1d_output", output)
        if call >= 10:
            step = call - 10
            self.add_step_tensor(step, "autoregressive_feedback_coordinate", inputs[0])
            self.add_step_tensor(step, "autoregressive_feedback_conv1d_output", output)

    def conv_sequence_hook(self, module, inputs, output):
        call = self.conv_call
        if call == 0:
            self.add_tensor("flattened_viewport_representation", output)
        if call >= 10:
            step = call - 10
            self.add_step_tensor(step, "autoregressive_feedback_flattened", output)

    def embed_vp_hook(self, module, inputs, output):
        self.embed_call += 1
        call = self.embed_call
        if call == 0:
            self.add_tensor("projected_temporal_embedding", output.unsqueeze(1))
        if call >= 10:
            step = call - 10
            self.add_step_tensor(step, "autoregressive_feedback_embedding", output.unsqueeze(1))

    def layer_norm_hook(self, module, inputs, output):
        self.add_tensor("concatenated_temporal_sequence", inputs[0])
        self.add_tensor("layernorm_output", output)

    def plm_pre_hook(self, module, args, kwargs):
        self.plm_step += 1
        step = self.plm_step
        embeds = kwargs["inputs_embeds"]
        mask = kwargs["attention_mask"]
        self.sequence_lengths.append(int(embeds.shape[1]))
        past_present = kwargs.get("past_key_values") is not None
        self.cache_input_present.append(past_present)
        self.add_step_tensor(step, "gpt2_input_embeddings", embeds)
        self.add_step_tensor(step, "attention_mask", mask)
        if step in SELECTED_STEPS:
            self.steps[str(step)]["input_past_key_values_present"] = past_present
            self.steps[str(step)]["attention_mask_all_ones"] = bool((mask == 1).all().item())

    def transformer_hook(self, module, inputs, output):
        step = self.plm_step
        hidden = output[0]
        self.add_step_tensor(step, "gpt2_final_hidden_states", hidden)
        past = getattr(output, "past_key_values", None)
        if step in SELECTED_STEPS and past is not None:
            self.steps[str(step)]["returned_cache_layers"] = len(past)
            self.steps[str(step)]["returned_cache_first_key"] = tensor_stats(past[0][0])
            self.steps[str(step)]["returned_cache_first_value"] = tensor_stats(past[0][1])

    def networking_head_pre_hook(self, module, inputs):
        self.add_step_tensor(self.plm_step, "networking_head_full_input", inputs[0])

    def networking_linear_hook(self, module, inputs, output):
        step = self.plm_step
        self.add_step_tensor(step, "networking_head_selected_input", inputs[0])
        self.add_step_tensor(step, "networking_head_linear_output", output)

    def networking_head_hook(self, module, inputs, output):
        self.add_step_tensor(self.plm_step, "networking_head_output", output)

    def plm_hook(self, module, inputs, output):
        past = getattr(output, "past_key_values", None)
        self.cache_output_present.append(past is not None)
        self.cache_layer_counts.append(0 if past is None else len(past))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise RuntimeError(f"Output already exists: {output_path}")

    required_env = {
        "HF_HOME": "/workspace/NetLLM-artifacts/hf_cache",
        "TRANSFORMERS_CACHE": "/workspace/NetLLM-artifacts/hf_cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for key, expected in required_env.items():
        actual = os.environ.get(key)
        if actual != expected:
            raise RuntimeError(f"Required environment mismatch: {key}={actual!r}, expected {expected!r}")

    os.chdir(VP_ROOT)
    sys.path.insert(0, str(VP_ROOT))

    import torch
    from torch.utils.data import DataLoader

    from config import cfg
    from dataset.load_dataset import create_dataset
    from models.networking_head import NetworkingHead
    from models.pipeline import Pipeline
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

    original_plms_dir = cfg.plms_dir
    cfg.plms_dir = str(ARTIFACT_ROOT)
    resolved_model_path = Path(cfg.plms_dir) / "gpt2/base"
    if resolved_model_path.resolve() != ARTIFACT_PATH.resolve():
        raise RuntimeError(f"Runtime model path mismatch: {resolved_model_path}")

    model_load_start = time.perf_counter()
    plm, tokenizer, model_config = load_plm("gpt2", str(resolved_model_path), plm_size="base")
    model_load_seconds = time.perf_counter() - model_load_start
    loader_class = type(plm).__name__
    if loader_class != "GPT2NetworkingHeadModel":
        raise RuntimeError(f"Unexpected model class: {loader_class}")
    if model_config.n_embd != 1024:
        raise RuntimeError(f"Unexpected hidden size: {model_config.n_embd}")

    model_to_device_start = time.perf_counter()
    plm = plm.to(device)
    torch.cuda.synchronize(device)
    model_to_device_seconds = time.perf_counter() - model_to_device_start

    pipeline_start = time.perf_counter()
    networking_head = NetworkingHead(input_dim=plm.hidden_size, output_dim=3, fut_window=20).to(device)
    plm.set_networking_head(networking_head)
    pipeline = Pipeline(
        plm=plm,
        loss_func=None,
        fut_window=20,
        device=str(device),
        embed_size=1024,
        frequency=5,
        using_multimodal=False,
        dataset="Jin2022",
    )
    plm.eval()
    pipeline.eval()
    torch.cuda.synchronize(device)
    pipeline_construction_seconds = time.perf_counter() - pipeline_start

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
    dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    history, future, video_user_info = next(iter(dataloader))
    video, user, timestep = [int(item.item()) for item in video_user_info]
    expected_metadata = tuple(dataset_test.trace_indices[0])
    if (video, user, timestep) != expected_metadata:
        raise RuntimeError(
            f"Sample metadata mismatch: batch={(video, user, timestep)}, dataset={expected_metadata}"
        )
    if history.shape != (1, 10, 3) or future.shape != (1, 20, 3):
        raise RuntimeError(f"Unexpected sample shapes: history={history.shape}, future={future.shape}")

    csv_relative = Path(
        f"viewport_prediction/data/viewports/Jin2022/video{video}/5Hz/simple_5Hz_user{user}.csv"
    )
    tracked_check = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "ls-files", "--error-unmatch", str(csv_relative)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if tracked_check.returncode != 0:
        raise RuntimeError(f"Dataset CSV is not Git tracked: {csv_relative}")

    trace = RuntimeTrace()
    trace.add_tensor("raw_history", history)
    trace.add_tensor("raw_future_ground_truth", future)
    normalized_history = normalize_data(history, "Jin2022")
    trace.add_tensor("normalized_history", normalized_history)

    handles = [
        pipeline.conv1d[0].register_forward_hook(trace.conv_layer_hook),
        pipeline.conv1d.register_forward_hook(trace.conv_sequence_hook),
        pipeline.embed_vp.register_forward_hook(trace.embed_vp_hook),
        pipeline.embed_ln.register_forward_hook(trace.layer_norm_hook),
        plm.register_forward_pre_hook(trace.plm_pre_hook, with_kwargs=True),
        plm.transformer.register_forward_hook(trace.transformer_hook),
        plm.networking_head.register_forward_pre_hook(trace.networking_head_pre_hook),
        plm.networking_head.networking_head[0].register_forward_hook(trace.networking_linear_hook),
        plm.networking_head.register_forward_hook(trace.networking_head_hook),
        plm.register_forward_hook(trace.plm_hook),
    ]

    normalized_history = normalized_history.to(device)
    future_device = future.to(device)
    trace.add_tensor("normalized_history_on_device", normalized_history)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before_forward = torch.cuda.memory_allocated(device)
    reserved_before_forward = torch.cuda.memory_reserved(device)
    torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    try:
        with torch.inference_mode():
            prediction, ground_truth = pipeline.inference(
                normalized_history,
                future_device,
                video_user_info,
            )
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    forward_elapsed_seconds = time.perf_counter() - forward_start

    trace.add_tensor("final_prediction", prediction)
    trace.add_tensor("returned_ground_truth", ground_truth)
    if prediction.shape != (1, 20, 3):
        raise RuntimeError(f"Unexpected prediction shape: {prediction.shape}")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Prediction contains non-finite values")
    if prediction.min().item() < -1.0 or prediction.max().item() > 1.0:
        raise RuntimeError("Prediction is outside the NetworkingHead Tanh range")
    if trace.plm_step + 1 != 20:
        raise RuntimeError(f"Unexpected autoregressive PLM call count: {trace.plm_step + 1}")
    if trace.sequence_lengths != list(range(10, 30)):
        raise RuntimeError(f"Unexpected sequence lengths: {trace.sequence_lengths}")

    first_parameter = next(plm.parameters())
    report = {
        "phase": "2B",
        "success": True,
        "execution": {
            "training": False,
            "backward": False,
            "optimizer": False,
            "scheduler": False,
            "adaptation": False,
            "lora": False,
            "multimodal": False,
            "checkpoint_written": False,
            "result_csv_written": False,
            "pipeline_inference_calls": 1,
            "plm_forward_calls": trace.plm_step + 1,
        },
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "offline": True,
        },
        "model": {
            "artifact_path": str(ARTIFACT_PATH),
            "cfg_plms_dir_before_override": original_plms_dir,
            "cfg_plms_dir_runtime_override": cfg.plms_dir,
            "resolved_model_path": str(resolved_model_path),
            "loader": "utils.plms_utils.load_plm",
            "model_class": loader_class,
            "tokenizer_class": type(tokenizer).__name__,
            "hidden_size": plm.hidden_size,
            "layers": model_config.n_layer,
            "attention_heads": model_config.n_head,
            "dtype": str(first_parameter.dtype),
            "device": str(first_parameter.device),
            "eval": not plm.training,
            "config_use_cache": bool(model_config.use_cache),
            "parameter_count": sum(parameter.numel() for parameter in plm.parameters()),
            "networking_head": {
                "class": type(plm.networking_head).__name__,
                "input_dim": plm.networking_head.networking_head[0].in_features,
                "output_dim": plm.networking_head.networking_head[0].out_features,
                "activation": type(plm.networking_head.networking_head[1]).__name__,
                "fut_window": plm.networking_head.fut_window,
            },
        },
        "pipeline": {
            "class": type(pipeline).__name__,
            "eval": not pipeline.training,
            "device": str(device),
            "embed_size": pipeline.embed_size,
            "future_window": pipeline.fut_window_length,
            "frequency": pipeline.frequency,
            "using_multimodal": pipeline.using_multimodal,
            "dataset": pipeline.dataset,
        },
        "sample": {
            "dataset": "Jin2022",
            "split": "test",
            "dataset_length": len(dataset_test),
            "dataset_index": 0,
            "batch_size": 1,
            "video": video,
            "user": user,
            "timestep": timestep,
            "csv_path": str(SOURCE_ROOT / csv_relative),
            "csv_git_tracked": True,
            "history_window": 10,
            "future_window": 20,
            "frequency": 5,
            "sample_step": 15,
            "trim_head": 30,
            "trim_tail": 60,
        },
        "tensor_trace": trace.tensors,
        "autoregressive_steps": trace.steps,
        "autoregressive": {
            "step_count": trace.plm_step + 1,
            "sequence_lengths_used": trace.sequence_lengths,
            "sequence_lengths_after_feedback_append": [length + 1 for length in trace.sequence_lengths],
            "cache_returned_each_step": trace.cache_output_present,
            "cache_layer_counts": trace.cache_layer_counts,
            "past_key_values_passed_each_step": trace.cache_input_present,
            "cache_reused": any(trace.cache_input_present),
        },
        "sanity": {
            "prediction_shape": list(prediction.shape),
            "prediction_all_finite": bool(torch.isfinite(prediction).all().item()),
            "prediction_within_tanh_range": bool(
                prediction.min().item() >= -1.0 and prediction.max().item() <= 1.0
            ),
            "input_device": str(normalized_history.device),
            "output_device": str(prediction.device),
            "requires_grad": bool(prediction.requires_grad),
            "performance_interpretation_allowed": False,
        },
        "runtime": {
            "model_load_seconds_cpu": model_load_seconds,
            "model_to_device_seconds": model_to_device_seconds,
            "pipeline_construction_seconds": pipeline_construction_seconds,
            "forward_elapsed_seconds_with_trace_hooks": forward_elapsed_seconds,
            "gpu_allocated_before_forward_bytes": allocated_before_forward,
            "gpu_reserved_before_forward_bytes": reserved_before_forward,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "source_observations": {
            "history_layernorm_applied_once": True,
            "feedback_embeddings_layernorm_applied": False,
            "full_sequence_recomputed_each_step": True,
            "final_step_feedback_embedding_computed_but_not_consumed": True,
            "cache_returned_but_not_reused": not any(trace.cache_input_present) and all(trace.cache_output_present),
            "batch_size_greater_than_one_supported_by_view_shape": False,
        },
    }

    if not output_path.parent.is_dir():
        raise RuntimeError(f"Runtime output directory is missing: {output_path.parent}")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "success": True,
        "sample": report["sample"],
        "sanity": report["sanity"],
        "runtime": report["runtime"],
        "autoregressive": report["autoregressive"],
        "output": str(output_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
