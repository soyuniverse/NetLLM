#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


ARTIFACT = Path("/workspace/NetLLM-artifacts/plms/gpt2/base")
VP_ROOT = Path("/workspace/NetLLM-source/viewport_prediction")


def main():
    os.environ["HF_HOME"] = "/workspace/NetLLM-artifacts/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/NetLLM-artifacts/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    sys.path.insert(0, str(VP_ROOT))

    import torch
    from peft import PeftModel
    from transformers import GPT2Config, GPT2Tokenizer
    from models.gpt2 import GPT2NetworkingHeadModel
    from models.networking_head import NetworkingHead
    from utils.plms_utils import add_special_tokens, get_model_class

    torch.set_grad_enabled(False)
    mapping = get_model_class("gpt2")
    if mapping.config is not GPT2Config:
        raise RuntimeError(f"Unexpected config class: {mapping.config}")
    if mapping.tokenizer is not GPT2Tokenizer:
        raise RuntimeError(f"Unexpected tokenizer class: {mapping.tokenizer}")
    if mapping.model is not GPT2NetworkingHeadModel:
        raise RuntimeError(f"Unexpected model class: {mapping.model}")

    config = mapping.config.from_pretrained(str(ARTIFACT), local_files_only=True)
    if (config.n_embd, config.n_layer, config.n_head) != (1024, 24, 16):
        raise RuntimeError(
            "Artifact architecture does not match NetLLM GPT-2 base expectations: "
            f"{config.n_embd=}, {config.n_layer=}, {config.n_head=}"
        )

    tokenizer = mapping.tokenizer.from_pretrained(str(ARTIFACT), local_files_only=True)
    model = mapping.model.from_pretrained(
        str(ARTIFACT),
        config=config,
        local_files_only=True,
    )

    parameter_count_pre_pad = sum(parameter.numel() for parameter in model.parameters())
    networking_head_before = model.get_networking_head()
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=["<pad>"])
    parameter_count_post_pad = sum(parameter.numel() for parameter in model.parameters())

    networking_head = NetworkingHead(
        input_dim=model.hidden_size,
        output_dim=3,
        fut_window=20,
    )
    model.set_networking_head(networking_head)
    linear = model.get_networking_head().networking_head[0]
    parameter_count_with_head = sum(parameter.numel() for parameter in model.parameters())

    first_parameter = next(model.parameters())
    report = {
        "artifact_path": str(ARTIFACT),
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            "local_files_only": True,
        },
        "model_class": type(model).__name__,
        "base_class": model.__class__.__mro__[1].__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "config_class": type(config).__name__,
        "hidden_size": model.hidden_size,
        "layers": config.n_layer,
        "attention_heads": config.n_head,
        "vocabulary_size_config": config.vocab_size,
        "tokenizer_length_after_pad": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "parameter_count_pre_pad": parameter_count_pre_pad,
        "parameter_count_post_pad": parameter_count_post_pad,
        "parameter_count_with_networking_head": parameter_count_with_head,
        "dtype": str(first_parameter.dtype),
        "device": str(first_parameter.device),
        "networking_head_before_setup": None if networking_head_before is None else type(networking_head_before).__name__,
        "networking_head_after_setup": type(model.get_networking_head()).__name__,
        "networking_head_linear_in_features": linear.in_features,
        "networking_head_linear_out_features": linear.out_features,
        "networking_head_activation": type(model.get_networking_head().networking_head[1]).__name__,
        "lora_applied": isinstance(model, PeftModel) or hasattr(model, "peft_config"),
        "forward_executed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
