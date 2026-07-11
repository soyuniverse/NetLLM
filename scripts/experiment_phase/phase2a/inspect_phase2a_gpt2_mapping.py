#!/usr/bin/env python3
import json
from pathlib import Path


SOURCE_ROOT = Path("/workspace/NetLLM-source")
VP_ROOT = SOURCE_ROOT / "viewport_prediction"


def require(path, snippets):
    text = path.read_text()
    for snippet in snippets:
        if snippet not in text:
            raise RuntimeError(f"Expected source marker missing from {path}: {snippet!r}")
    return text


def main():
    config_path = VP_ROOT / "config.py"
    run_path = VP_ROOT / "run_plm.py"
    loader_path = VP_ROOT / "utils/plms_utils.py"
    wrapper_path = VP_ROOT / "models/gpt2.py"
    head_path = VP_ROOT / "models/networking_head.py"
    abr_config_path = SOURCE_ROOT / "adaptive_bitrate_streaming/config.py"
    cjs_run_path = SOURCE_ROOT / "cluster_job_scheduling/run_plm.py"

    require(config_path, [
        "for gpt2, 'base' is 340M",
        "plms_dir =",
    ])
    require(run_path, [
        "load_plm(args.plm_type, os.path.join(cfg.plms_dir, args.plm_type, args.plm_size)",
        "if args.plm_type == 'gpt2':\n        embed_size = 1024",
        "input_dim = plm.hidden_size",
        "NetworkingHead(input_dim=input_dim, output_dim=out_dim",
    ])
    require(loader_path, [
        "'config': GPT2Config",
        "'tokenizer': GPT2Tokenizer",
        "'model': GPT2NetworkingHeadModel",
    ])
    require(wrapper_path, [
        "class GPT2NetworkingHeadModel(GPT2LMHeadModel)",
        "self.hidden_size = config.n_embd",
        "self.networking_head = None",
    ])
    require(head_path, [
        "nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)",
        "nn.Tanh()",
    ])
    require(abr_config_path, [
        "'base': 1024",
        "'base': 24",
    ])
    require(cjs_run_path, [
        "'base': 1024",
        "'base': 24",
    ])

    mapping = {
        "plm_type": "gpt2",
        "plm_size": "base",
        "upstream_size_label": "340M",
        "expected_model_class": "GPT2NetworkingHeadModel",
        "base_huggingface_class": "GPT2LMHeadModel",
        "expected_config_class": "GPT2Config",
        "expected_tokenizer_class": "GPT2Tokenizer",
        "expected_hidden_size": 1024,
        "expected_layers": 24,
        "expected_attention_heads": 16,
        "expected_official_model_id": "openai-community/gpt2-medium",
        "expected_revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "expected_local_directory_from_source": "../downloaded_plms/gpt2/base",
        "phase2a_external_directory": "/workspace/NetLLM-artifacts/plms/gpt2/base",
        "networking_head": "Linear(1024, 3) -> Tanh",
    }
    print(json.dumps(mapping, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
