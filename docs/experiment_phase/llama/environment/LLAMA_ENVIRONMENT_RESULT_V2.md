# Llama Environment Result V2

## Result

The new isolated environment was created successfully at
`/root/venvs/vp_netllm_llama`. The GPT-2 environment was not changed.

| Check | Result |
| --- | --- |
| Python | 3.8.10 |
| PyTorch | 2.2.0+cu121 |
| CUDA runtime / available | 12.1 / true |
| GPU | NVIDIA GeForce RTX 4090 |
| Total VRAM | 25,282,281,472 bytes (24,111 MiB) |
| Transformers | 4.34.1 |
| PEFT | 0.6.2 |
| Accelerate | 0.24.1 |
| huggingface-hub | 0.17.3 |
| tokenizers | 0.14.1 |
| safetensors | 0.5.3 |
| sentencepiece | 0.1.99 |
| protobuf | 4.25.3 |
| `pip check` | pass |
| Current Llama imports | pass |
| Checkpoint-era Llama imports | pass |

The documented checkpoint-era torch version is 2.1.0, while the isolated
inference environment uses the explicitly requested/project-pinned
2.2.0+cu121. This difference is retained in the manifest and must be considered
when interpreting reproduction results.

## Reproducibility files

- `experiments/vp/llama_environment_v2/environment_manifest.json`
- `experiments/vp/llama_environment_v2/pip_freeze.txt`
- `experiments/vp/llama_environment_v2/pip_check.txt`

Pip-freeze SHA-256:
`8a7e30ccc90703c0810291255f84704a7d5b2e95635ce04d94e80845b5de00f1`.
