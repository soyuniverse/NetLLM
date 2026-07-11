#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path


ARTIFACT = Path("/workspace/NetLLM-artifacts/plms/gpt2/base")
EXPECTED = {
    "model_type": "gpt2",
    "n_embd": 1024,
    "n_layer": 24,
    "n_head": 16,
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_ctx": 1024,
}
REQUIRED = {
    "config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/workspace/NetLLM-artifacts/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/NetLLM-artifacts/hf_cache")

    from transformers import AutoConfig, GPT2Tokenizer

    if not ARTIFACT.is_dir():
        raise RuntimeError(f"Artifact directory missing: {ARTIFACT}")
    names = {path.name for path in ARTIFACT.iterdir() if path.is_file()}
    missing = sorted(REQUIRED - names)
    if missing:
        raise RuntimeError(f"Required artifact files missing: {missing}")

    config = AutoConfig.from_pretrained(str(ARTIFACT), local_files_only=True)
    tokenizer = GPT2Tokenizer.from_pretrained(str(ARTIFACT), local_files_only=True)
    actual = {key: getattr(config, key) for key in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f"Config mismatch: expected {EXPECTED}, got {actual}")

    files = []
    total_size = 0
    for path in sorted(item for item in ARTIFACT.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_size += size
        files.append({
            "path": str(path.relative_to(ARTIFACT)),
            "size_bytes": size,
            "sha256": sha256(path),
        })

    report = {
        "artifact_path": str(ARTIFACT),
        "offline": True,
        "total_size_bytes": total_size,
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "config": actual,
        "files": files,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
