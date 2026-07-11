# GPT-2 Medium artifact manifest

## 1. 식별 정보

| 항목 | 값 |
|---|---|
| NetLLM mapping | `plm_type=gpt2`, `plm_size=base` |
| repository ID | `openai-community/gpt2-medium` |
| alias | `gpt2-medium` |
| revision | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` |
| source URL | `https://huggingface.co/openai-community/gpt2-medium` |
| license | Modified MIT License (`license: mit`) |
| artifact path | `/workspace/NetLLM-artifacts/plms/gpt2/base` |
| cache path | `/workspace/NetLLM-artifacts/hf_cache` |
| download time | 2026-07-11 15:20:30–15:20:46 UTC |
| total artifact size | `1,522,851,760` bytes |

repository에는 독립 `LICENSE` 파일이 없으며 model card front matter가 `license: mit`를 선언하고 OpenAI GPT-2의 Modified MIT License를 연결한다.

## 2. 다운로드 명령

재다운로드 대상 경로가 존재하지 않을 때 다음을 실행한다.

```bash
cd /workspace/NetLLM
./scripts/experiment_phase/phase2a/download_phase2a_gpt2.sh
```

script 내부의 핵심 조건:

```text
repo_id=openai-community/gpt2-medium
revision=6dcaa7a952f72f9298047fd5137cd6e4f05f41da
local_dir=/workspace/NetLLM-artifacts/plms/gpt2/base
local_dir_use_symlinks=False
```

허용된 repository file pattern:

```text
README.md
config.json
generation_config.json
generation_config_for_text_generation.json
merges.txt
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
```

대상 경로가 이미 존재하면 script는 덮어쓰기나 삭제 없이 중단한다.

## 3. 파일 manifest

| 파일 | 크기(bytes) | SHA-256 |
|---|---:|---|
| `README.md` | 11,890 | `6ac13d83ab7a7fa24d6df5b1cb5e9654e373c35f773dd337bf11e3c4087923c2` |
| `config.json` | 718 | `ef1a44d889ad1a0acc7731c78134f1b87d2d222f110e97dd10fd4117331caf22` |
| `generation_config.json` | 124 | `b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805` |
| `generation_config_for_text_generation.json` | 165 | `a69e99c4a690c2015aaaeaef066516f1df9316317b3605869039fa896108efce` |
| `merges.txt` | 456,318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| `model.safetensors` | 1,519,984,962 | `fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd` |
| `tokenizer.json` | 1,355,256 | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `tokenizer_config.json` | 26 | `5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f` |
| `vocab.json` | 1,042,301 | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |

모든 artifact file은 일반 파일이며 symlink가 없다.

## 4. config manifest

```text
architectures=[GPT2LMHeadModel]
model_type=gpt2
n_embd=1024
n_layer=24
n_head=16
vocab_size=50257
n_positions=1024
n_ctx=1024
bos_token_id=50256
eos_token_id=50256
```

NetLLM wrapper 기대값과의 비교:

| 항목 | artifact | NetLLM 기대 | 판정 |
|---|---:|---:|---|
| hidden/embedding size | 1024 | 1024 | 일치 |
| layers | 24 | 24 | 일치 |
| output head input | 1024 | `plm.hidden_size` | 일치 |
| networking output | 해당 없음 | 3 | runtime에 별도 연결 |

## 5. local-only 검증

```bash
cd /workspace/NetLLM
export HF_HOME=/workspace/NetLLM-artifacts/hf_cache
export TRANSFORMERS_CACHE=/workspace/NetLLM-artifacts/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/venv/vp_netllm_repro/bin/python -B \
  scripts/experiment_phase/phase2a/check_phase2a_gpt2_artifact.py
/venv/vp_netllm_repro/bin/python -B \
  scripts/experiment_phase/phase2a/smoke_load_phase2a_gpt2.py
```

검증 script는 `local_files_only=True`를 사용한다. network fallback 방지를 확인한 Phase 2A 실행에서는 offline 환경 변수와 함께 unreachable localhost proxy도 설정했다.

## 6. checksum 재검증

```bash
cd /workspace/NetLLM-artifacts/plms/gpt2/base
sha256sum \
  README.md \
  config.json \
  generation_config.json \
  generation_config_for_text_generation.json \
  merges.txt \
  model.safetensors \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json
```

manifest와 하나라도 다르면 model을 load하지 말고 artifact 상태를 조사한다. 기존 artifact를 자동 삭제하거나 덮어쓰지 않는다.

## 7. 범위

이 manifest는 PyTorch safetensors 기반 local model load에 필요한 artifact만 다룬다. `pytorch_model.bin`, TensorFlow, Flax, Rust 및 ONNX artifact는 다운로드하지 않았으며 model forward, generation과 training 검증은 포함하지 않는다.
