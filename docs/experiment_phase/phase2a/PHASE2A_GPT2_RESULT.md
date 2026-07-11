# Phase 2A 결과: GPT-2 artifact 식별 및 local-only load

- 실행 시각: 2026-07-11 UTC
- 결과: **성공**
- upstream NetLLM: `/workspace/NetLLM-source`
- upstream commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- Python: `/venv/vp_netllm_repro/bin/python`
- artifact: `/workspace/NetLLM-artifacts/plms/gpt2/base`
- runtime output: `/workspace/NetLLM/experiments/vp/phase2a_runtime`

## 1. source에서 확인한 mapping

| 항목 | 확인 결과 | 근거 |
|---|---|---|
| `plm_type` | `gpt2` | `config.py::Config.plm_types` |
| `plm_size` | `base` | `config.py::Config.plm_sizes` |
| upstream size 설명 | `340M` | `config.py:12` |
| model class | `GPT2NetworkingHeadModel` | `utils/plms_utils.py::_MODEL_CLASSES` |
| base class | `GPT2LMHeadModel` | `models/gpt2.py` |
| config class | `GPT2Config` | `utils/plms_utils.py::_MODEL_CLASSES` |
| tokenizer class | `GPT2Tokenizer` | `utils/plms_utils.py::_MODEL_CLASSES` |
| hidden/embedding size | `1024` | `run_plm.py:274-275`, `model.hidden_size=config.n_embd` |
| layer 수 | `24` | ABR/CJS 공통 GPT-2 size mapping |
| attention head 수 | `16` | 공식 GPT-2 Medium config |
| networking head | `Linear(1024,3) → Tanh` | `run_plm.py:263-270`, `models/networking_head.py` |
| source 기본 local 경로 | `../downloaded_plms/gpt2/base` | VP cwd 기준 `cfg.plms_dir/gpt2/base` |

VP `config.py`는 `gpt2/base`를 340M이라고 명시한다. 같은 NetLLM repository의 ABR 및 CJS mapping은 GPT-2 size를 다음처럼 구분한다.

```text
small: hidden 768,  12 layers
base:  hidden 1024, 24 layers
large: hidden 1280, 36 layers
xl:    hidden 1600, 48 layers
```

공식 GPT-2 Medium config는 `n_embd=1024`, `n_layer=24`, `n_head=16`이므로 `base` mapping은 하나로 확정된다. NetLLM의 `340M` 표기와 현재 Hugging Face model card의 `355M` 표기는 parameter counting/표기 차이이며 architecture는 동일하다.

## 2. 선택한 공식 artifact

- repository ID: `openai-community/gpt2-medium`
- 일반 alias: `gpt2-medium` — 공식 repository로 redirect됨
- resolved revision: `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`
- source URL: `https://huggingface.co/openai-community/gpt2-medium`
- license: Modified MIT License
- license 근거: repository metadata의 `license:mit`와 model card

공식 model ID가 VP source에 직접 적혀 있지는 않지만, size label, hidden size, layer 수와 공식 config의 조합이 유일하게 GPT-2 Medium과 일치하므로 다운로드를 진행했다.

## 3. artifact 다운로드

실행 명령:

```bash
cd /workspace/NetLLM
./scripts/experiment_phase/phase2a/download_phase2a_gpt2.sh
```

다운로드 조건:

```text
HF_HOME=/workspace/NetLLM-artifacts/hf_cache
TRANSFORMERS_CACHE=/workspace/NetLLM-artifacts/hf_cache
target=/workspace/NetLLM-artifacts/plms/gpt2/base
revision=6dcaa7a952f72f9298047fd5137cd6e4f05f41da
```

PyTorch용 `model.safetensors`, config, tokenizer와 소형 generation metadata만 받았다. `pytorch_model.bin`, TensorFlow, Flax, Rust 및 ONNX weight는 받지 않았다.

- 다운로드 시작: `2026-07-11T15:20:30Z`
- 다운로드 완료: `2026-07-11T15:20:46Z`
- artifact file 수: 9개
- artifact 실제 합계: `1,522,851,760` bytes
- symlink: 없음
- download log: `experiments/vp/phase2a_runtime/logs/20260711T152030Z_gpt2_download.log`

## 4. artifact 구조 및 config 검사

필수 파일을 모두 확인했다.

```text
config.json
model.safetensors
tokenizer_config.json
vocab.json
merges.txt
```

추가로 `tokenizer.json`, 두 generation config와 model card를 보존했다. GPT-2 원본 tokenizer에는 pad token이 없으므로 별도 `special_tokens_map.json`은 필수가 아니다. NetLLM `load_plm()`은 load 후 `<pad>`를 추가하고 embedding을 resize한다.

local-only config/tokenizer 검사 결과:

```text
config_class=GPT2Config
tokenizer_class=GPT2Tokenizer
model_type=gpt2
n_embd=1024
n_layer=24
n_head=16
vocab_size=50257
n_positions=1024
n_ctx=1024
```

## 5. local-only model load

다음 조건을 동시에 사용했다.

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
local_files_only=True
HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=unreachable localhost endpoint
```

원본 `load_plm()`은 `local_files_only` parameter를 전달할 interface가 없다. 따라서 external smoke script가 원본 `_MODEL_CLASSES['gpt2']`에서 실제 `GPT2Config`, `GPT2Tokenizer`, `GPT2NetworkingHeadModel` class를 가져와 동일 순서로 호출하되 모든 `from_pretrained()`에 `local_files_only=True`를 명시했다. 원본 `add_special_tokens()`와 `NetworkingHead`도 그대로 호출했다.

load 결과:

| 항목 | 결과 |
|---|---|
| model class | `GPT2NetworkingHeadModel` |
| base class | `GPT2LMHeadModel` |
| config class | `GPT2Config` |
| tokenizer class | `GPT2Tokenizer` |
| dtype | `torch.float32` |
| device | `cpu` |
| hidden size | `1024` |
| layers / heads | `24 / 16` |
| pretrained parameter count | `354,823,168` |
| `<pad>` 추가 후 parameter count | `354,824,192` |
| pad token / ID | `<pad>` / `50257` |
| tokenizer length | `50,258` |
| networking head before setup | `None` |
| networking head after setup | `NetworkingHead` |
| networking head | `Linear(1024,3) → Tanh` |
| head 연결 후 total parameters | `354,827,267` |
| LoRA | 적용하지 않음 |
| forward | 실행하지 않음 |

load log: `experiments/vp/phase2a_runtime/logs/20260711T152110Z-model-load-offline.log`

## 6. 원본 및 환경 무결성

| 항목 | 작업 전 | 작업 후 |
|---|---|---|
| upstream commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| upstream status porcelain | 빈 결과 | 빈 결과 |
| upstream diff name-status | 빈 결과 | 빈 결과 |
| upstream `__pycache__` | 0개 | 0개 |
| repro environment freeze hash | `4601ad2592a119fc91953a4cc142783db59d8a1cdb548097205a1ac5c057ffbe` | 동일 |

원본 안에 model, cache, symlink, log 또는 untracked file을 만들지 않았다. artifact는 Git repository 밖의 `/workspace/NetLLM-artifacts`에만 존재한다.

## 7. 제외 범위와 다음 단계

- input ID 또는 embedding을 model에 전달하지 않았다.
- VP dataset, Pipeline 및 tensor flow를 실행하지 않았다.
- model forward, generation, adaptation, training, LoRA 및 checkpoint 저장을 수행하지 않았다.
- Llama, image, precomputed feature와 다른 model은 다운로드하지 않았다.
- selector를 구현하지 않았다.

Phase 2A 성공 조건은 충족했다. 다음 단계에서는 별도 승인 후에만 GPT-2 VP minimal forward와 tensor shape를 검증할 수 있다. 현재 남은 주요 blocker는 원본 `run_plm.run()`이 output directory를 생성하는 구조, batch-size 1 전제, optimizer 구성 및 PLM result path 불확실성이다.
