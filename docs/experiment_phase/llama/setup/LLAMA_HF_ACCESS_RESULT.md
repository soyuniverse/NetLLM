# Llama2 Hugging Face access 결과

- 확인 시각: 2026-07-26 UTC
- Gate B 결과: 성공
- HF CLI: `/root/.local/bin/hf`
- HF CLI version: `1.24.0`
- 계정: `soyuniverse`
- Token 기록: 하지 않음

## Online 인증

다음 세 offline 변수를 명령 환경에서 제거한 상태로 `hf auth whoami`를 실행했다.

```text
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
HF_DATASETS_OFFLINE
```

인증 확인은 성공했으며 강제 login이나 token 갱신은 실행하지 않았다.

## Gated repository 접근

- model ID: `meta-llama/Llama-2-7b-hf`
- access-check target:
  `/root/NetLLM-assets/llama/access_check/config.json`
- resolved immutable revision:
  `01c7f73d771dfac7d292323805ebc428287df4f9`
- config SHA-256:
  `9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c`

`config.json` 단일 파일 download가 성공했으므로 해당 계정의 repository 접근권한이
확인됐다.

## Config identity

```text
model_type=llama
architectures=[LlamaForCausalLM]
hidden_size=4096
num_hidden_layers=32
num_attention_heads=32
torch_dtype=float16
```

Gate E에서는 `main`이 아니라 위 immutable revision만 사용해야 한다.
