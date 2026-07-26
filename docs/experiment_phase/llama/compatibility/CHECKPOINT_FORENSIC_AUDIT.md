# Llama checkpoint forensic audit

- 확인 시각: 2026-07-26 UTC
- Gate 1 결과: 성공
- checkpoint:
  `/root/NetLLM-assets/checkpoints/try_llama2_7b`
- 검사 정책: CPU, `torch.load(..., weights_only=True)`, tensor 값 미출력

## 포함 파일

```text
adapter_config.json
adapter_model.bin
modules_except_plm.bin
README.md
```

별도 training args/config는 없다. README는 PEFT model-card template이며 framework version
`PEFT 0.6.0` 외 training configuration은 제공하지 않는다.

## Adapter

| 항목 | 값 |
|---|---|
| base_model_name_or_path | `/data/data1/wuduo/2023_prompt_learning/downloaded_plms/llama/base` |
| revision | `null` |
| PEFT type | `LORA` |
| rank | `32` |
| alpha | `32` |
| dropout | `0.05` |
| target modules | `q_proj`, `v_proj` |
| task type | `FEATURE_EXTRACTION` |
| inference mode | `true` |
| bias | `none` |
| key prefix | `base_model.model.model.layers` |
| key/tensor count | `128 / 128` |
| parameter count | `16,777,216` |
| dtype count | `torch.float32: 128` |

32개 layer마다 q/v projection의 LoRA A/B tensor가 존재한다. Shapes는 A `[32,4096]`,
B `[4096,32]`로 rank/config 및 Llama2-7B hidden size와 일치한다.

## modules_except_plm.bin

- top-level type: `OrderedDict`
- key/tensor count: `10 / 10`
- total parameter count: `4,224,003`
- dtype count: `torch.float32: 10`
- optimizer state: 없음
- scheduler state: 없음

Upstream `Pipeline.modules_except_plm` 순서에 따른 prefix 분류:

| Prefix | 분류 | Keys / shapes |
|---|---|---|
| `0` | viewport projection (`embed_vp`) | weight `[4096,256]`, bias `[4096]` |
| `1` | multimodal projection (`embed_multimodal`) | weight `[4096,768]`, bias `[4096]` |
| `2` | LayerNorm (`embed_ln`) | weight/bias `[4096]` |
| `3.0` | Conv1d | weight `[256,1,3]`, bias `[256]` |
| `4.task_head.0` | checkpoint networking/task head | weight `[3,4096]`, bias `[3]` |

Multimodal projection tensor가 저장돼 있지만 current Pipeline은 `using_multimodal=False`에서도
해당 module을 항상 생성하고 `modules_except_plm`에 포함한다. 따라서 key 존재만으로 실제
multimodal training을 증명할 수 없다.

## Machine-readable manifests

- `experiments/vp/llama_compatibility/adapter_key_manifest.json`
- `experiments/vp/llama_compatibility/modules_except_plm_key_manifest.json`

각 manifest는 모든 key, shape, dtype 및 numel만 포함하며 tensor 값은 포함하지 않는다.
