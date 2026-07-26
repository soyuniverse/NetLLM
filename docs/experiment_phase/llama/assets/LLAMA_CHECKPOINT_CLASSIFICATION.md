# Llama checkpoint classification

- Gate D 결과: **실패 — checkpoint incomplete**
- archive source: `/root/NetLLM/try_llama2_7b.zip`
- canonical path: `/root/NetLLM-assets/checkpoints/try_llama2_7b`
- extraction path safety: pass
- staging→canonical manifest checksum match: yes

## 유형

판정: **LoRA checkpoint**

파일:

| 파일 | 크기 |
|---|---:|
| `adapter_config.json` | 542 bytes |
| `adapter_model.bin` | 67,155,338 bytes |
| `modules_except_plm.bin` | 16,900,050 bytes |
| `README.md` | 5,479 bytes |

Relative-path checksum manifest:

```text
/root/NetLLM-assets/manifests/checkpoint_manifest.sha256
```

Manifest file SHA-256:
`44cbaaa6a174207bd98c21030200ad4244f09b5273a3ec0355ece0830519c1c6`

## LoRA configuration

```text
peft_type=LORA
r=32
lora_alpha=32
lora_dropout=0.05
target_modules=[v_proj,q_proj]
task_type=FEATURE_EXTRACTION
inference_mode=true
bias=none
```

`base_model_name_or_path`는 다음 과거 machine local path만 기록한다.

```text
/data/data1/wuduo/2023_prompt_learning/downloaded_plms/llama/base
```

Repository ID와 immutable revision은 checkpoint 안에 없다.

## Safe state-dict 검사

두 pickle 기반 `.bin`은 torch 2.1의 `weights_only=True`, CPU map location으로만 읽었다.
임의 object/code load는 허용하지 않았다.

### adapter_model.bin

- safe load: 성공
- key count/tensor count: `128 / 128`
- dtype: `torch.float32`
- total parameters: `16,777,216`
- 32개 layer의 `q_proj`, `v_proj` 각각에 LoRA A/B key 존재
- A/B shape: `[32,4096]`, `[4096,32]`

### modules_except_plm.bin

- safe load: 성공
- key count/tensor count: `10 / 10`
- dtype: `torch.float32`
- total parameters: `4,224,003`
- VP embedding `[4096,256]`
- multimodal embedding `[4096,768]`
- LayerNorm `[4096]`
- Conv1d `[256,1,3]`
- task/networking head `[3,4096]`

현재 upstream의 networking head key는 `networking_head.0.*`인데 checkpoint key summary에는
`task_head.0.*`가 보인다. 실제 model/checkpoint load 전에는 key mapping 또는 code version
호환성을 추측할 수 없다.

## Completeness 판정

존재:

- adapter config
- adapter weight
- `modules_except_plm.bin`
- LoRA rank/target modules

누락 또는 불확정:

- exact base repository ID/revision
- training args/config file
- upstream commit/code version provenance
- `using_multimodal`
- train/evaluation dataset configuration
- history/future/trim/frequency/sample-step
- seed, batch size, loss definition
- checkpoint 선택 기준
- PEFT README에는 `PEFT 0.6.0`만 있고 training hyperparameter는 없음
- current upstream state-dict key와의 compatibility

따라서 random head 없이 checkpoint를 정확히 재현할 충분한 provenance가 없으며
**checkpoint incomplete**로 판정한다.
