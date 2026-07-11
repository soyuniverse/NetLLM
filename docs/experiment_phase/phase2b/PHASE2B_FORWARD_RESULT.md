# Phase 2B 결과: Minimal VP GPT-2 forward

- 실행 시각: 2026-07-11 UTC
- 결과: **성공**
- Python: `/venv/vp_netllm_repro/bin/python`
- NetLLM source: `/workspace/NetLLM-source`
- NetLLM commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- GPT-2 artifact: `/workspace/NetLLM-artifacts/plms/gpt2/base`
- runtime: `/workspace/NetLLM/experiments/vp/phase2b_runtime`

## 1. 실행 범위

실제 tracked Jin2022 test sample 한 개에 대해 다음 경로를 정확히 한 번 실행했다.

```text
cooked Jin2022 sample
→ history normalization
→ timestep viewport embedding
→ GPT-2 input embedding sequence
→ GPT-2 full-sequence forward × 20 autoregressive steps
→ NetworkingHead
→ coordinate feedback embedding
→ prediction [1,20,3]
```

`Pipeline.inference()` 호출은 한 번이며, 원본 `Pipeline.auto_regressive()` 내부에서 GPT-2가 미래 window 길이만큼 20회 호출됐다. training, backward, optimizer, scheduler, adaptation, LoRA, checkpoint 및 result CSV 저장은 수행하지 않았다.

## 2. 실제 sample

원본 `dataset.load_dataset.create_dataset()`과 PyTorch default `DataLoader(batch_size=1, shuffle=False)`를 사용했다.

| 항목 | 값 |
|---|---|
| dataset / split | `Jin2022` / `test` |
| dataset index | `0` |
| test dataset 길이 | `1,698` |
| video / user / timestep | `4 / 83 / 30` |
| CSV | `/workspace/NetLLM-source/viewport_prediction/data/viewports/Jin2022/video4/5Hz/simple_5Hz_user83.csv` |
| Git tracked | `True` |
| history / future | `10 / 20` |
| frequency / step | `5Hz / 15` |
| trim head / tail | `30 / 60` |
| batch size | `1` |

입력:

```text
raw history: [1,10,3], torch.float32, CPU
raw future:  [1,20,3], torch.float32, CPU
normalized history range: [-0.0085593555, 0.8024204969]
```

synthetic data는 사용하지 않았다.

## 3. model 및 Pipeline 구성

원본 `utils.plms_utils.load_plm()`을 사용했다. source의 상대 artifact 경로는 수정하지 않고 runtime object만 다음과 같이 override했다.

```text
cfg.plms_dir before: ../downloaded_plms
cfg.plms_dir runtime: /workspace/NetLLM-artifacts/plms
resolved path: /workspace/NetLLM-artifacts/plms/gpt2/base
```

다음 offline 조건과 접속 불가능한 localhost proxy를 함께 적용했다.

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/workspace/NetLLM-artifacts/hf_cache
TRANSFORMERS_CACHE=/workspace/NetLLM-artifacts/hf_cache
```

| 항목 | 결과 |
|---|---|
| PLM | `GPT2NetworkingHeadModel` |
| tokenizer | `GPT2Tokenizer` |
| hidden size | `1024` |
| layers / heads | `24 / 16` |
| dtype / device | `torch.float32 / cuda:0` |
| parameter count | `354,827,267` |
| networking head | `Linear(1024,3) → Tanh` |
| Pipeline embed size | `1024` |
| future window | `20` |
| multimodal | `False` |
| model/Pipeline mode | `eval()` |
| grad mode | `torch.inference_mode()` |

## 4. 주요 tensor shape

| 단계 | shape | dtype | device |
|---|---|---|---|
| raw / normalized history | `[1,10,3]` | `float32` | CPU → CUDA |
| single history timestep | `[1,3]` | `float32` | CUDA |
| initial Conv1d output | `[256,1]` | `float32` | CUDA |
| flattened initial representation | `[256,1]`, 이후 `.view(1,256)` | `float32` | CUDA |
| projected temporal embedding | `[1,1,1024]` | `float32` | CUDA |
| concatenated history sequence | `[1,10,1024]` | `float32` | CUDA |
| LayerNorm output | `[1,10,1024]` | `float32` | CUDA |
| step 0 GPT-2 input / hidden | `[1,10,1024]` / `[1,10,1024]` | `float32` | CUDA |
| step 1 GPT-2 input / hidden | `[1,11,1024]` / `[1,11,1024]` | `float32` | CUDA |
| step 19 GPT-2 input / hidden | `[1,29,1024]` / `[1,29,1024]` | `float32` | CUDA |
| NetworkingHead selected input | `[1,1,1024]` | `float32` | CUDA |
| step prediction | `[1,1,3]` | `float32` | CUDA |
| feedback Conv1d output | `[1,256,1]` | `float32` | CUDA |
| feedback embedding | `[1,1,1024]` | `float32` | CUDA |
| final prediction | `[1,20,3]` | `float32` | CUDA |

모든 기록 tensor는 finite이고 `requires_grad=False`였다. 각 tensor의 min, max, mean은 `phase2b_tensor_trace.json`에 기록했다.

## 5. 최종 output sanity

```text
shape=[1,20,3]
dtype=torch.float32
device=cuda:0
finite=True
requires_grad=False
min=-0.9917044639587402
max=0.9857870936393738
mean=-0.11429629176855087
Tanh range=True
```

NetworkingHead와 viewport projection은 학습되지 않은 random initialization이므로 prediction 값이나 ground truth 차이를 성능 결과로 해석하지 않는다. MAE/RMSE를 계산하지 않았다.

## 6. runtime 및 GPU memory

| 측정 | 결과 |
|---|---:|
| original loader CPU model load | `2.175828 s` |
| model CPU→GPU transfer | `0.174342 s` |
| networking head + Pipeline 구성 | `0.005253 s` |
| traced single inference | `0.234260 s` |
| GPU allocated before forward | `1,448,703,488 bytes` (`1381.591 MiB`) |
| GPU peak allocated | `1,476,698,624 bytes` (`1408.290 MiB`) |
| GPU peak reserved | `1,491,075,072 bytes` (`1422.000 MiB`) |

forward 시간에는 hook의 tensor 통계 계산과 CUDA synchronization이 포함된다. 단 한 번만 측정했으므로 benchmark 수치가 아니라 이번 traced smoke run의 관측값이다.

## 7. cache 동작

- GPT-2 config `use_cache=True`
- 20개 step 모두 24-layer `past_key_values` 반환
- step 0 첫 key/value: `[1,16,10,64]`
- step 1 첫 key/value: `[1,16,11,64]`
- step 19 첫 key/value: `[1,16,29,64]`
- 20개 step 모두 입력 `past_key_values=None`
- 결론: cache는 반환되지만 `Pipeline.auto_regressive()`가 재사용하지 않는다.

실제 GPT-2 input sequence length는 `10,11,...,29`이며 매 step 전체 sequence를 다시 처리한다.

## 8. 발견한 source-level 문제

1. `x[:, i, :]`와 `.view(1,256)` 때문에 initial history embedding은 batch size 1을 전제한다.
2. initial history Conv1d는 unbatched `[256,1]`, autoregressive feedback Conv1d는 batched `[1,256,1]`로 shape contract가 비대칭이다.
3. `LayerNorm`은 최초 history sequence에 한 번만 적용되며 이후 append되는 feedback embedding에는 적용되지 않는다.
4. `past_key_values`를 매 step 반환하지만 다음 step에 전달하지 않아 full sequence를 반복 계산한다.
5. step 19 prediction 뒤에도 feedback embedding을 계산해 length 30 sequence를 만들지만 후속 GPT-2 호출이 없어 사용되지 않는다.
6. 원본 `load_plm()`은 `local_files_only` argument를 노출하지 않는다. 이번에는 local path, offline flags와 invalid proxy로 network fallback을 차단했다.

원본은 수정하지 않았으며 위 항목은 관측 및 진단 결과다.

## 9. 산출물

- runner: `scripts/experiment_phase/phase2b/run_phase2b_vp_forward.py`
- shell wrapper: `scripts/experiment_phase/phase2b/run_phase2b_vp_forward.sh`
- full JSON trace: `experiments/vp/phase2b_runtime/phase2b_tensor_trace.json`
- stdout/stderr log: `experiments/vp/phase2b_runtime/phase2b_forward.log`
- run status: `experiments/vp/phase2b_runtime/run_status.txt`
- upstream before/after: `experiments/vp/phase2b_runtime/upstream_before.txt`, `upstream_after.txt`

JSON trace SHA-256:

```text
b6ceedae4b741936f1d287f3cf8725c9bc01f6676bff7bb388ad3c92bb3c7c75
```

## 10. 무결성

| 항목 | 작업 전 | 작업 후 |
|---|---|---|
| upstream commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| upstream status/diff | 빈 결과 | 빈 결과 |
| upstream `__pycache__` | 0개 | 0개 |
| repro environment freeze hash | `4601ad2592a119fc91953a4cc142783db59d8a1cdb548097205a1ac5c057ffbe` | 동일 |
| artifact fingerprint | `9eb853117884a343db1500f673fb1b0f79104e40074405d1bffabd8b067a0680` | 동일 |

upstream의 `logs`, `data/results`, `data/ft_plms`, `downloaded_plms`는 생성되지 않았다.

## 11. 다음 단계

Phase 2B의 forward 및 tensor-contract 목표는 완료됐다. 다음 연구 단계 전에 batch-size 1 전제, feedback LayerNorm 비대칭, cache 미사용과 마지막 unused embedding을 원본 변경 없이 어떻게 다룰지 사용자 결정이 필요하다. Training, LoRA, adaptation 또는 selector 구현은 수행하지 않았다.
