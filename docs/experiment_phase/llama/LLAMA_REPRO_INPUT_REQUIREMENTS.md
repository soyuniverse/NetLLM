# Llama2 NetLLM 재현 입력 요구사항

- audit 시각: 2026-07-26 UTC
- Gate 1 결과: 실패/blocked
- 중단 위치: Gate 1
- 다운로드 실행: 없음
- Llama environment 생성: 없음

## 탐색 범위

다음을 read-only로 탐색했다.

- `/root`
- `/workspace`
- 현재 mount table

현재 container에는 `/root` 또는 `/workspace`에 연결된 별도 persistent/mounted storage가
없다. 검색 결과 Llama model/checkpoint로 오인할 수 있는 Python package source와 GPT-2
artifact는 제외했다.

## Asset 분류

### 1. Llama2-7B base model

상태: **없음**

다음 Llama base artifact marker가 발견되지 않았다.

```text
config.json + Llama architecture
tokenizer.model
Llama tokenizer files
model*.safetensors
pytorch_model*.bin
consolidated*.pth
```

Upstream mapping은 `plm_type=llama`, `plm_size=base`를 7B로 정의하며 local path
`cfg.plms_dir/llama/base`를 기대한다. 실제 loader는 다음 class를 사용한다.

- config: `transformers.LlamaConfig`
- tokenizer: `transformers.LlamaTokenizer`
- model: `models.llama.LlamaNetworkingHeadModel`
- base class: `transformers.LlamaForCausalLM`
- expected hidden size in `run_plm.py`: `4096`
- device map: 32 transformer blocks 전제

그러나 upstream source는 정확한 Llama2 repository ID, revision 및 file manifest를 pin하지
않는다. 따라서 임의 model ID로 대체하거나 다운로드하지 않았다.

필요 입력:

- 팀이 승인한 정확한 Llama2-7B Hugging Face repository ID
- immutable revision/commit SHA
- license/access 승인 상태
- local target path
- expected file list와 SHA-256 manifest
- 해당 artifact가 Llama 2인지 확인할 `config.json` identity

### 2. Fine-tuned checkpoint 또는 LoRA adapter

상태: **없음**

`adapter_config.json`, adapter weight, `modules_except_plm.bin`, NetLLM `model.bin`,
checkpoint config 및 training args가 발견되지 않았다.

Upstream checkpoint format은 training의 `rank`에 따라 다르다.

LoRA (`rank != -1`):

```text
<checkpoint>/
  adapter_config.json
  adapter_model.safetensors 또는 adapter_model.bin
  modules_except_plm.bin
```

Full model (`rank == -1`):

```text
<checkpoint>/
  model.bin
```

`modules_except_plm.bin`에는 viewport embedding, multimodal embedding, LayerNorm,
Conv1d 및 networking head state가 포함되므로 adapter weight만으로는 baseline을 복구할 수
없다.

필요 입력:

- checkpoint 또는 adapter directory 전체
- checkpoint 전체 SHA-256 manifest
- 사용한 upstream commit
- base model repository ID와 revision
- `rank` 및 LoRA configuration
- `freeze_plm`
- `using_multimodal`
- train/test dataset
- history/future window
- trim head/tail
- dataset frequency와 sample step
- batch size, seed
- scheduled sampling 및 mix rate
- training/evaluation loss 정의
- checkpoint 선택 기준(best/checkpoint/epoch/step)
- tokenizer special token 처리

이 정보 없이는 `run_plm.load_model()`의 LoRA/full-model branch와 정확한 loss를 결정할 수
없다.

### 3. Cooked viewport CSV

상태: **존재**

- path:
  `/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022`
- structure: `video1..video27/5Hz/simple_5Hz_user*.csv`
- file count: `2268`
- total size: `27,396,178 bytes`
- relative-name content manifest SHA-256:
  `ebc8038f6ff18143407bf0359a100b4ff41181714d0607daded06ebbd0a5d288`
- Git-tracked: yes

이는 viewport coordinate CSV이며 원본 image dataset이 아니다.

### 4. Team VP original image dataset

상태: **없음**

Upstream이 기대하는 위치:

```text
/root/NetLLM-source/viewport_prediction/data/images/
  Jin2022images/
    saliencyMap/
      video1_images/
      ...
      video27_images/
    features/
      video1_images/
      ...
      video27_images/
```

실제 `viewport_prediction/data/images` directory는 존재하지 않는다.

- 팀 original image file count: `0`
- 팀 original image total size: `0 bytes`
- directory structure: 없음
- checksum manifest: 생성 불가(asset 없음)

Upstream loader는 saliency map PNG를
`Jin2022images/saliencyMap/video{video}_images/{frame}.png` 형식으로 읽는다. Source의
frame-count contract로부터 예상되는 전체 frame 수는 45,900개지만, 이 값은 제공될 팀
artifact의 실제 manifest를 대신하지 않는다.

Repository root의 `/root/NetLLM-source/images`에는 논문/README용 PNG 6개,
`3,141,871 bytes`가 있지만 VP team dataset이 아니므로 사용하지 않는다. 다른 repository의
유사 image도 대체하지 않았다.

필요 입력:

- 팀이 사용한 정확한 Jin2022 original/saliency image directory
- video별 file count와 naming convention
- 전체 size
- relative path + file SHA-256 manifest
- dataset version/source/전처리 provenance
- multimodal checkpoint라면 precomputed `features` 전체와 그 생성 configuration/manifest

## Hugging Face access 상태

- `HF_TOKEN` 계열 environment variable: 없음
- `/root/.cache/huggingface/token`: 없음
- `/root/.huggingface/token`: 없음
- Hugging Face Hub가 resolve한 token: 없음

Token 값은 조회하거나 기록하지 않았다. Gated Llama2 asset을 사용하려면 license가 승인된
계정의 read token 또는 팀이 제공한 verified local artifact가 필요하다.

## GPU 및 Gate 2/3 판단

- GPU: `NVIDIA GeForce RTX 4090`
- total memory: `24564 MiB`
- audit 시 free memory: `24110 MiB`

총 GPU memory가 30GB 미만이므로 Llama2 LoRA training은 자동 시작할 수 없다. Inference-only
smoke test도 base/checkpoint 확보 후 dtype, quantization 사용 여부, device map 및 예상 peak
memory를 먼저 계산해야 한다.

## 정확한 blocker

Gate 2로 진행하기 전에 최소한 다음이 필요하다.

1. 승인된 Llama2-7B base model ID/revision 또는 verified local artifact
2. matching fine-tuned NetLLM checkpoint 전체
3. checkpoint training/evaluation configuration
4. checkpoint가 multimodal이면 팀 original image/features dataset 전체
5. 각 asset checksum manifest
6. gated artifact를 받을 경우 유효한 Hugging Face access

Base model과 fine-tuned checkpoint가 없으므로 별도 Llama environment를 만드는 것만으로는
Gate 3 baseline을 재현할 수 없다. 절대 제약에 따라 Gate 2 환경 생성, Gate 3 baseline,
Gate 4 identity, Gate 5 benchmark, Gate 6 visualization, Gate 7 summary 및 Gate 8
continuous draft-and-verify prototype을 시작하지 않았다.
