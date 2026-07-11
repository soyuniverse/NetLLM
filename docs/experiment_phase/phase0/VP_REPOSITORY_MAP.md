# VP repository map

## 1. 범위와 경로 기준

- NetLLM 원본 root: upstream 기준 `.` / `/workspace/NetLLM-source`
- VP root: upstream 기준 `viewport_prediction/` / `/workspace/NetLLM-source/viewport_prediction`
- 이 문서의 source 상대경로는 NetLLM 원본 root를 기준으로 한다.
- 표시 기준:
  - **실측**: 실제 cooked data를 한 sample만 읽거나 import/status 검사로 확인
  - **코드 확정**: source의 명시적 연산으로 결정
  - **shape 추론**: source 연산으로 계산했으나 model/data file 부재로 runtime 미확인
  - **미검증**: 실제 forward 또는 end-to-end 실행이 필요

## 2. entry point와 주요 파일

| 역할 | 상대경로 / 절대경로 | class/function |
|---|---|---|
| baseline entry | `viewport_prediction/run_baseline.py` / `/workspace/NetLLM-source/viewport_prediction/run_baseline.py` | `run`, `test`, `track_train`, `track_test` |
| current NetLLM entry | `viewport_prediction/run_plm.py` / `/workspace/NetLLM-source/viewport_prediction/run_plm.py` | `run`, `adapt`, `test`, `save_model`, `load_model` |
| paper checkpoint용 old entry | `viewport_prediction/run_old.py` / `/workspace/NetLLM-source/viewport_prediction/run_old.py` | `run`, `adapt`, `test` |
| config | `viewport_prediction/config.py` / `/workspace/NetLLM-source/viewport_prediction/config.py` | `Config`, `cfg` |
| dataset loader | `viewport_prediction/dataset/load_dataset.py` / `/workspace/NetLLM-source/viewport_prediction/dataset/load_dataset.py` | `pack_data`, `ViewportDataset`, `create_dataset` |
| preprocessing | `viewport_prediction/dataset/preprocess.py` / `/workspace/NetLLM-source/viewport_prediction/dataset/preprocess.py` | `process_datasets`, `simplify_datasets`, `euler_from_quaternion` |
| image preprocessing | `viewport_prediction/dataset/extract_saliency.py` / `/workspace/NetLLM-source/viewport_prediction/dataset/extract_saliency.py` | `saliency`, `processeachsub` |
| ViT feature extraction | `viewport_prediction/dataset/extract_features.py` / `/workspace/NetLLM-source/viewport_prediction/dataset/extract_features.py` | `extract_vit_features`, `store_feature` |
| current PLM pipeline | `viewport_prediction/models/pipeline.py` / `/workspace/NetLLM-source/viewport_prediction/models/pipeline.py` | `Pipeline` |
| PLM loader | `viewport_prediction/utils/plms_utils.py` / `/workspace/NetLLM-source/viewport_prediction/utils/plms_utils.py` | `load_plm`, `create_device_map_for_llama` |
| GPT-2 wrapper | `viewport_prediction/models/gpt2.py` / `/workspace/NetLLM-source/viewport_prediction/models/gpt2.py` | `GPT2NetworkingHeadModel` |
| Llama wrapper | `viewport_prediction/models/llama.py` / `/workspace/NetLLM-source/viewport_prediction/models/llama.py` | `LlamaNetworkingHeadModel` |
| networking head | `viewport_prediction/models/networking_head.py` / `/workspace/NetLLM-source/viewport_prediction/models/networking_head.py` | `NetworkingHead` |
| LoRA | `viewport_prediction/models/low_rank.py` / `/workspace/NetLLM-source/viewport_prediction/models/low_rank.py` | `peft_model` |
| normalize | `viewport_prediction/utils/normalize.py` / `/workspace/NetLLM-source/viewport_prediction/utils/normalize.py` | `normalize_data`, `denormalize_data` |
| metric/result | `viewport_prediction/utils/metrics.py`, `viewport_prediction/utils/result_notebook.py` / 해당 절대경로 | `compute_mae`, `compute_rmse`, `ResultNotebook` |

별도 `collate_fn`은 존재하지 않는다. 모든 entry point는 PyTorch `DataLoader`의 default collate를 사용한다.

## 3. current PLM 전체 흐름

### 3.1 dataset → dataset item

1. `create_dataset()`이 `cfg.dataset[dataset]`과 video/user split을 읽는다.
2. `pack_data()`가 각 `simple_5Hz_user*.csv`를 `np.loadtxt(..., float32)`로 읽는다.
3. timestamp인 첫 열을 버리고 `data[:, 1:]`만 trace로 저장한다.
4. `ViewportDataset.__getitem__()`이 sliding window로 `history`, `future`, `(video,user,timestep)`을 반환한다.

실제 Jin2022 video1/user1 한 trace에서 확인한 결과:

| 값 | shape/type | 상태 |
|---|---|---|
| item `history` | `(10, 3)`, `numpy.float32` | 실측 |
| item `future` | `(20, 3)`, `numpy.float32` | 실측 |
| item metadata | `(1, 1, 30)` | 실측 |
| batch size 1 `history` | `[1, 10, 3]`, `torch.float32` | 실측 |
| batch size 1 `future` | `[1, 20, 3]`, `torch.float32` | 실측 |
| batch metadata | 세 개의 `[B]` `torch.int64` Tensor list | 실측 |

세 channel의 순서는 preprocessing과 normalize code상 Roll, Pitch, Yaw다.

### 3.2 viewport history input → viewport embedding

- `run_plm.adapt()`는 history와 future를 각각 `normalize_data()`로 `[-1,1]` 범위에 맞춘다.
- `run_plm.test()`는 history만 normalize하고 raw future를 ground truth로 유지한다.
- `Pipeline.auto_regressive()`와 `Pipeline.teaching_forcing()`은 각 timestep의 3개 좌표를 하나씩 처리한다.
- 각 timestep에 `Conv1d(1,256,kernel_size=3) → LeakyReLU → Flatten → Linear(256, embed_size)`가 적용된다.

현재 구현은 `x[:, i, :]`에 대해 `.view(1, 256)`을 사용하므로 PLM path는 사실상 `B=1`을 전제한다.

| 단계 | shape | 상태 |
|---|---|---|
| 한 timestep 좌표 | `[1,3]` | 코드 확정, B=1 전제 |
| unbatched `Conv1d` 출력 | `[256,1]` | shape 추론 |
| flatten/view | `[1,256]` | 코드 확정 |
| 한 temporal embedding | `[1,1,E]` | shape 추론 |
| history temporal sequence | `[1,H,E]` | shape 추론 |

기본 `H=10`, `F=20`이다. 따라서 **과거 viewport 한 시점이 하나의 temporal token-like embedding이 되고 기본 history는 10개 embedding**이다.

### 3.3 image/image-feature loading → multimodal projection

- 조건: CLI `--multimodal` → `args.using_multimodal=True` → `Pipeline.using_multimodal=True`.
- raw image를 current PLM forward에서 직접 읽거나 ViT에 전달하지 않는다.
- `Pipeline.get_multimodal_information()`이 disk의 precomputed `.pth` dictionary를 `torch.load()`한다.
- feature는 offline `extract_vit_features()`가 ViT encoder의 CLS 위치 `x[:,0]`에서 생성하도록 작성되어 있다.
- feature key는 계산된 단일 `image_index` 문자열이다.
- `Linear(768,E)`와 `unsqueeze(1)` 후 temporal sequence 앞에 prepend한다.

| 단계 | shape | 상태 |
|---|---|---|
| stored ViT CLS feature | 예상 `[1,768]` | shape 추론; feature file 없음 |
| projected visual embedding | 예상 `[1,1,E]` | shape 추론 |
| multimodal input sequence | 예상 `[1,H+1,E]` | shape 추론 |

따라서 current PLM multimodal path에는 patch token sequence가 없다. sample당 선택된 한 frame의 CLS feature vector 하나가 visual token-like embedding 하나로 추가된다. 실제 feature file이 없어 key coverage와 runtime shape는 미검증이다.

### 3.4 LLM input embedding → attention mask → PLM

- temporal embedding과 optional visual embedding을 concatenate한 뒤 전체에 `LayerNorm(E)`를 적용한다.
- attention mask는 매 forward마다 `torch.ones(B,L,dtype=torch.long,device=self.device)`로 생성한다.
- padding, token type, modality별 mask 차이는 없다.
- PLM은 `inputs_embeds`로 호출되며 tokenizer output/input ID는 VP data path에서 사용하지 않는다.
- `utils.plms_utils.load_plm()`은 local `from_pretrained(model_path)`로 config/model/tokenizer를 읽는다.
- GPT-2 base의 code상 `E=1024`; Llama/Mistral base의 code상 `E=4096`이다.

PLM wrapper는 base decoder의 final hidden states `[B,L,E]`를 받아 LM vocabulary head 대신 `NetworkingHead`로 보낸다.

### 3.5 networking head → autoregressive future prediction

`NetworkingHead`는 `Linear(E,3) → Tanh`다.

- 일반 forward: final hidden state 하나를 골라 `[B,1,3]` continuous normalized Roll/Pitch/Yaw를 출력한다.
- teacher forcing: hidden sequence에서 `size-F-1:size-1` 구간을 골라 `[B,F,3]`을 한 번에 출력한다.
- auto-regressive inference:
  1. 현재 전체 embedding sequence를 PLM에 전달한다.
  2. 마지막 hidden state에서 다음 `[B,1,3]`을 예측한다.
  3. 예측 좌표를 다시 viewport embedding 하나로 변환해 sequence 끝에 붙인다.
  4. 이를 `F`번 반복하고 `[B,F,3]`으로 concatenate한다.

기본 non-multimodal sequence length는 forward마다 `H, H+1, ..., H+F-1`이고, multimodal이면 각 길이에 1이 더해진다. wrapper가 `past_key_values`를 받을 수 있고 반환 객체에도 cache field가 있지만 `Pipeline.auto_regressive()`는 이전 cache를 다음 호출에 전달하지 않는다. 전체 sequence를 매 step 다시 처리하는 구조다.

### 3.6 loss → metric → result/checkpoint

- adaptation loss: normalized prediction과 normalized future 사이 `nn.MSELoss()`.
- test metric: prediction을 degree로 denormalize한 뒤 `ResultNotebook`에서 rotation-aware MAE/RMSE를 계산한다.
- LoRA 사용 시 `plm.save_pretrained()`와 `modules_except_plm.bin`을 저장한다.
- LoRA 미사용 시 `model.bin` 전체 state dict를 저장한다.
- optimizer 구성상 `Pipeline.conv1d`는 어느 branch에서도 optimizer group에 포함되지 않는다. `--freeze-plm` branch에서는 networking head도 별도 group에 포함되지 않아, 논문상 trainable module과 실제 update 대상이 일치하는지는 실행 전 해결해야 할 불확실성이다.
- baseline result는 `data/results/<model>/<dataset>/<Hz>/result_*.csv`를 대상으로 한다.
- PLM result는 `data/results/<plm...>/<dataset>/<Hz>/*_results.csv`를 대상으로 한다.

실제 checkpoint/result 저장은 실행하지 않았다. current PLM result path에는 `result_` prefix가 없어 `ResultNotebook.write()`의 detail filename 치환과 맞지 않으며, 이어지는 `write_detail(result_path)`도 동일 path를 다시 연다. 따라서 summary 덮어쓰기 가능성이 코드상 확정되며 실행 전 해결 방침이 필요하다.

## 4. baseline과 PLM path 차이

| path | 입력 | 모델/출력 | image 사용 |
|---|---|---|---|
| regression | history `[B,H,3]` | channel별 sklearn LR, autoregressive `[B,F,3]` | 없음 |
| velocity | history `[B,H,3]` | 첫 sample 속도 extrapolation | 없음 |
| TRACK | history/future 좌표 + grayscale saliency frames | LSTM encoder/decoder, `[B,F,3]` | raw saliency PNG를 `cv2`로 읽음 |
| current PLM | temporal token-like embeddings + optional CLS feature | LLM hidden state + continuous coordinate head | optional precomputed feature 한 개 |
| old PLM | current와 유사한 별도 wrapper | 제공된 paper Llama checkpoint용 | optional precomputed feature 한 개 |

TRACK은 raw saliency image sequence를 사용하지만 current PLM multimodal path와 공유되지 않는다.

## 5. LiteVLM 대비 가능한 extension point 후보

아래는 구현 지점이 아니라 후보다.

1. Temporal selection 후보: `Pipeline.auto_regressive()` 및 `teaching_forcing()`에서 `batch_embeddings`를 concatenate한 직후, `embed_ln` 이전.
2. Visual selection 후보: 현재는 visual embedding이 하나뿐이라 selection 대상이 없다. 여러 frame feature를 명시적으로 구성하는 별도 input path가 먼저 필요하다.
3. Attention mask 연동 후보: embedding selection으로 sequence length가 바뀌면 동일 위치에서 ones mask 길이를 함께 갱신해야 한다.
4. Continuous draft 후보: `NetworkingHead` output 이후 좌표 trajectory proposal을 별도 연구할 수 있으나 LiteVLM/EAGLE-2 token speculative decoding과 동일하다고 볼 수 없다.

## 6. 아직 확정할 수 없는 부분

- precomputed feature file의 실제 `[1,768]` shape와 모든 `image_index` key 존재 여부
- 실제 GPT-2/Llama config와 `hidden_size`, cache default
- GPU forward에서의 dtype/device 이동과 memory usage
- `--multimodal`의 실제 end-to-end 정상 동작
- batch size 1을 넘는 current PLM 지원 가능성
- paper 성능에 사용된 old/current implementation의 정확한 대응 checkpoint
