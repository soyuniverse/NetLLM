# 누락 요구사항 및 충돌

## 1. Phase 1 전 즉시 blocker

| 우선순위 | blocker | 상태/영향 |
|---:|---|---|
| 1 | 원본 밖 writable output 경로 | 현재 wrapper run script는 원본 아래에 log/result를 생성하므로 그대로 실행할 수 없음 |
| 2 | `peft`/`accelerate` runtime import 불일치 | `run_plm.py` import 자체가 실패하므로 PLM 단계 진입 불가 |
| 3 | pretrained PLM 없음 | 실제 code가 찾는 `/workspace/NetLLM-source/downloaded_plms/<type>/<size>`가 없음 |
| 4 | image/saliency/precomputed feature 없음 | TRACK과 `--multimodal` 실행 불가 |
| 5 | VP/TRACK checkpoint 없음 | checkpoint 기반 test 불가; regression만 weight 없이 가능 |

Phase 1 regression smoke 자체에는 2~5가 필수는 아니지만, blocker 1은 원본 보호를 위해 먼저 해결해야 한다.

## 2. dataset 및 model artifact

| 항목 | 분류 | 실제 확인 |
|---|---|---|
| cooked Jin2022 | Present and structurally verified | upstream `viewport_prediction/data/viewports/Jin2022`; 2,268 CSV |
| cooked Wu2017 | Present and structurally verified | upstream `viewport_prediction/data/viewports/Wu2017`; 864 CSV |
| raw Jin2022/Wu2017 | Path referenced by code but missing | `dataset/preprocess.py`의 `/data2/wuduo/...` 경로 없음 |
| saliency/raw image | Missing | `viewport_prediction/data/images` 없음 |
| precomputed ViT feature | Path referenced by code but missing | `config.py`의 `data/images/*images/features` 없음 |
| pretrained GPT-2 | Path referenced by code but missing | `/workspace/NetLLM-source/downloaded_plms/gpt2/base` 없음 |
| pretrained Llama | Path referenced by code but missing | `/workspace/NetLLM-source/downloaded_plms/llama/base` 없음 |
| Hugging Face cache | Missing | `/workspace/.hf_home`은 비어 있고 `/root/.cache/huggingface` 없음 |
| NetLLM VP checkpoint | Missing | `viewport_prediction/data/ft_plms` 없음 |
| baseline/TRACK checkpoint | Missing | `viewport_prediction/data/models` 없음 |
| VP result/log | Missing | `viewport_prediction/data/results`, `viewport_prediction/logs` 없음 |

## 3. dependency 누락 및 불일치

### 설치되어 있으나 import 실패

- `peft==0.6.2`
- `accelerate==0.32.1`
- `huggingface-hub==0.17.3`
- 실패 symbol: `split_torch_state_dict_into_shards`
- 관련 source: upstream `viewport_prediction/models/low_rank.py::peft_model`, `viewport_prediction/run_plm.py`

### 미설치

- `torchvision`
- 관련 source: upstream `viewport_prediction/dataset/extract_features.py`, `viewport_prediction/dataset/extract_saliency.py`
- `requirements-vp.txt`에는 포함되어 있지 않다.

환경 변경은 수행하지 않았다. 호환 version 선정과 설치는 별도 승인이 필요하다.

## 4. 경로 및 문서 불일치

### 읽기 전용 연구 문서 형식

- 프로젝트 `docs/RESEARCH_DIRECTION.md`는 `patches/`가 포함된 fenced block 도중 파일이 끝나며 closing fence와 이후 문맥이 없다.
- 현재 `docs/MEETING_NOTES.md`에는 문서명 heading과 `markdown` fence를 감싼 pasted-document 형식 및 `다음과 같다.r` 문자열이 존재한다.
- 두 파일은 사용자 변경을 포함한 읽기 전용 자료이므로 수정하지 않았다. 잘린 내용 또는 형식 artifact인지 사용자 확인이 필요하다.

### pretrained PLM 경로

- 실제 source: upstream `viewport_prediction/config.py::Config.plms_dir`
- VP cwd에서 실제 해석: `/workspace/NetLLM-source/downloaded_plms`
- 프로젝트 `README.md` 설명: `/workspace/downloaded_plms/gpt2/base`
- 내용: wrapper 문서의 절대경로가 실제 code resolution과 다르다.

### image feature 경로

- upstream `viewport_prediction/README.md`: `data/image_features`
- 실제 source `viewport_prediction/config.py::Config.dataset_image_features`: `data/images/Jin2022images/features`, `data/images/Wu2017images/features`
- 내용: 문서와 current code의 directory layout이 다르다.

### preprocessing/extraction 경로

- upstream `viewport_prediction/dataset/preprocess.py`: 개발자 machine의 `/data2/wuduo/...` 절대경로를 하드코딩한다.
- upstream `viewport_prediction/dataset/extract_features.py`: `source_dir`, `target_dir` placeholder와 별도 `/data/data1/...` 값을 혼용한다.
- `store_feature()`는 계산한 `target_dir` 변수를 사용하지 않고 문자열 `target_dir/...`에 저장하도록 작성되어 있다.
- 따라서 raw preprocessing과 feature extraction은 현재 repository/config만으로 재현할 수 없다.

## 5. NetLLM 논문과 current code 비교

### future answer generation

- 논문: NetLLM section 4.2는 networking head가 task answer를 single inference로 직접 생성한다고 설명한다.
- source: upstream `viewport_prediction/models/pipeline.py::Pipeline.auto_regressive`는 미래 좌표 한 시점마다 PLM 전체 forward를 반복해 기본 `F=20`회 호출한다.
- 결론: continuous coordinate head를 사용한다는 점은 일치하지만, current VP autoregressive inference의 “전체 future answer가 single inference”라고 볼 수 없다. 구현 수정 없이 충돌로 유지한다.

### ViT 사용 방식

- 논문: NetLLM section 4.1 및 Appendix A.2는 VP multimodal encoder가 ViT로 image를 encode한다고 설명한다.
- source: current `Pipeline`은 online ViT가 아니라 offline `.pth` CLS feature 하나를 load한다. `get_multimodal_information()`에는 on-the-fly extraction이 TODO로 남아 있다.
- 결론: offline frozen-ViT feature도 논문 개념과 양립 가능하지만, current runtime에 raw image/patch pipeline은 없다.

### output과 autoregression

- 논문 Appendix A.2: VP head는 Roll, Pitch, Yaw 좌표를 출력한다.
- source `NetworkingHead`: `Linear(hidden_size,3)+Tanh`, normalized continuous coordinate triple을 출력한다.
- 결론: 일치한다. 일반 vocabulary token generation은 current VP path에 없다.

## 6. LiteVLM 원형과 직접 호환되지 않는 부분

| LiteVLM 원형 | current VP 상태 | 결론 |
|---|---|---|
| Patch Selection: text query로 multi-camera view/patch를 ViT 전에 선택 | text query 없음, online ViT 없음, precomputed CLS vector 한 개 | 직접 적용 불가 |
| Token Selection: fine-tuned LLM 첫 decoder layer를 standalone module로 사용해 visual token을 scoring/pruning | temporal token-like embedding H개와 optional visual embedding 1개 | temporal selection은 별도 VP 변형이며 논문 원형이 아님 |
| 중요도 label: VLM self-attention + nuScenes bbox critical object 보강 | bbox/critical object annotation 없음 | 동일 학습 신호 구성 불가 |
| Speculative Decoding: draft가 discrete token sequence 제안, main LLM이 KV cache로 일괄 검증 | continuous `[B,1,3]` 좌표, LM head 미사용, cache 전달 없음 | 직접 호환 불가 |

## 7. current source 내부의 추가 불확실성

- `Pipeline` temporal embedding의 `.view(1,256)`와 multimodal metadata의 `.item()`은 batch size 1을 전제한다.
- `run_plm.py::adapt`의 optimizer group에는 `pipeline.conv1d`가 어느 branch에도 포함되지 않는다. 또한 `--freeze-plm` branch는 주석과 달리 `pipeline.plm.networking_head`도 명시적으로 포함하지 않는다. 논문 section 4와 Appendix A.2의 trainable multimodal encoder/networking head 설명에 비추어 실제 update 대상 확인이 필요하다.
- `run_plm.py::adapt`와 `test`의 default best-model `file_prefix`가 다르다. test 쪽에만 `_axes_`가 있어 자동 checkpoint 탐색이 맞지 않을 가능성이 있다.
- `run_plm.py`의 `--compile` branch는 정의되지 않은 `prompt_model`을 참조한다.
- PLM result filename은 `result_` prefix가 없어 `ResultNotebook.write()`가 detail path를 분리하지 못하고, `test()`는 같은 `result_path`에 `write_detail()`도 호출한다.
- `VelocityMethod`는 `x[0]`만 사용하므로 batch size가 1보다 큰 경우 batch 전체를 처리하지 않는다.
- TRACK train/test는 normalized input을 사용하지만 loss/record ground truth는 raw degree tensor를 사용한다. output scale 의도는 실행 전 확인이 필요하다.
- `pack_data(for_track=True)`에서 첫 image가 없으면 `pre_image=None`이 저장될 수 있다. 현재 image data가 전부 없어 TRACK path는 그 전에 blocked된다.

위 항목은 원본 수정 없이 Phase 1/후속 설계의 blocker 또는 검증 항목으로 유지한다.
