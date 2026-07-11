# VP PLM dependency compatibility

## 1. 조사 범위

- 현재 환경: `/venv/vp_netllm` — 읽기 전용으로 보존
- 격리 test 환경: `/venv/vp_netllm_plmtest`
- upstream VP: `/workspace/NetLLM-source/viewport_prediction`
- 검사 범위: package metadata, `pip check`, Python import
- 제외 범위: model download/load, forward, training, adaptation, checkpoint, source patch 적용

## 2. 최초 오류 원인

최초 조합은 다음과 같다.

```text
torch==2.1.0+cu118
transformers==4.34.1
peft==0.6.2
accelerate==0.32.1
huggingface-hub==0.17.3
safetensors==0.5.3
tokenizers==0.14.1
```

`accelerate/accelerator.py:34`가 import 시 다음 symbol을 요구한다.

```python
from huggingface_hub import split_torch_state_dict_into_shards
```

하지만 `huggingface-hub==0.17.3`의 public namespace에는 해당 symbol이 없다. 그 결과 `accelerate`가 직접 실패하고, `peft`는 `peft.utils.other → accelerate` chain에서 실패하며, `run_plm`은 `transformers.generation.utils → accelerate` chain에서 실패한다.

설치 metadata가 `accelerate`에 필요한 Hub 최소 version을 명시하지 않아 최초 환경의 `pip check`는 성공하지만 runtime import는 실패한다.

## 3. 선언 dependency 관계

| package | version | 관련 설치 metadata |
|---|---:|---|
| `torch` | 2.1.0+cu118 | 기준 runtime, 변경하지 않음 |
| `transformers` | 4.34.1 | `huggingface-hub>=0.16.4,<1.0`, `tokenizers>=0.14,<0.15`, `safetensors>=0.3.1` |
| `peft` | 0.6.2 | `torch>=1.13.0`, `transformers`, `accelerate>=0.21.0`, `safetensors` |
| `accelerate` | 0.32.1 | `torch>=1.10.0`, `huggingface-hub`, `safetensors>=0.3.1`; Hub 최소 version 미명시 |
| `huggingface-hub` | 0.17.3 | 최초 symbol provider이지만 필요한 symbol은 없음 |
| `tokenizers` | 0.14.1 | `huggingface_hub>=0.16.4,<0.18` |
| `safetensors` | 0.5.3 | 변경하지 않음 |

핵심 constraint는 `tokenizers==0.14.1`의 Hub `<0.18`이다. 필요한 symbol을 제공하는 Hub 0.21.0을 사용하면 import는 해결되지만 declared dependency가 깨진다.

## 4. test environment 생성

먼저 다음 clone을 시도했다.

```bash
/opt/miniforge3/condabin/conda create \
  --prefix /venv/vp_netllm_plmtest \
  --clone /venv/vp_netllm --yes
```

원본 환경의 pip-installed package들을 conda channel에서 `pypi_0` build로 찾으려 하면서 `PackagesNotFoundInChannelsError`가 발생했다. 실패한 conda clone은 target directory를 남기지 않았고 원본 환경도 변경하지 않았다.

대신 다음과 같이 격리 overlay venv를 생성했다.

```bash
/venv/vp_netllm/bin/python -m venv \
  --copies --system-site-packages \
  /venv/vp_netllm_plmtest
```

target의 `sys.prefix`는 `/venv/vp_netllm_plmtest`, `sys.base_prefix`는 `/venv/vp_netllm`이다. 원본 package를 읽기 기반으로 사용하고 변경 package는 target `site-packages`에만 설치된다. 실제 pip output도 원본 package를 `outside environment`라며 제거하지 않았음을 확인했다.

## 5. 검토한 후보

### 후보 1: Hub만 0.21.0으로 변경 — 기각

```bash
/venv/vp_netllm_plmtest/bin/python -m pip install \
  'huggingface-hub==0.21.0'
```

결과:

- `huggingface_hub` import: 성공
- `accelerate==0.32.1` import: 성공
- `peft==0.6.2` import: 성공
- `pip check`: 실패

```text
tokenizers 0.14.1 has requirement huggingface_hub<0.18,>=0.16.4,
but you have huggingface-hub 0.21.0 which is incompatible.
```

Hub 하나만 바꾸면 runtime symbol은 제공되지만 Transformers 4.34.1이 사용하는 Tokenizers 0.14.1의 declared constraint를 위반한다. 따라서 재현 가능한 최소 조합으로 채택하지 않았다.

### 후보 2: Hub 복원 + Accelerate 0.24.1 — 성공

```bash
/venv/vp_netllm_plmtest/bin/python -m pip install \
  'huggingface-hub==0.17.3' \
  'accelerate==0.24.1'
```

최종적으로 최초 조합에서 달라진 package는 `accelerate` 하나다.

```text
accelerate: 0.32.1 → 0.24.1
```

검증 결과:

- `pip check`: `No broken requirements found.`
- `torch`, `transformers`, `peft`, `accelerate`, `huggingface_hub`, `cv2`, `yacs`: 모두 import 성공
- upstream `run_plm`: import 성공
- upstream `run_baseline`: import 성공
- model load/forward/training: 실행하지 않음

`accelerate==0.24.1`은 `peft==0.6.2`의 `accelerate>=0.21.0` 조건과 `transformers==4.34.1` optional 조건 `accelerate>=0.20.3`을 모두 만족한다. 또한 Hub 0.17.3에서 존재하지 않는 shard split symbol을 import하지 않는다.

PEFT version 변경 후보는 검토할 필요가 없었다.

## 6. 성공한 최소 조합

```text
torch==2.1.0+cu118
transformers==4.34.1
peft==0.6.2
accelerate==0.24.1
huggingface-hub==0.17.3
safetensors==0.5.3
tokenizers==0.14.1
```

선택 근거:

1. 실질적으로 한 package만 변경한다.
2. 기존 Torch, Transformers, PEFT, Hub, Tokenizers를 유지한다.
3. 모든 declared constraint를 만족한다.
4. 목표 import 세 개와 regression import가 모두 성공한다.

## 7. 향후 재현 방법

새 환경을 구성할 때 기존 `requirements-vp.txt` 중 다음 한 줄만 바꾸는 것이 검증된 최소 반영안이다.

```diff
-accelerate==0.32.1
+accelerate==0.24.1
```

`setup.sh`는 이미 `requirements-vp.txt`를 설치하므로 version 수정 외 별도 setup logic 변경은 필요하지 않다. 제안 patch는 `patches/setup_plm_dependency_fix.diff`에만 작성했으며 실제 적용하지 않았다.

검증 명령:

```bash
cd /workspace/NetLLM
bash scripts/check_phase1_5_plm_env.sh
```

