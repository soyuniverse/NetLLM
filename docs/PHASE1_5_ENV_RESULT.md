# Phase 1.5A 결과: PLM dependency compatibility

## 1. 결과 요약

- 결과: **성공**
- 현재 환경 `/venv/vp_netllm`: 변경하지 않음
- test environment: `/venv/vp_netllm_plmtest`
- 실질적인 최소 변경: `accelerate==0.32.1 → 0.24.1`
- `peft`, `accelerate`, upstream `run_plm`: import 성공
- upstream `run_baseline`: import 성공
- `pip check`: 성공
- model/dataset 다운로드 및 model 실행: 없음

## 2. 현재 환경 기록

### Python과 주요 version

```text
Python 3.8.10
pip 25.0.1
torch==2.1.0+cu118
transformers==4.34.1
peft==0.6.2
accelerate==0.32.1
huggingface-hub==0.17.3
safetensors==0.5.3
tokenizers==0.14.1
```

현재 환경의 `pip check` 결과는 다음과 같다.

```text
No broken requirements found.
```

작업 전후 `pip freeze | sha256sum`은 모두 다음 값으로 동일했다.

```text
c3717faa56495a1b8e38cac864f1151aa40bdfa25e85c38842d52bb469a47631
```

그러나 실제 import traceback은 다음과 같다.

```text
peft:
  peft/utils/other.py imports accelerate
  accelerate/accelerator.py:34 imports
  huggingface_hub.split_torch_state_dict_into_shards
ImportError: cannot import name 'split_torch_state_dict_into_shards'
from 'huggingface_hub' (.../huggingface_hub/__init__.py)

accelerate:
  accelerate/__init__.py imports Accelerator
  accelerate/accelerator.py:34 imports
  huggingface_hub.split_torch_state_dict_into_shards
ImportError: cannot import name 'split_torch_state_dict_into_shards'
from 'huggingface_hub' (.../huggingface_hub/__init__.py)

run_plm:
  run_plm.py imports utils.plms_utils
  transformers.modeling_utils imports transformers.generation
  transformers.generation.utils imports accelerate.hooks
RuntimeError: Failed to import transformers.generation.utils because:
cannot import name 'split_torch_state_dict_into_shards'
from 'huggingface_hub'
```

### 현재 환경 `pip freeze`

```text
accelerate==0.32.1
certifi==2026.6.17
charset-normalizer==3.4.9
contourpy==1.1.1
cycler==0.12.1
einops==0.8.1
filelock==3.16.1
fonttools==4.57.0
fsspec==2025.3.0
huggingface-hub==0.17.3
idna==3.15
importlib_resources==6.4.5
Jinja2==3.1.6
joblib==1.4.2
kiwisolver==1.4.7
MarkupSafe==2.1.5
matplotlib==3.7.5
mpmath==1.3.0
munch==4.0.0
networkx==3.1
numpy==1.24.4
opencv-python-headless==4.8.1.78
packaging==26.2
pandas==2.0.3
peft==0.6.2
pillow==10.4.0
prettytable==3.11.0
psutil==7.2.2
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2026.2
PyYAML==6.0.3
regex==2024.11.6
requests==2.32.4
safetensors==0.5.3
scikit-learn==1.3.2
scipy==1.10.1
six==1.17.0
sympy==1.13.3
threadpoolctl==3.5.0
tokenizers==0.14.1
torch==2.1.0+cu118
tqdm==4.68.4
transformers==4.34.1
triton==2.1.0
typing_extensions==4.12.2
tzdata==2026.3
urllib3==2.2.3
wcwidth==0.8.2
yacs==0.1.8
zipp==3.20.2
```

## 3. test environment

- path: `/venv/vp_netllm_plmtest`
- Python: 3.8.10
- 방식: `/venv/vp_netllm`을 base로 하는 `venv --copies --system-site-packages` overlay
- 생성 전 disk 여유: 약 90GB
- 생성 전 target path: 없음

`conda create --clone`은 pip package의 `pypi_0` build를 configured conda channel에서 찾지 못해 실패했으며 target path를 남기지 않았다. 이후 overlay venv를 생성했다. target에 설치한 package는 원본 환경 밖의 package를 uninstall하지 않았다는 pip output을 확인했다.

## 4. 적용한 변경과 후보 결과

| 후보 | 변경 | import | `pip check` | 판정 |
|---|---|---|---|---|
| 1 | Hub 0.17.3 → 0.21.0 | `peft`, `accelerate` 성공 | Tokenizers constraint 위반 | 기각 |
| 2 | Hub 0.17.3 복원, Accelerate 0.32.1 → 0.24.1 | 모두 성공 | 성공 | 채택 |

최종 test environment 조합:

```text
torch==2.1.0+cu118
transformers==4.34.1
peft==0.6.2
accelerate==0.24.1
huggingface-hub==0.17.3
safetensors==0.5.3
tokenizers==0.14.1
```

## 5. import smoke test

| import | 결과 |
|---|---|
| `torch` | OK |
| `transformers` | OK |
| `peft` | OK |
| `accelerate` | OK |
| `huggingface_hub` | OK |
| `cv2` | OK |
| `yacs` | OK |
| upstream `run_baseline` | OK |
| upstream `run_plm` | OK |

추가 확인:

```text
torch CUDA runtime: 11.8
torch.cuda.is_available(): True
```

model load, forward, training, adaptation은 실행하지 않았다.

## 6. `pip check`

```text
No broken requirements found.
```

## 7. upstream Git 무결성

| 항목 | 작업 전 | 작업 후 |
|---|---|---|
| commit | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54` | 동일 |
| status porcelain | 빈 결과 | 빈 결과 |
| diff name-status | 빈 결과 | 빈 결과 |
| source patch | 없음 | 없음 |
| source `__pycache__` | 없음 | 없음 |

## 8. 생성한 산출물

- `/venv/vp_netllm_plmtest`
- `scripts/check_phase1_5_plm_env.sh`
- `docs/PHASE1_5_ENV_RESULT.md`
- `docs/PLM_DEPENDENCY_COMPATIBILITY.md`
- `constraints-vp-plm-proposed.txt`
- `patches/setup_plm_dependency_fix.diff`

기존 `setup.sh`, `requirements-vp.txt`에는 변경을 적용하지 않았다.

## 9. 남은 blocker와 다음 단계

PLM import dependency blocker는 격리 환경에서 해결됐다. 다음 단계에서 GPT-2를 실제로 load하려면 pretrained PLM이 여전히 필요하지만 이번 단계에서는 다운로드하지 않았다.

Phase 1.5B 또는 model load로 진행하려면 사용자 검토와 별도 승인이 필요하다. 이번 단계에서는 더 진행하지 않는다.

## Appendix A. 최초 traceback 전문

### `import accelerate`

```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/venv/vp_netllm/lib/python3.8/site-packages/accelerate/__init__.py", line 16, in <module>
    from .accelerator import Accelerator
  File "/venv/vp_netllm/lib/python3.8/site-packages/accelerate/accelerator.py", line 34, in <module>
    from huggingface_hub import split_torch_state_dict_into_shards
ImportError: cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub' (/venv/vp_netllm/lib/python3.8/site-packages/huggingface_hub/__init__.py)
```

### `import peft`

```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/venv/vp_netllm/lib/python3.8/site-packages/peft/__init__.py", line 22, in <module>
    from .auto import (
  File "/venv/vp_netllm/lib/python3.8/site-packages/peft/auto.py", line 30, in <module>
    from .config import PeftConfig
  File "/venv/vp_netllm/lib/python3.8/site-packages/peft/config.py", line 24, in <module>
    from .utils import CONFIG_NAME, PeftType, TaskType
  File "/venv/vp_netllm/lib/python3.8/site-packages/peft/utils/__init__.py", line 22, in <module>
    from .other import (
  File "/venv/vp_netllm/lib/python3.8/site-packages/peft/utils/other.py", line 20, in <module>
    import accelerate
  File "/venv/vp_netllm/lib/python3.8/site-packages/accelerate/__init__.py", line 16, in <module>
    from .accelerator import Accelerator
  File "/venv/vp_netllm/lib/python3.8/site-packages/accelerate/accelerator.py", line 34, in <module>
    from huggingface_hub import split_torch_state_dict_into_shards
ImportError: cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub' (/venv/vp_netllm/lib/python3.8/site-packages/huggingface_hub/__init__.py)
```

### upstream `import run_plm`

```text
Traceback (most recent call last):
  File "/venv/vp_netllm/lib/python3.8/site-packages/transformers/utils/import_utils.py", line 1282, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/venv/vp_netllm/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/venv/vp_netllm/lib/python3.8/site-packages/transformers/generation/utils.py", line 84, in <module>
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
  File "/venv/vp_netllm/lib/python3.8/site-packages/accelerate/__init__.py", line 16, in <module>
    from .accelerator import Accelerator
  File "/venv/vp_netllm/lib/python3.8/site-packages/accelerate/accelerator.py", line 34, in <module>
    from huggingface_hub import split_torch_state_dict_into_shards
ImportError: cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub' (/venv/vp_netllm/lib/python3.8/site-packages/huggingface_hub/__init__.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/workspace/NetLLM-source/viewport_prediction/run_plm.py", line 14, in <module>
    from utils.plms_utils import load_plm
  File "/workspace/NetLLM-source/viewport_prediction/utils/plms_utils.py", line 11, in <module>
    from transformers.modeling_utils import PreTrainedModel
  File "/venv/vp_netllm/lib/python3.8/site-packages/transformers/modeling_utils.py", line 39, in <module>
    from .generation import GenerationConfig, GenerationMixin
  File "/venv/vp_netllm/lib/python3.8/site-packages/transformers/utils/import_utils.py", line 1284, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub' (/venv/vp_netllm/lib/python3.8/site-packages/huggingface_hub/__init__.py)
```
