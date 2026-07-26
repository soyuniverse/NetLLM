# Llama Inference Environment Plan V2

## Isolation

- Target: `/root/venvs/vp_netllm_llama`
- Creator: Conda at `/opt/conda/bin/conda`
- Existing target before creation: absent
- Existing GPT-2 environment: out of scope and unchanged
- Purpose: inference-only checkpoint-era `run_old.py` compatibility

## Version comparison and decision

| Component | Checkpoint-era evidence | Requested/project inference pin | Selected |
| --- | --- | --- | --- |
| Python | 3.8.10 | 3.8.10 | 3.8.10 |
| PyTorch | 2.1.0, CUDA build unspecified | 2.2.0+cu121 | 2.2.0+cu121 |
| NumPy | 1.24.4 | 1.24.4 | 1.24.4 |
| Transformers | 4.34.1 | 4.34.1 | 4.34.1 |
| PEFT | README 0.6.2; adapter card 0.6.0 | 0.6.2 | 0.6.2 |

PyTorch 2.2.0+cu121 is selected deliberately, not merely because it is newer:
it is the explicit project constraint for this isolated Llama environment,
matches the requested Gate 5 target, is supported by the installed NVIDIA
560.35.03 driver, and does not change checkpoint serialization or tensor
shapes. The source's torch 2.1.0 remains a provenance difference and will be
reported. PEFT 0.6.2 follows the source README and project constraint; adapter
load compatibility with its PEFT 0.6.0 model-card provenance must be verified
by strict smoke.

## Exact package plan

### CUDA/PyTorch index

- `torch==2.2.0+cu121`
- `torchvision==0.17.0+cu121`
- `torchaudio==2.2.0+cu121`

### Runtime and checkpoint stack

- `numpy==1.24.4`
- `munch==4.0.0`
- `transformers==4.34.1`
- `peft==0.6.2`
- `accelerate==0.24.1`
- `huggingface-hub==0.17.3`
- `tokenizers==0.14.1`
- `safetensors==0.5.3`
- `sentencepiece==0.1.99`
- `protobuf==4.25.3`

### Dataset/import dependencies

- `scipy==1.10.1`
- `pandas==2.0.3`
- `scikit-learn==1.3.2`
- `einops==0.8.1`
- `yacs==0.1.8`
- `opencv-python-headless==4.8.1.78`
- `prettytable==3.11.0`
- `matplotlib==3.7.5`

The environment will be accepted only if `pip check`, CUDA detection, PEFT
import, current Llama import, and checkpoint-era Llama import all succeed.
Training, optimizer construction, and backward execution are excluded.
