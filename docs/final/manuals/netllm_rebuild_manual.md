# NetLLM × LiteVLM 재현 세팅 매뉴얼

## 0. 현재 상태 요약

현재 작업은 다음 단계까지 완료된 상태로 정리하면 된다.

- Llama2 checkpoint-era source를 찾아 strict load 성공
- 실제 Jin2022 VP sample technical smoke 성공
- Llama selector identity equivalence 성공
- full benchmark 완료: 1,698 samples × 6 configurations
- MAE/RMSE/Loss/Latency/GPU memory 시각화 완료
- Continuous VP Draft-and-Verify speculative prototype smoke 성공
- training/fine-tuning은 수행하지 않음
- 결과는 `paper reproduction`이 아니라 `recovered-artifact controlled comparison`이다.

## 1. 인스턴스 폐기 전 반드시 할 일

### 1.1 Git에 올릴 것과 올리면 안 되는 것

Git에 올릴 수 있는 것:

- `src/`
- `scripts/`
- `tests/`
- `docs/`
- `configs/`
- `manifests/`
- 작은 CSV, JSON, figure, 결과 요약 파일

Git에 올리면 안 되는 것:

- `try_llama2_7b.zip`
- `data.zip`
- `/root/NetLLM-assets/llama/base`
- checkpoint 원본
- dataset 원본
- Hugging Face token
- `.cache`, `__pycache__`, `.pyc`

권장 확인 명령:

```bash
cd /root/NetLLM

git status --short
git diff --stat

git status --short | grep -E 'zip|safetensors|\.bin|\.pth|NetLLM-assets|data\.zip|try_llama' || true
```

외부 asset이 stage되어 있으면 반드시 제외한다.

```bash
git restore --staged try_llama2_7b.zip data.zip 2>/dev/null || true
printf '\ntry_llama2_7b.zip\ndata.zip\nNetLLM-assets/\n*.safetensors\n*.bin\n*.pth\n' >> .git/info/exclude
```

### 1.2 결과 백업

가장 중요한 결과 위치:

```text
/root/NetLLM/docs/experiment_phase/llama/benchmark/
/root/NetLLM/docs/experiment_phase/speculative/
/root/NetLLM/docs/implementation/
/root/NetLLM/experiments/vp/llama_benchmark/full/
/root/NetLLM/experiments/vp/llama_benchmark/figures/
/root/NetLLM/experiments/vp/llama_speculative_smoke/
/root/NetLLM/manifests/llama/
```

가능하면 Git commit/push로 저장한다.

```bash
cd /root/NetLLM
git add src scripts tests docs configs manifests experiments/vp/llama_benchmark/figures experiments/vp/llama_benchmark/full/benchmark_summary.csv experiments/vp/llama_benchmark/full/benchmark_summary.json experiments/vp/llama_speculative_smoke
git commit -m "Add NetLLM Llama selector benchmark and speculative prototype results"
git push
```

대용량 per-sample CSV까지 Git에 올릴지는 저장소 정책에 따라 판단한다. 불안하면 별도 압축 파일로 다운로드한다.

```bash
cd /root

tar -czf netllm_results_backup_$(date -u +%Y%m%d).tgz \
  NetLLM/docs \
  NetLLM/manifests \
  NetLLM/experiments/vp/llama_benchmark \
  NetLLM/experiments/vp/llama_selector_equivalence \
  NetLLM/experiments/vp/llama_speculative_smoke
```

### 1.3 외부 asset 보존

Llama base는 Hugging Face에서 다시 받을 수 있다. 반드시 보존해야 하는 것은 팀이 준 zip이다.

```text
try_llama2_7b.zip
data.zip
```

이 두 파일이 로컬 PC 또는 Google Drive에 있는지 확인한다. 없다면 폐기 전에 다운로드한다.

## 2. 새 인스턴스에서 재세팅하는 표준 절차

### 2.1 권장 인스턴스

- GPU: RTX 4090 24GB 이상
- Disk: 최소 100GB, 권장 150GB 이상
- Network: Hugging Face Llama download 가능
- Template: CUDA 12.1 계열 PyTorch 또는 일반 CUDA/Jupyter 환경

### 2.2 기본 순서

```bash
cd /root

git clone https://github.com/soyuniverse/NetLLM.git
cd /root/NetLLM
chmod +x setup_netllm_repro_master.sh
```

팀 zip 파일을 새 인스턴스에 올린다.

```text
/root/NetLLM/try_llama2_7b.zip
/root/NetLLM/data.zip
```

Hugging Face 로그인 확인:

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
hf auth login
hf auth whoami
```

그 다음 master script 실행:

```bash
cd /root/NetLLM
./setup_netllm_repro_master.sh all
```

완료 후 검증:

```bash
./setup_netllm_repro_master.sh verify
```

기존 smoke runner가 repo에 있으면:

```bash
./setup_netllm_repro_master.sh smoke
```

## 3. 실패 시 빠른 진단

### Hugging Face offline 오류

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
hf auth whoami
```

### GPU 미인식

```bash
nvidia-smi
/root/venvs/vp_netllm_llama/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### checkpoint path 오류

```bash
find /root/NetLLM-assets/checkpoints/try_llama2_7b -maxdepth 2 -type f | sort | head -50
```

### dataset path 오류

```bash
find /root/NetLLM-assets/datasets/team_data -maxdepth 3 -type f | wc -l
find /root/NetLLM-source/viewport_prediction/data/viewports/Jin2022 -name '*.csv' | wc -l
```

## 4. 재현 후 바로 하면 좋은 작업

1. strict load 재검증
2. single sample technical smoke 재검증
3. Identity equivalence 재검증
4. full benchmark는 필요할 때만 재실행
5. 우선은 이미 생성된 CSV/figures를 발표자료에 사용

## 5. 발표에서 써야 할 표현

사용해도 되는 표현:

```text
recovered-artifact controlled comparison
동일 checkpoint 기준 selector 비교
strict load 및 VP technical smoke 검증 완료
full benchmark 1,698 samples × 6 configurations 완료
```

피해야 할 표현:

```text
공식 NetLLM 논문 수치 재현 완료
Llama2 training 재현 완료
Speculative decoding speedup 달성
```
