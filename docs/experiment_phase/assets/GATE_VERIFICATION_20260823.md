# Gate-A/B Verification — 2026-08-23 (new instance)

Fourth physical instance this project has run on. Per this session's
Task 0 instruction, checked for an existing pass record on *this*
instance before trusting any GPU number; none existed (`/root/NetLLM-assets`,
`/root/llama2-7b-base`, `/root/NetLLM-source` were all absent at session
start — same pattern as the prior three asset losses documented in
`ASSET_RECOVERY_VERIFICATION*.md` and `GATE_A_VERIFICATION.md`). Ran the
standard procedure end to end.

## Step 1 — staging zip checksums vs. `BACKUP_MANIFEST.md`: MATCH

The two staging zips were present at `/root/NetLLM-assets/staging/`
(`data.zip`, `try_llama2_7b.zip`) at session start — no re-upload
needed this time.

```
9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d  data.zip
57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53  try_llama2_7b.zip
```

Both identical to `BACKUP_MANIFEST.md`'s reference values.

## Step 2 — extraction + placement

`try_llama2_7b.zip` reproduced the same double-nesting as every prior
extraction (`try_llama2_7b/try_llama2_7b/{...}`), de-nested into
`/root/NetLLM-assets/checkpoints/try_llama2_7b/`. File sizes match the
recorded reference exactly: `adapter_config.json` 542 B,
`adapter_model.bin` 67,155,338 B, `modules_except_plm.bin` 16,900,050 B,
`README.md` 5,479 B.

`data.zip` extracted to `/root/NetLLM-source/viewport_prediction/`,
producing `data/{viewports,images,ft_plms,models,results}/` with
`data/viewports/Jin2022/` containing 27 video directories.

Base Llama2-7b weights were not present on this instance and were
re-pulled via `hf download meta-llama/Llama-2-7b-hf --local-dir
/root/llama2-7b-base` (both `model-0000{1,2}-of-00002.safetensors`
landed, ~13 GiB; the CLI also pulled duplicate `pytorch_model*.bin`
shards, which were deleted afterward as redundant — only the
safetensors path is used by `checkpoint_era_runtime.py`'s
`DEFAULT_BASE_MODEL_PATH`).

**Environment note**: this instance's Python environment (`/opt/conda`,
not a dedicated venv) had `transformers`/`peft`/`accelerate`/`cv2`/numpy
all missing or mismatched at session start (same failure mode as the
prior "third asset loss" session, `CLAUDE.md` current-state section).
Reinstalled to the pinned versions from `requirements-vp.txt`:
`transformers==4.34.1`, `peft==0.6.2`, `accelerate==0.24.1`,
`opencv-python-headless==4.8.1.78`, `numpy==1.24.4` (torch stayed at
the pre-existing 2.2.0+cu121). `pip check` clean after.

## Step 3 — full model strict load: PASS

`checkpoint_strict_load_20260823.json` (same procedure as
`scripts/experiment_phase/assets/verify_checkpoint_strict_load.py`,
run with `NETLLM_PROJECT_ROOT` override so it could execute from the
session scratchpad without clobbering the git-tracked
`checkpoint_strict_load.json`):

```json
{
  "checkpoint_loaded": true,
  "load_seconds": 3.25,
  "adapter_missing_count": 0,
  "adapter_unexpected_count": 0,
  "adapter_value_mismatch_count": 0,
  "non_plm_missing_count": 0,
  "non_plm_unexpected_count": 0,
  "strict_load_pass": true
}
```

## Step 4 — 50-sample baseline MAE reproduction: MATCH

`run_speculative_benchmark.py --checkpoint-path
/root/NetLLM-assets/checkpoints/try_llama2_7b --dataset-path
/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022
--thresholds 0.35 --gammas 8 --num-samples 50 --device cuda:0`:

| config | MAE | latency median |
|---|---:|---:|
| baseline (this instance, 2026-08-23) | **11.036768085417648** | 467.17 ms |
| threshold=0.35_gamma=8 (this instance) | 11.031369 | 96.30 ms |
| reference (2026-08-02 origin) | 11.036768 | 571.7 ms |
| 2026-08-09 instance | 11.036768085417648 | 680.3 ms |

MAE matches the reference to the full 15 significant figures recorded
this run — identical, not just within fp16 noise. Full output:
`results/speculative/20260823T063906Z/{results.csv,summary.json,summary.md}`.

Latency is the fastest of the three instances so far (467 ms baseline
vs. 571–680 ms previously) — expected per-instance GPU/host variance,
not a code change. Per `NEW_INSTANCE_CALIBRATION.md`'s established
rule, this instance's latency numbers are not compared against prior
instances' absolute values anywhere in this session's deliverables.

## Result: Gate-A + Gate-B COMPLETE (this instance, 2026-08-23)

All steps pass. This instance's checkpoint + dataset + base-model
assembly is verified trustworthy for GPU experiments this session.
Task 3 (adaptive-K selector) is unblocked. See
`NEW_INSTANCE_CALIBRATION_20260823.md` for this instance's own 200-sample
A/D latency baseline (required before any adaptive-K latency claim, per
this session's instruction not to mix latency numbers across instances).
