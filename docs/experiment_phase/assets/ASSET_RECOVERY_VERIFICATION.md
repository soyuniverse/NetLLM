# Asset Recovery Verification

VP fine-tuned checkpoint and Jin2022 dataset were re-uploaded to
`/root/NetLLM-assets/staging/{try_llama2_7b.zip,data.zip}` on 2026-08-02.
This records their placement and re-verification before any benchmark
number produced this session is treated as trustworthy.

## Pre-extraction findings (reported before acting, per instruction)

1. **`docs/RUNBOOK_ASSETS_ARRIVAL.md` does not exist.** Searched the full
   working tree and git history (`git log --all --diff-filter=A
   --name-only`); it was never committed. Instead of guessing, the
   expected paths were cross-checked against the paths already hardcoded,
   consistently, across multiple previously-verified scripts:
   - checkpoint: `/root/NetLLM-assets/checkpoints/try_llama2_7b`
     (`run_llama_vp_technical_smoke.py`, `run_llama_selector_benchmark.py`,
     `run_llama_continuous_speculative_smoke.py`,
     `setup_netllm_llama.sh`, `LLAMA_CHECKPOINT_CLASSIFICATION.md`, ...)
   - dataset: `/root/NetLLM-source/viewport_prediction/data/viewports/Jin2022`
     (`run_llama_vp_technical_smoke.py`, `run_llama_selector_benchmark.py`,
     `run_llama_continuous_speculative_smoke.py`)

     This does **not** match `NetLLM-assets/datasets/`, which is what was
     initially proposed this session. Flagged and the user chose the
     existing-script path (`/root/NetLLM-source/...`) — see decision below.
2. **`try_llama2_7b.zip` is double-nested**:
   `try_llama2_7b/try_llama2_7b/{adapter_config.json, adapter_model.bin,
   modules_except_plm.bin, README.md}` — one level deeper than the
   canonical `.../checkpoints/try_llama2_7b/{files}` layout. Corrected
   during extraction (see below), not treated as a content problem — the
   README inside is standard boilerplate consistent with a normal PEFT
   adapter save.
3. **`data.zip`** unzips cleanly to `data/{viewports,images,ft_plms,
   models,results}/`, matching `config.py`'s expected layout exactly.
   `data/viewports/Jin2022/video{1..27}/5Hz/simple_5Hz_user*.csv` is
   present (27 videos, matching `cfg.dataset_video_split['Jin2022']`'s
   train+valid+test = 15+6+6). No `llama` content inside `data.zip` —
   the checkpoint lives exclusively in `try_llama2_7b.zip`, no overlap.

## Decision (user, this session)

Dataset extracted to the existing-script convention,
`/root/NetLLM-source/viewport_prediction/data/...`, not
`NetLLM-assets/datasets/`.

## Placement

- `try_llama2_7b.zip` extracted to a temp dir, then the de-nested files
  moved to `/root/NetLLM-assets/checkpoints/try_llama2_7b/`:
  `adapter_config.json` (542B), `adapter_model.bin` (67,155,338B),
  `modules_except_plm.bin` (16,900,050B), `README.md` (5,479B).
- `data.zip` extracted directly to
  `/root/NetLLM-source/viewport_prediction/` (produces
  `.../viewport_prediction/data/...`).
- Both staging zips left untouched at
  `/root/NetLLM-assets/staging/{try_llama2_7b.zip,data.zip}` as a
  re-extraction backup, per instruction.

## Checkpoint strict-load re-verification

Script: `scripts/experiment_phase/assets/verify_checkpoint_strict_load.py`
(replicates the exact procedure
`run_llama_vp_technical_smoke.py` used the first time this checkpoint was
strict-loaded, since no RUNBOOK exists to follow instead: adapter
missing/unexpected/value-mismatch keys via `get_peft_model_state_dict`,
plus non-PLM module missing/unexpected keys via a strict `state_dict`
load — two independent checks, not just one).

Raw output: `experiments/vp/asset_recovery/checkpoint_strict_load.json`.

| check | result |
|---|---:|
| adapter missing keys | 0 |
| adapter unexpected keys | 0 |
| adapter value mismatches (vs. raw `adapter_model.bin`) | 0 |
| non-PLM missing keys | 0 |
| non-PLM unexpected keys | 0 |
| **strict_load_pass** | **true** |

Matches the original `missing/unexpected key 0/0` result recorded in
`claude.md` before the asset loss.

## Dataset verification

`create_dataset("Jin2022", his_window=10, fut_window=20, trim_head=30,
trim_tail=60, frequency=5, step=15, include=["test"])` (exact parameters
`run_llama_selector_benchmark.py` used to produce
`docs/experiment_phase/llama/benchmark/LLAMA_SELECTOR_FULL_BENCHMARK_RESULT.md`,
the "7.26 report"):

- test split length: **1,698** — matches the 7.26 report's "sample count:
  configuration별 1,698" and total measured inference count 10,188
  (= 1,698 × 6 configurations) exactly.
- sample 0 shapes: history `(10, 3)`, future `(20, 3)`, info `(4, 83, 30)`
  — matches the `[his_window, 3]`/`[fut_window, 3]` contract used
  everywhere else in this repo.

## Result

**PASS.** Both the checkpoint strict-load and the dataset test-split
count reproduce the pre-loss recorded state exactly. Benchmark numbers
produced after this point in the session may be treated as trustworthy
per the gate this document exists to satisfy.
