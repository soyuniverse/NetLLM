# Gate-A Verification — 2026-08-09 (model-independent checks)

**Status: COMPLETE as of the second check below (same day).** The
initial attempt (kept in full below as history, not deleted) found
Gate-A blocked because the zips hadn't arrived; a later re-upload via
a Google Drive relay (scp had connectivity problems) landed them in
staging, and this document was revisited and completed the same day.
Jump to "2026-08-09 update — Gate-A COMPLETE" below for the completing
run; everything above that heading is the original (accurate at the
time) INCOMPLETE record, preserved as history per instruction.

Restructures the Task 0 gate into two stages per this session's
instruction: Gate-A (no base-model weights needed, runs now) and Gate-B
(needs the base model, deferred — see
`ASSET_RECOVERY_VERIFICATION_20260809.md` for the earlier full-gate
attempt and why it failed outright).

**Correction accepted from the user**: the absence of Llama2-7b base
weights on a fresh instance is expected, not an error — the 2026-08-02
backup (`docs/final/BACKUP_MANIFEST.md`) only ever covered the
checkpoint/dataset zips, never the base weights. A background
re-download (`hf download meta-llama/Llama-2-7b-hf ...`, PID 12362,
log `/root/llama_download.log`, target `/root/llama2-7b-base/`) is
correctly in progress at the time of this check and is **not** waited
on here, per instruction.

(Note: an earlier download attempt, PID 2627, had been pointed at the
repo root by mistake and overwrote `README.md` with the Llama2 model
card plus several loose config/tokenizer files. That process is no
longer running; `README.md` was restored via `git checkout` and the
stray loose files removed from the repo root as part of this session's
cleanup, before this document was written.)

## Step 1 — staging zip checksums vs. `BACKUP_MANIFEST.md`: BLOCKED, zips absent

Checked every path this project has ever used for these zips:

| path | expected | found |
|---|---|---:|
| `/root/NetLLM-assets/staging/try_llama2_7b.zip` | 77,861,701 B, sha256 `57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53` | **absent** |
| `/root/NetLLM-assets/staging/data.zip` | 3,199,081,523 B, sha256 `9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d` | **absent** |

`/root/NetLLM-assets/{staging,checkpoints,datasets,llama/base}/` all
exist as directories (created 2026-08-09 05:18, presumably scaffolded
ahead of an upload that hasn't landed yet) but **every one of them is
empty** — `find /root/NetLLM-assets -type f` returns nothing.
`find / -iname "*llama2_7b*" -o -iname "data.zip"` (repeated at the
start of this check) finds nothing anywhere on the instance, and no
`scp`/`rsync`/`wget`/`curl` transfer is currently running (`ps aux`
checked). This is not a checksum mismatch — there is nothing to
checksum. Unlike the base-model weights, this genuinely is a gap: the
checkpoint/dataset zips have not been re-uploaded to this instance yet.

## Steps 2–4: not attempted

Extraction, checkpoint file-structure verification
(`torch.load(map_location='cpu')`), and a live dataset test-split count
all require the zip contents from Step 1. None were attempted, per
instruction not to fabricate or approximate around a missing
precondition.

**Indirect, non-live evidence only** (not a substitute for a live
re-check, noted for context): the git-tracked per-sample CSVs from the
2026-08-02 run (e.g. `results/speculative/20260802T101802Z/
per_sample_baseline_selector=recent_k:2.csv`) each have exactly 1,698
data rows, consistent with the dataset test-split count recorded at the
time. This reflects the *previous* session's gated run, not a fresh
verification of dataset content on this instance.

## Result: Gate-A INCOMPLETE

Not a pass or a fail in the strict sense — the checkpoint/dataset zips
needed to run any of the four steps are not present on this instance.
**Task 1 and Task 2 this session do not depend on this gate** (Task 1
is interface/documentation work verified with a tiny CPU model; Task 2
re-reads already-git-tracked per-sample CSVs from the prior gated run)
and proceeded regardless, per instruction. Re-run this document (or a
dated successor) once `try_llama2_7b.zip`/`data.zip` actually land on
this instance.

## Gate-B (base model + checkpoint + dataset, deferred)

Not run this session. Requires, in addition to what Gate-A above still
needs: the `hf download` currently in progress
(`/root/llama2-7b-base/`) to finish downloading the actual
`*.safetensors` shards (only metadata/tokenizer files have landed as of
this check, ~275 MB vs. an expected ~13 GB for fp16 7B weights), **and**
the checkpoint/dataset zips from Step 1 above to actually arrive. Two
independent blockers, not one. Once both are satisfied: (1) base +
adapter strict load (missing/unexpected/mismatch keys must all be 0,
same procedure as `verify_checkpoint_strict_load.py`), (2) 50-sample
baseline MAE must reproduce 11.0368 within fp16 noise. Until Gate-B
passes, no new GPU benchmark number is trustworthy this session.

## 2026-08-09 update — Gate-A COMPLETE

`try_llama2_7b.zip` and `data.zip` were re-uploaded to
`/root/NetLLM-assets/staging/` via a Google Drive relay (direct `scp`
had connectivity problems). Base-weight download separately completed
in the background during this session (`/root/llama2-7b-base/`, both
`model-0000{1,2}-of-00002.safetensors` present, ~13 GiB total,
`hf_download.log` ends with `✓ Downloaded`) — tracked here for
context but not itself part of Gate-A, per the earlier correction that
base-weight absence was never the actual asset loss.

### Step 1 — staging zip checksums vs. `BACKUP_MANIFEST.md`: MATCH

```
9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d  data.zip
57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53  try_llama2_7b.zip
```

Both identical to the `BACKUP_MANIFEST.md` reference values. The
Google Drive relay did not corrupt either transfer.

### Step 2 — extraction + standard-path placement

`try_llama2_7b.zip` reproduces the same double-nesting as the original
2026-08-02 upload (`try_llama2_7b/try_llama2_7b/{adapter_config.json,
adapter_model.bin, modules_except_plm.bin, README.md}`), corrected the
same way: de-nested into `/root/NetLLM-assets/checkpoints/try_llama2_7b/`.
File sizes match the original `ASSET_RECOVERY_VERIFICATION.md` record
exactly: `adapter_config.json` 542 B, `adapter_model.bin` 67,155,338 B,
`modules_except_plm.bin` 16,900,050 B, `README.md` 5,479 B.

`data.zip` extracted directly to `/root/NetLLM-source/viewport_prediction/`
(same convention as before), producing `data/{viewports,images,ft_plms,
models,results}/` with `data/viewports/Jin2022/video{1..27}/` all
present (27 videos, confirmed by directory listing).

Both staging zips were **left in place** at
`/root/NetLLM-assets/staging/` after extraction (not deleted) — this
instance's own local backup copy, per this session's instruction.

### Step 3 — checkpoint file-structure verification (file-level, no model build)

`torch.load(map_location="cpu", weights_only=True)` on both files
directly (no `EmbeddingForViewportPrediction`/PEFT model assembled
yet — that's Gate-B):

- `adapter_model.bin`: 128 keys, all named
  `base_model.model.model.layers.{0..31}.self_attn.{q,v}_proj.lora_{A,B}.weight`
  — exactly 32 layers × 2 target modules × 2 (A/B) = 128, consistent
  with `adapter_config.json`'s `r=32`, `target_modules=["v_proj",
  "q_proj"]`.
- `modules_except_plm.bin`: 10 keys with shapes `0.weight (4096,256)`,
  `0.bias (4096,)`, `1.weight (4096,768)`, `1.bias (4096,)`,
  `2.weight (4096,)`, `2.bias (4096,)`, `3.0.weight (256,1,3)`,
  `3.0.bias (256,)`, `4.task_head.0.weight (3,4096)`,
  `4.task_head.0.bias (3,)` — matches the non-PLM submodule shapes
  `EmbeddingForViewportPrediction`'s embedding/task-head stack expects
  (linear_layer, embed_ln, conv1d1, task_head).
- Combined with the file-size and whole-zip sha256 matches above, this
  is strong evidence against corruption, but is not the same claim as
  a strict-load 0/0/0/0/0 result — that requires the real
  `EmbeddingForViewportPrediction`/PEFT model assembled against the
  base weights, which is Gate-B's job (next section).

### Step 4 — dataset test split

`create_dataset("Jin2022", his_window=10, fut_window=20, trim_head=30,
trim_tail=60, frequency=5, step=15, include=["test"])` (identical
parameters to every prior run): **test split length = 1,698**, exact
match. Sample 0 shapes `(10,3)`/`(20,3)`, `(video, user, timestep) =
(4, 83, 30)` — identical to the value recorded in this project's
history before any asset loss.

### Result: Gate-A COMPLETE

All four steps pass. Checkpoint and dataset are byte-for-byte
transferred and structurally sane. Gate-B (full strict load + 50-sample
MAE reproduction, now possible since the base weights finished
downloading) follows in this same document.
