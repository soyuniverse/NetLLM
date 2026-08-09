# Gate-A Verification — 2026-08-09 (model-independent checks)

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
