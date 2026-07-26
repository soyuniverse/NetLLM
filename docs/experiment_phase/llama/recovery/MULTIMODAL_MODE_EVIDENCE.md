# Multimodal Mode Evidence

## Decision

Classification: **proven_non_multimodal**

## Evidence chain

1. The authoritative checkpoint-era README identifies the published
   `try_llama2_7b` checkpoint and requires `run_old.py`.
2. Its exact checkpoint invocation does not include `--multimodal`.
3. `run_old.py` defines `--multimodal` with `action="store_true"`, so absence
   deterministically sets `using_multimodal=False`.
4. The README separately states that image features are enabled by appending
   `--multimodal`; the published checkpoint command does not append it.
5. The example/debug block for the same `try_llama2_7b` path explicitly assigns
   `args.using_multimodal = False`.
6. `embed_multimodal` is unconditionally created and stored, explaining why its
   tensors are present even in a non-multimodal checkpoint.

The uploaded dataset's missing processed features is therefore not a blocker
for this checkpoint's non-multimodal technical smoke. It remains a blocker for
any separate multimodal experiment.

## Scope

This determination applies to the upstream-published `try_llama2_7b`
checkpoint identified by the README and the supplied archive bearing that
name. It does not assert that every rank-32 Llama checkpoint is
non-multimodal.
