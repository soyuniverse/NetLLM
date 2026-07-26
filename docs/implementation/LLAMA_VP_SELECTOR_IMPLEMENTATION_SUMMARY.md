# Llama VP Selector Implementation Summary

## Original checkpoint-era path

The recovered checkpoint is executed with the immutable source at:

- source: `/root/NetLLM-source-checkpoint-era`
- commit: `ee4d8726898610e4ae7df08bdd26728cafb4701f`
- entry path: `viewport_prediction/run_old.py`
- model path: `viewport_prediction/models/old/`
- trained prediction head: `task_head`

The original source, base model, LoRA adapter, `modules_except_plm.bin`, and dataset are not patched or rewritten.

## External composition wrapper

`src/netllm_litevlm/vp/llama_old_selectable_pipeline.py` defines `LlamaOldSelectablePipeline`. It receives the already strict-loaded checkpoint-era `EmbeddingForViewportPrediction` instance and reuses every trained module owned by it. The wrapper does not monkey-patch the old source and does not construct a replacement prediction head.

The selector insertion point is:

```text
normalized viewport history
→ old Conv1d/Linear temporal embedding
→ old embed_ln LayerNorm
→ selector (one call on initial history)
→ unchanged old Llama autoregressive loop
→ unchanged task_head
```

This location preserves the original embedding and LayerNorm operations while allowing the initial history length to change.

## Selector contract

`SelectionOutput` carries:

- selected embeddings `[B,K,E]`
- matching attention mask `[B,K]`, if present
- original time indices
- optional scores
- original and selected lengths
- metadata describing the selection

`IdentitySelector` returns the exact input tensor, mask, and ordered indices `0..L-1`. `RecentKSelector(k)` returns the suffix `L-k..L-1` without changing temporal order and slices the attention mask identically.

## Attention, feedback, and cache behavior

The wrapper rebuilds the initial all-ones attention mask for the selected length. At every future step it appends one mask entry together with the original feedback embedding.

Selection is not applied to autoregressive feedback. Each predicted coordinate is embedded by the original Conv1d/Linear modules and appended exactly as in the checkpoint-era implementation. Feedback does not receive an additional LayerNorm. The original final unused feedback computation is preserved.

Batch size remains 1. KV cache remains unused. Every 20-step prediction therefore performs 20 full PLM forwards with sequence lengths `K..K+19`.

## Verification and benchmark flow

1. Strict-load the Llama base, LoRA adapter, and non-PLM state with missing/unexpected keys equal to zero.
2. Reuse one model instance for Original, selector-disabled, and Identity paths.
3. Require Identity outputs and metrics to match Original within the FP16 contract.
4. Run a 128-sample pilot with identical sample order and measurement policy.
5. Run the complete 1,698-sample test split only after the pilot succeeds.
6. Store per-sample errors, prediction digests, latency, selector latency, and resumable progress.
7. Aggregate MAE, upstream RMSE, corrected rotation-aware RMSE, mean angular error, exact normalized MSE loss, latency, and GPU memory.
8. Generate figures only from the successful full summary.

The result is a recovered-artifact controlled comparison. It is not claimed as a paper reproduction because training-time provenance remains incomplete.

