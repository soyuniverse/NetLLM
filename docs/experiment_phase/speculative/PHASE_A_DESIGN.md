# Phase A — Reconnaissance and Block Verification Design

## 1. Baseline autoregressive loop and KV cache usage

Source: `src/netllm_litevlm/vp/llama_old_selectable_pipeline.py`,
`LlamaOldSelectablePipeline.auto_regressive`.

- History `[1,10,3]` is embedded one timestep at a time through the
  checkpoint-era `conv1d1` (`Conv1d(1,256,3)`) + `linear_layer`
  (`Linear(256,4096)`) pair, then `embed_ln` (LayerNorm) is applied once to
  the whole `[1,10,4096]` sequence. An optional selector runs exactly once
  on this normalized sequence (identity by default).
- The loop runs `fut_window_length` (20) times. **Every iteration calls**
  `old.plm(inputs_embeds=sequence, attention_mask=attention_mask)` **with
  no `past_key_values` and no `use_cache`** — confirmed by
  `docs/experiment_phase/llama/benchmark/CHECKPOINT_ERA_RUNTIME_CONTRACT.md`
  ("No `past_key_values` is passed back, so cache is never reused") and by
  `tests/llama_benchmark/test_llama_old_pipeline_contract.py` asserting
  `last_trace["cache_reused"] is False`. Sequence length grows `K, K+1,
  ..., K+19`; each forward reprocesses the entire prefix from scratch, so
  total attention work is `O(sum_{i=0..19}(K+i)^2)`, not `O(20)` forwards
  of fixed cost.
- **VP head position.** Recovered upstream source (`duowuyms/NetLLM` @
  `ee4d8726898610e4ae7df08bdd26728cafb4701f`,
  `viewport_prediction/models/old/llama.py` +
  `models/old/networking_head.py`) shows `LlamaTaskHeadModel2.forward()`
  passes the **full** last-layer `hidden_states` (all positions) to
  `SimpleLinearTaskHead.forward(hidden_states, input_ids_len)`, but that
  head unconditionally does:
  ```python
  last_one = input_logits.shape[1]
  needed_logits = input_logits[:, last_one - 1, :]   # LAST position only
  prediction = self.task_head(needed_logits)          # Linear(4096,3) -> Tanh
  ```
  So regardless of how long the input sequence is, exactly one `[1,1,3]`
  coordinate — the task head applied to the **last** position's hidden
  state — comes out per forward call. This matches the `[1,1,3]` contract
  documented in `CHECKPOINT_ERA_RUNTIME_CONTRACT.md` and the shape
  `DummyPLM` reproduces in the existing contract test.
- There is also a `teacher_forcing` path on the same head:
  `input_logits[:, size-fut_window-1:size-1, :]` fed through the *same*
  `Linear+Tanh` in one call — proof that the head is a plain
  position-independent `nn.Sequential`, safely callable on an arbitrary
  slice of hidden states. Block verification below relies on calling this
  raw `task_head.task_head` submodule directly on multiple new positions
  at once, bypassing the outer `forward()`'s single-last-position slicing
  (not modifying it).
- **Feedback embedding.** `feedback = old.linear_layer(old.conv1d1(result.logits)).unsqueeze(1)`
  reuses the exact same `conv1d1`/`linear_layer` pair as history embedding
  (no `embed_ln` on feedback — confirmed by
  `docs/implementation/LLAMA_VP_SELECTOR_IMPLEMENTATION_SUMMARY.md`:
  "Feedback ... is not LayerNorm-normalized"). The predicted `[1,1,3]`
  coordinate is the only continuous → embedding path; there is no
  intermediate discretization.

## 2. Why `continuous_draft_verify.py` still costs 20 target forwards

Source: `src/netllm_litevlm/speculative/continuous_draft_verify.py` and its
smoke driver `scripts/experiment_phase/speculative/run_llama_continuous_speculative_smoke.py`.

`ContinuousDraftVerify.run()` calls `self.target_predictor(history, steps,
context)` **unconditionally, before any comparison happens**. In the smoke
script, `target_predictor` is:

```python
def target_predictor(history_value, steps, context):
    prediction, _ = model.inference(history_value, future_raw, info)  # full 20-step baseline
    return TargetOutput(coordinates=prediction, forward_count=20, ...)
```

i.e. it always runs the complete, unmodified `LlamaOldSelectablePipeline`
autoregressive loop (20 forwards, no cache) to get the *entire* target
trajectory, and only afterward splices the draft's accepted prefix into
that already-fully-computed target output. `forward_count=20` is a literal
constant, not a measurement — there is no code path by which drafting could
short-circuit or reduce it. The class **validates acceptance/output-splice
control flow** correctly (as its own docstring says), but by construction
it cannot reduce target forward count, because verification never happens
*during* target generation — only *after* target generation is already
complete. This is the root cause CLAUDE.md refers to.

## 3. Block verification design

Single-forward, multi-position verification, replacing the target's
per-step black-box call with direct control over `old.plm`'s
`past_key_values`/`use_cache`/`output_hidden_states`, using a `carry`
value (the most recently confirmed-but-not-yet-cached coordinate) so every
iteration's forward embeds exactly `1 + gamma` new positions: `[carry,
draft_0, ..., draft_{gamma-1}]`.

- **KV cache format (measured, not assumed).** This environment now has
  `transformers==4.34.1` / `peft==0.6.2` installed per
  `docs/experiment_phase/llama/environment/LLAMA_ENVIRONMENT_PLAN_V2.md`.
  `transformers.cache_utils` does not exist at this version — `LlamaModel`
  returns the **legacy tuple-of-tuples** format: `tuple[num_layers]` of
  `(key, value)`, each `[B, num_heads, seq_len, head_dim]`. Rollback is a
  plain per-layer slice on `dim=2`. Verified empirically (fp32/CPU and
  fp16/GPU, real `LlamaModel`) that incremental cached decoding is
  **bit-exact** (`torch.equal=True`) versus full-sequence recompute — there
  is no fused-attention kernel in this version to introduce reduction-order
  drift, so the threshold=0 exactness gate is achievable by construction,
  not just approximately.
- **Per-iteration forward.** Embed `[carry, draft_0..draft_{g-1}]`
  (`g = min(gamma, remaining_steps)`) via the same `conv1d1`/`linear_layer`
  path as baseline feedback (no LayerNorm). One call:
  `old.plm(inputs_embeds=chunk, attention_mask=ones(cache_len+g+1),
  past_key_values=cache, use_cache=True, output_hidden_states=True,
  return_dict=True)`. Take `hidden = result.hidden_states[-1]` (`[1,
  g+1, 4096]`, confirmed identical to `result.last_hidden_state`) and apply
  `old.plm.task_head.task_head(hidden)` once to get `preds` (`[1, g+1,
  3]`) for **all** new positions in that single forward.
- **Indexing.** Causal masking makes `preds[:, k, :]` depend only on
  `cache + carry + draft_0..draft_{k-1}` — exactly the prefix `draft_k`
  itself was conditioned on. So `preds[k]` is the target's own opinion of
  what `draft_k` should be, for `k = 0..g-1`; `preds[g]` is a bonus
  (nothing to compare it to yet).
- **Acceptance.** Walk `k = 0..g-1`, accept while
  `||preds[k] - draft_k||_2 <= acceptance_threshold` (L2 over the 3
  coordinate dims, degree units after denormalization elsewhere in the
  pipeline); stop at the first rejection index `j`.
- **Output.** Append `preds[0..j-1]` (target's own values, not the drafts)
  to the confirmed sequence. If `j < g` (a rejection occurred), also append
  `preds[j]` — the bonus/rejected-position output, already computed in this
  same forward, no extra forward needed — and that becomes next
  iteration's `carry`. If `j == g` (full acceptance), `preds[g]` becomes
  next iteration's `carry` instead.
- **Cache rollback.** Always commit position `0` (`carry`, exact). Commit
  the `j` accepted draft positions. Truncate (discard) positions `j+1..g`
  — the rejected/unconsumed drafts — via a per-layer tensor slice back to
  `cache_len + 1 + j`. The bonus/carry value itself is *not* pre-committed;
  it is (re-)embedded fresh at the start of the next iteration, exactly
  like the very first `carry` produced by the initial warmup forward — this
  keeps the design symmetric across iterations and avoids ever caching an
  embedding for a value that was not exactly the one actually confirmed.
- **Degenerate case (`threshold=0`).** A naive constant-velocity draft
  essentially never exactly matches a full transformer forward, so `j=0`
  every iteration: exactly one new confirmed step per iteration, matching
  baseline forward-for-forward — `1` (initial warmup) + `19` (iterations)
  `= 20` target forwards, and because causal masking makes `preds[0]`
  independent of the (wasted) draft positions appended after it in the same
  batched call, the accepted value is bit-identical to what baseline would
  have produced at that step. This is what makes the threshold=0 gate
  achievable by construction rather than by luck.
- **Counters.** `target_forward_count` (real `old.plm` calls),
  `draft_forward_count` (always `0` — drafting is pure tensor
  arithmetic), `accepted_per_iteration` (list of `j` per iteration) are
  tracked directly on the pipeline instance.

## 4. Environment/asset status at time of writing

- `transformers==4.34.1`, `peft==0.6.2`, `accelerate==0.24.1` installed
  (pinned per `LLAMA_ENVIRONMENT_PLAN_V2.md` +
  `PLM_DEPENDENCY_COMPATIBILITY.md`'s accelerate/huggingface-hub fix);
  `torch` left at the pre-existing `2.2.0`.
- VP fine-tuned checkpoint (`try_llama2_7b` LoRA adapter +
  `modules_except_plm.bin`) and the Jin2022 dataset are **absent** from
  this instance (filesystem-wide search found only a key manifest, no
  weights, no raw viewport data). Base Llama2-7b weights are present
  (landed at `/root/NetLLM/*.safetensors` instead of
  `/root/NetLLM-assets/llama/base`; architecture confirmed matching:
  32 layers, hidden=4096).
- Decision (user, this session): proceed in reduced scope — validate
  block verification's control-flow/KV-cache correctness (threshold=0
  exactness, forward-count arithmetic) against a synthetic/base-weight
  stand-in rather than the fine-tuned checkpoint. Real predictive accuracy
  and real acceptance-rate behavior on Jin2022 remain unverified until the
  checkpoint and dataset are restored.
