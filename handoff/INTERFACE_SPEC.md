# Interface Spec

Exact input/output contracts for the two pieces you're integrating.
Read this before writing a custom selector or swapping the adapter.

## 1. Selector interface (`BaseSelector`)

`src/netllm_litevlm/selectors/base.py`

```python
@dataclass
class SelectionOutput:
    embeddings: torch.Tensor            # [1, selected_length, embed_size]
    attention_mask: Optional[torch.Tensor]  # [1, selected_length] or None
    selected_indices: torch.Tensor      # [selected_length], long, indices into the ORIGINAL sequence
    scores: Optional[torch.Tensor]      # per-selected-position importance score, or None
    original_length: int
    selected_length: int
    metadata: Dict[str, Any]            # free-form; existing selectors put {"selector": class name, "k"/"policy", "context": ...}

class BaseSelector(nn.Module, ABC):
    def forward(
        self,
        embeddings: torch.Tensor,             # [1, L, embed_size], L=10 (history window) in this pipeline
        attention_mask: Optional[torch.Tensor] = None,  # [1, L] or None
        context: Optional[Dict[str, Any]] = None,        # e.g. {"task": "viewport_prediction", "stage": "initial_history"}
    ) -> SelectionOutput:
        ...
```

**Contract every implementation must satisfy** (see
`BaseSelector.validate_inputs`, called by every existing selector before
touching its input):

- `embeddings.ndim == 3`, batch dimension is always 1 (`B=1` — this
  pipeline processes one sample at a time, see `SpeculativeBlockVerifyPipeline
  .auto_regressive`'s own `history.shape[0] != 1` check).
- If `attention_mask` is given, it must be `[B, L]` and share `L` and
  device with `embeddings`.
- **Time-order preservation**: `selected_indices` must be strictly
  increasing (ascending order into the original sequence). Every
  existing selector sets `metadata["preserves_order"] = True` — this
  isn't enforced by an assertion in `base.py`, but both pipeline call
  sites feed `SelectionOutput.embeddings` straight into a causal
  transformer (`old.plm(inputs_embeds=sequence, ...)`), so an
  out-of-order selection silently corrupts the causal structure the
  model was fine-tuned under. Don't reorder.
- `embeddings.shape[2]` (the embedding dim, 4096 for this checkpoint)
  must be unchanged — selectors prune along the sequence axis only,
  never the feature axis.
- Output `embeddings.shape[1]` must equal `selected_length`, and
  `selected_length <= original_length`.

**Patch selection is a drop-in for this interface**: if your teammate's
patch-selection code currently produces "which positions/patches to
keep" as an index list or boolean mask over a sequence, wrapping it to
return a `SelectionOutput` (slice `embeddings`/`attention_mask` by the
kept indices, set `selected_indices` accordingly) makes it usable
anywhere a selector is accepted — `LlamaOldSelectablePipeline(model,
selector=your_patch_selector)` and
`SpeculativeBlockVerifyPipeline(model, selector=your_patch_selector,
...)` both just call `self.selector(sequence, attention_mask,
context={...})` once, at the very start of `auto_regressive`, before
the first target forward. See `example_integration.py` §(a) for a
worked skeleton.

**Where selection is applied**: exactly once, on the LayerNorm'd
initial history embeddings (10 timesteps), before any autoregressive
step — not re-applied per future step. Look at
`SpeculativeBlockVerifyPipeline.auto_regressive`,
`src/netllm_litevlm/speculative/block_verify.py:121-140`, or the
baseline's equivalent in `llama_old_selectable_pipeline.py`.

## 2. `SpeculativeBlockVerifyPipeline`

`src/netllm_litevlm/speculative/block_verify.py:66`

```python
SpeculativeBlockVerifyPipeline(
    pipeline: nn.Module,          # an EmbeddingForViewportPrediction instance (or anything
                                   # exposing .embedding_model / .using_multimodal, see
                                   # checkpoint_era_runtime.load_checkpoint_era_model)
    selector: Optional[BaseSelector] = None,
    draft_model: Optional[ContinuousDraftModel] = None,   # defaults to RecentVelocityDraft()
    gamma: int = 4,                # draft block size (coordinates proposed per target forward)
    acceptance_threshold: float = 0.0,
)
```

**`acceptance_threshold` unit space: normalized, NOT degrees.** The
task head's output is Tanh-bounded (~[-1,1] per channel); the L2
distance compared against `acceptance_threshold` in
`block_verify.py:194-199` is computed directly on that normalized
output, before any denormalization to degrees. If you're calibrating a
threshold for a new selector or checkpoint, do it in this same
normalized space — do not reuse a degree-space error budget directly.
Empirical calibration reference (10 real samples, this checkpoint):
draft-vs-target normalized L2 disagreement has median 0.174, most mass
in 0.01–0.7, rare fast-yaw outliers reaching 3–9
(`docs/experiment_phase/speculative/PHASE_A_DESIGN.md`).

Call contract:

```python
prediction, target = pipeline.inference(history, future, video_user_position)
# history: [1, 10, 3] normalized (Tanh-input-space) coordinates
# future:  [1, 20, 3] raw-degree target coordinates (denormalized by caller after)
# prediction: [1, 20, 3] normalized-space output — denormalize before computing MAE/RMSE
#             (see run_speculative_benchmark.py's utils.normalize.denormalize_data call —
#             this was a real bug the first time this harness was written, see
#             docs/experiment_phase/assets/ASSET_RECOVERY_VERIFICATION.md)
```

After a call, `pipeline.target_forward_count` (int),
`pipeline.draft_forward_count` (int, always 0 — the draft model here is
a closed-form extrapolation, not a second LLM forward), and
`pipeline.accepted_per_iteration` (`List[int]`, per-target-forward
accept counts, not persisted anywhere by the benchmark harness — see
`HANDOFF.md`'s note if you need per-iteration granularity) are
populated. `pipeline.last_trace` has all of these plus
`prediction_shape` and `final_cache_length` in one dict.

**Gate every new draft model / selector combination must pass**
(mirrored in `tests/speculative/test_block_verify.py`, safe to copy):
at `acceptance_threshold=0.0`, every draft coordinate must be rejected,
so `SpeculativeBlockVerifyPipeline`'s output must match
`LlamaOldSelectablePipeline`'s output on the same
`(model, selector, history)` within floating-point tolerance
(**atol=1e-5 on fp32/CPU, atol=2e-3 on fp16/GPU** — see `HANDOFF.md`'s
environment warning for why this isn't exact-equal), and
`target_forward_count` must equal `fut_window` (20) exactly, with
`sum(accepted_per_iteration) == 0`. `handoff/smoke_test.py` runs this
exact check on a tiny CPU model.

## 3. AdaLoRA integration point

The LoRA adapter is attached and loaded in exactly one place:
`src/netllm_litevlm/vp/checkpoint_era_runtime.py`, function
`load_checkpoint_era_model`:

```python
# line 88 -- adapter attachment (currently plain LoRA via peft.LoraConfig)
plm = peft_model(base, "llama", rank)
#     ^ third_party/netllm_upstream/viewport_prediction/models/low_rank.py:28
#       uses `from peft import LoraConfig, get_peft_model, TaskType`
#       target_modules=["q_proj", "v_proj"], r=rank, lora_alpha=32, lora_dropout=0.05

# line 107-108 -- fine-tuned weights load
model.plm.load_adapter(str(checkpoint_path), adapter_name="default")
model.plm.set_adapter("default")
```

**To swap in AdaLoRA**: replace the `peft_model(...)` call (or the
`LoraConfig`/`get_peft_model` call it wraps) with peft==0.6.2's
`AdaLoraConfig`/`get_peft_model`. Everything downstream —
`model.plm.load_adapter(...)`, the task head, the embedding modules,
both pipeline wrappers, both selectors — is unaware of which PEFT
method produced `model.plm`; they only call `.forward`/`__call__` and
read `.task_head.task_head`, which live on the non-PEFT-specific
wrapper (`EmbeddingForViewportPrediction`, `LlamaTaskHeadModel2`), not
on the adapter itself. **Do not edit `third_party/netllm_upstream/...`
in place** (vendored, unmodified upstream source, see its own
`PROVENANCE.md`) — add the AdaLoRA config as a new function/branch in
`checkpoint_era_runtime.py`, this project's own wrapper file.

**Invariant that must hold after the swap**: the threshold=0
equivalence gate above (§2) is adapter-agnostic — it doesn't test
prediction *accuracy*, only that speculative decoding reproduces
whatever the base pipeline outputs. **Run it against your AdaLoRA
checkpoint before trusting any accuracy number** — it's the fastest way
to confirm the new adapter's forward pass is wired correctly into both
pipelines (same shapes, same dtype, same call contract) before
spending GPU time on a real accuracy benchmark. If it fails, the bug is
in the adapter/checkpoint assembly, not in `block_verify.py`'s
accept/reject logic, since that logic is unreachable at threshold=0
(every draft is rejected by construction).

One structural constraint carried over from the base pipeline
(`checkpoint_era_runtime.py:80`): `only the proven non-multimodal
checkpoint is supported` — both pipeline wrappers raise `ValueError` if
`pipeline.using_multimodal` is `True`. If your AdaLoRA checkpoint is
multimodal, that guard needs a deliberate decision, not silent removal.
