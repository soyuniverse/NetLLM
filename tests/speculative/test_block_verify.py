"""Gate tests for SpeculativeBlockVerifyPipeline.

Real fine-tuned checkpoint + Jin2022 dataset are unavailable in this
instance (see docs/experiment_phase/speculative/PHASE_A_DESIGN.md). Gates
1-2 below use a real (tiny) transformers LlamaModel wrapped in the same
`inputs_embeds/attention_mask/past_key_values/use_cache/output_hidden_states`
contract as the checkpoint-era `LlamaTaskHeadModel2`, so the KV-cache
mechanics under test are the genuine transformers==4.34.1 code path, not a
hand-rolled approximation of attention. This validates block verification's
control-flow correctness; it says nothing about real VP prediction
accuracy or real acceptance rates on Jin2022.
"""

from types import SimpleNamespace
from typing import List

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from netllm_litevlm.selectors import RecentKSelector
from netllm_litevlm.speculative import (
    ContinuousDraftModel,
    DraftOutput,
    RecentVelocityDraft,
    SpeculativeBlockVerifyPipeline,
    slice_past_key_values,
)
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline

EMBED_SIZE = 16
FUT_WINDOW = 7
GAMMA = 3


class _FakeTaskHeadModel(nn.Module):
    """Wraps a real LlamaModel with the LlamaTaskHeadModel2 call contract:
    accepts past_key_values/use_cache/output_hidden_states and exposes a
    `.task_head.task_head` submodule callable on arbitrary hidden-state
    slices (mirrors the recovered upstream SimpleLinearTaskHead)."""

    def __init__(self, config: LlamaConfig, embed_size: int):
        super().__init__()
        self.model = LlamaModel(config)
        self.task_head = SimpleNamespace(
            task_head=nn.Sequential(nn.Linear(embed_size, 3, bias=True), nn.Tanh())
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        prediction = self.task_head.task_head(outputs.last_hidden_state[:, -1:, :])
        return SimpleNamespace(
            logits=prediction,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
        )


def _make_pipeline(fut_window_length: int, embed_size: int = EMBED_SIZE):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=embed_size,
        intermediate_size=embed_size * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    old = nn.Module()
    old.plm = _FakeTaskHeadModel(config, embed_size)
    old.conv1d1 = nn.Sequential(nn.Conv1d(1, 256, 3), nn.Flatten())
    old.linear_layer = nn.Linear(256, embed_size)
    old.embed_ln = nn.LayerNorm(embed_size)
    old.embed_size = embed_size
    old.fut_window_length = fut_window_length
    pipeline = nn.Module()
    pipeline.embedding_model = old
    pipeline.using_multimodal = False
    pipeline.loss_fct = nn.MSELoss()
    return pipeline


def _large_velocity_history(seed: int) -> torch.Tensor:
    """Velocity large enough that RecentVelocityDraft output magnitude
    exceeds 1, which is outside the Tanh-bounded target range ([-1,1]^3).
    This makes threshold=0 rejection deterministic by construction, not by
    chance of the random model weights."""
    torch.manual_seed(seed)
    base = torch.randn(1, 2, 3)
    velocity = torch.full((1, 1, 3), 5.0)
    steps = torch.arange(2).view(1, 2, 1).float()
    return base[:, :1, :] + steps * velocity


def _small_velocity_history(seed: int) -> torch.Tensor:
    """Small velocity keeps draft extrapolation within a few units of the
    Tanh-bounded target range, so a generously large (but finite) threshold
    guarantees acceptance regardless of the specific random model weights."""
    torch.manual_seed(seed)
    base = torch.randn(1, 2, 3) * 0.1
    velocity = torch.full((1, 1, 3), 0.05)
    steps = torch.arange(2).view(1, 2, 1).float()
    return base[:, :1, :] + steps * velocity


def test_threshold_zero_matches_baseline_exactly():
    # Tolerance note (measured, not assumed): a *single* KV-cache extension
    # hop is bit-exact (torch.equal) versus full recompute on this
    # transformers==4.34.1 / eager-attention stack, in both fp32/CPU and
    # fp16/GPU (see docs/experiment_phase/speculative/PHASE_A_DESIGN.md).
    # Chaining many sequential hops (as this 7-step loop does) exposes
    # shape-dependent floating-point reassociation in the underlying
    # matmul/attention kernels -- e.g. a length-1 batch and a length-2
    # batch are not guaranteed to reduce a causally-masked position's own
    # attention sum in the same operation order, even though causal
    # masking guarantees they read the same logical values. This reproduces
    # independent of this pipeline (isolated LlamaModel-only chains show
    # the same ~1e-8..1e-7 drift) and is unaffected by
    # torch.use_deterministic_algorithms(True), so it is inherent BLAS
    # shape-dependent noise, not a control-flow bug. 1e-5 absolute
    # tolerance is ~40x the largest diff observed in this suite.
    torch.manual_seed(0)
    pipeline = _make_pipeline(FUT_WINDOW)
    baseline = LlamaOldSelectablePipeline(pipeline)
    speculative = SpeculativeBlockVerifyPipeline(
        pipeline, gamma=GAMMA, acceptance_threshold=0.0
    )

    for seed in range(5):
        history = _large_velocity_history(seed)
        with torch.no_grad():
            expected = baseline.auto_regressive(history, None)
            actual = speculative.auto_regressive(history, None)

        assert actual.shape == expected.shape == (1, FUT_WINDOW, 3)
        max_abs_diff = (actual - expected).abs().max().item()
        assert torch.allclose(actual, expected, atol=1e-5, rtol=0), (
            f"seed={seed}: max abs diff {max_abs_diff}"
        )
        assert speculative.target_forward_count == FUT_WINDOW
        assert speculative.draft_forward_count == 0
        assert sum(speculative.accepted_per_iteration) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16/GPU is the deployment precision")
def test_threshold_zero_matches_baseline_within_fp16_precision_on_gpu():
    # fp16 has ~2^-10 (~1e-3) relative precision, so the same chained
    # KV-cache reassociation noise as the CPU/fp32 test above (there
    # ~1e-7) is amplified to ~1e-3 here -- measured directly below, not
    # assumed. This is a property of fp16 arithmetic, not of this
    # pipeline: any bit-for-bit-equivalent reimplementation would show the
    # same floor. torch.testing's own fp16 defaults use a comparable
    # (rtol=1e-3) tolerance for this reason.
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    pipeline = _make_pipeline(FUT_WINDOW).to(device=device, dtype=dtype)
    pipeline.embedding_model.plm.task_head.task_head.to(device=device, dtype=dtype)
    baseline = LlamaOldSelectablePipeline(pipeline)
    speculative = SpeculativeBlockVerifyPipeline(
        pipeline, gamma=GAMMA, acceptance_threshold=0.0
    )

    for seed in range(5):
        history = _large_velocity_history(seed).to(device=device, dtype=dtype)
        with torch.no_grad():
            expected = baseline.auto_regressive(history, None)
            actual = speculative.auto_regressive(history, None)

        assert actual.shape == expected.shape == (1, FUT_WINDOW, 3)
        max_abs_diff = (actual - expected).abs().max().item()
        assert torch.allclose(actual, expected, atol=2e-3, rtol=0), (
            f"seed={seed}: max abs diff {max_abs_diff}"
        )
        assert speculative.target_forward_count == FUT_WINDOW
        assert speculative.draft_forward_count == 0
        assert sum(speculative.accepted_per_iteration) == 0


def test_threshold_large_reduces_forward_count():
    torch.manual_seed(1)
    pipeline = _make_pipeline(FUT_WINDOW)
    speculative = SpeculativeBlockVerifyPipeline(
        pipeline, gamma=GAMMA, acceptance_threshold=10.0
    )

    for seed in range(5):
        history = _small_velocity_history(seed)
        with torch.no_grad():
            prediction = speculative.auto_regressive(history, None)

        assert prediction.shape == (1, FUT_WINDOW, 3)
        assert torch.isfinite(prediction).all()
        assert speculative.target_forward_count < FUT_WINDOW
        assert speculative.draft_forward_count == 0
        assert sum(speculative.accepted_per_iteration) == FUT_WINDOW - 1


class _ScriptedDraft(ContinuousDraftModel):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self._values = values

    def forward(self, history, steps, context=None):
        del history, context
        return DraftOutput(
            coordinates=self._values[:, :steps, :], forward_count=1, metadata={}
        )


class _ScriptedPLM(nn.Module):
    """Returns a pre-scripted hidden state per new position, in call order,
    ignoring actual embedding content. Isolates the accept/reject/bonus/
    rollback index arithmetic from real attention numerics (covered
    separately by the LlamaModel-backed gate tests above)."""

    def __init__(self, position_outputs):
        super().__init__()
        self.task_head = SimpleNamespace(task_head=lambda h: h)
        self._position_outputs = position_outputs
        self._cursor = 0
        self.call_count = 0

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        self.call_count += 1
        n_new = inputs_embeds.shape[1]
        chunk = self._position_outputs[self._cursor : self._cursor + n_new]
        self._cursor += n_new
        hidden = torch.cat([value.view(1, 1, 3) for value in chunk], dim=1)
        old_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        dummy = torch.zeros(1, 1, old_len + n_new, 1)
        return SimpleNamespace(hidden_states=(hidden,), past_key_values=((dummy, dummy),))


def test_partial_acceptance_bonus_and_rollback_scripted():
    # gamma=3, fut_window=5: iteration 1 accepts 2/3 drafts and rejects the
    # 3rd (bonus token becomes the confirmed value there, no extra forward);
    # iteration 2 fully accepts its single remaining draft.
    position_outputs = [
        torch.tensor([0.0, 0.0, 0.0]),  # warmup -> carry (c0)
        torch.tensor([1.0, 0.0, 0.0]),  # iter1 pred0 vs draft0 -> accept
        torch.tensor([2.0, 0.0, 0.0]),  # iter1 pred1 vs draft1 -> accept
        torch.tensor([10.0, 0.0, 0.0]),  # iter1 pred2 vs draft2 -> reject, used as bonus
        torch.tensor([99.0, 0.0, 0.0]),  # iter1 bonus position, discarded (never read)
        torch.tensor([1.0, 0.0, 0.0]),  # iter2 pred0 vs draft0 -> accept (full accept)
        torch.tensor([42.0, 0.0, 0.0]),  # iter2 bonus position, computed but unused
    ]
    draft_values = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    old = nn.Module()
    old.plm = _ScriptedPLM(position_outputs)
    old.conv1d1 = nn.Sequential(nn.Conv1d(1, 256, 3), nn.Flatten())
    old.linear_layer = nn.Linear(256, 8)
    old.embed_ln = nn.LayerNorm(8)
    old.embed_size = 8
    old.fut_window_length = 5
    pipeline = nn.Module()
    pipeline.embedding_model = old
    pipeline.using_multimodal = False

    speculative = SpeculativeBlockVerifyPipeline(
        pipeline,
        draft_model=_ScriptedDraft(draft_values),
        gamma=3,
        acceptance_threshold=0.5,
    )
    history = torch.zeros(1, 1, 3)
    with torch.no_grad():
        prediction = speculative.auto_regressive(history, None)

    expected = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    assert torch.equal(prediction, expected)
    assert speculative.target_forward_count == 3  # warmup + 2 iterations
    assert speculative.draft_forward_count == 0
    assert speculative.accepted_per_iteration == [2, 1]


def test_slice_past_key_values_truncates_each_layer():
    key = torch.arange(2 * 1 * 6 * 2, dtype=torch.float32).view(2, 1, 6, 2)
    value = key + 100.0
    past_key_values = ((key, value), (key * 2, value * 2))

    truncated = slice_past_key_values(past_key_values, keep_length=4)

    assert len(truncated) == 2
    for (orig_k, orig_v), (trunc_k, trunc_v) in zip(past_key_values, truncated):
        assert trunc_k.shape == (2, 1, 4, 2)
        assert trunc_v.shape == (2, 1, 4, 2)
        assert torch.equal(trunc_k, orig_k[:, :, :4, :])
        assert torch.equal(trunc_v, orig_v[:, :, :4, :])


# --- Selector + speculative combination gates ---------------------------
# Requested before any (Selector, SpeculativeBlockVerifyPipeline) ablation
# is trusted at real-checkpoint scale.

HIS_WINDOW = 10


def _ten_step_large_velocity_history(seed: int) -> torch.Tensor:
    """Same large-velocity construction as _large_velocity_history, but
    HIS_WINDOW=10 steps long so RecentKSelector(k<10) has something real
    to select from."""
    torch.manual_seed(seed)
    base = torch.randn(1, 1, 3)
    velocity = torch.full((1, 1, 3), 5.0)
    steps = torch.arange(HIS_WINDOW).view(1, HIS_WINDOW, 1).float()
    return base + steps * velocity


class _RecordingDraftModel(ContinuousDraftModel):
    """Wraps RecentVelocityDraft, recording every draft_history it was
    called with so a test can compare across selector configurations
    without needing to inspect block_verify.py's internals directly."""

    def __init__(self):
        super().__init__()
        self._inner = RecentVelocityDraft()
        self.calls: List[torch.Tensor] = []

    def forward(self, history, steps, context=None):
        self.calls.append(history.clone())
        return self._inner(history, steps, context)


def test_draft_velocity_ignores_selector_k_uses_full_history():
    # block_verify.py's draft_history = cat(history, *confirmed) always
    # uses the `history` argument passed into auto_regressive() directly
    # -- never `sequence`/the selector's embeddings -- so the selector's
    # reduction from HIS_WINDOW to K should have zero effect on what the
    # draft model sees, regardless of K.
    torch.manual_seed(0)
    pipeline = _make_pipeline(FUT_WINDOW)
    history = _ten_step_large_velocity_history(seed=0)

    draft_full = _RecordingDraftModel()
    speculative_full = SpeculativeBlockVerifyPipeline(
        pipeline, draft_model=draft_full, gamma=GAMMA, acceptance_threshold=0.0
    )
    with torch.no_grad():
        speculative_full.auto_regressive(history, None)

    draft_selected = _RecordingDraftModel()
    speculative_selected = SpeculativeBlockVerifyPipeline(
        pipeline,
        selector=RecentKSelector(4),
        draft_model=draft_selected,
        gamma=GAMMA,
        acceptance_threshold=0.0,
    )
    with torch.no_grad():
        speculative_selected.auto_regressive(history, None)

    # The confirmed/carry portion of draft_history legitimately differs
    # between the two runs (it's the LLM's own output, and the selector
    # changes what the LLM sees) -- that's not what's under test here.
    # What must hold regardless of K is that every draft_history call's
    # *original-history* prefix is exactly the untruncated HIS_WINDOW
    # history, never a K-selected slice of it.
    assert len(draft_full.calls) == len(draft_selected.calls)
    for full_call, selected_call in zip(draft_full.calls, draft_selected.calls):
        assert full_call.shape[1] >= HIS_WINDOW
        assert selected_call.shape[1] >= HIS_WINDOW
        assert torch.equal(full_call[:, :HIS_WINDOW, :], history)
        assert torch.equal(selected_call[:, :HIS_WINDOW, :], history)


@pytest.mark.parametrize("k", [4, 6, 10])
def test_threshold_zero_matches_baseline_with_recent_k_selector(k):
    # Combined-pipeline exactness gate: KV-cache position indexing must
    # stay correct when the initial prefill length is K (from the
    # selector) instead of HIS_WINDOW, for every K including the
    # no-op K=HIS_WINDOW case.
    torch.manual_seed(1)
    pipeline = _make_pipeline(FUT_WINDOW)
    baseline = LlamaOldSelectablePipeline(pipeline, selector=RecentKSelector(k))
    speculative = SpeculativeBlockVerifyPipeline(
        pipeline, selector=RecentKSelector(k), gamma=GAMMA, acceptance_threshold=0.0
    )

    for seed in range(5):
        history = _ten_step_large_velocity_history(seed)
        with torch.no_grad():
            expected = baseline.auto_regressive(history, None)
            actual = speculative.auto_regressive(history, None)

        assert actual.shape == expected.shape == (1, FUT_WINDOW, 3)
        max_abs_diff = (actual - expected).abs().max().item()
        assert torch.allclose(actual, expected, atol=1e-5, rtol=0), (
            f"k={k} seed={seed}: max abs diff {max_abs_diff}"
        )
        assert speculative.target_forward_count == FUT_WINDOW
        assert speculative.last_selection_output.selected_length == k
