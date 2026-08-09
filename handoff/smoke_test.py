#!/usr/bin/env python3
"""Standalone smoke test -- no checkpoint, no dataset, CPU only, ~1 minute.

Run this after integrating these modules into your own code to confirm
the interface contracts and the threshold=0 equivalence gate still hold.
It builds a tiny REAL transformers LlamaModel (2 layers, hidden_size=16)
wrapped in the exact `inputs_embeds/attention_mask/past_key_values/
use_cache/output_hidden_states` call contract the real checkpoint-era
model uses, so the KV-cache mechanics under test are genuine
transformers==4.34.1 code, not a hand-rolled approximation. It says
nothing about real VP prediction accuracy or real acceptance rates --
only that the plumbing (selectors, block verification, KV-cache
slicing) is wired correctly. Mirrors
tests/speculative/test_block_verify.py's own gate tests; kept
standalone here (no pytest dependency) so it runs the same way whether
or not your integration target has pytest installed.

Usage: python handoff/smoke_test.py
Exit code 0 = all checks passed, non-zero = see printed failure.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:
    from transformers import LlamaConfig  # noqa: E402
    from transformers.models.llama.modeling_llama import LlamaModel  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"FAIL: transformers not importable ({exc}). Run `pip install -r requirements-vp.txt`.")
    sys.exit(1)

from netllm_litevlm.selectors import IdentitySelector, RecentKSelector  # noqa: E402
from netllm_litevlm.speculative import SpeculativeBlockVerifyPipeline  # noqa: E402
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline  # noqa: E402

EMBED_SIZE = 16
FUT_WINDOW = 7
GAMMA = 3
FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


class _FakeTaskHeadModel(nn.Module):
    """Real LlamaModel wrapped with the LlamaTaskHeadModel2 call contract."""

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


def _make_pipeline() -> nn.Module:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=EMBED_SIZE,
        intermediate_size=EMBED_SIZE * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    old = nn.Module()
    old.plm = _FakeTaskHeadModel(config, EMBED_SIZE)
    old.conv1d1 = nn.Sequential(nn.Conv1d(1, 256, 3), nn.Flatten())
    old.linear_layer = nn.Linear(256, EMBED_SIZE)
    old.embed_ln = nn.LayerNorm(EMBED_SIZE)
    old.embed_size = EMBED_SIZE
    old.fut_window_length = FUT_WINDOW
    pipeline = nn.Module()
    pipeline.embedding_model = old
    pipeline.using_multimodal = False
    pipeline.loss_fct = nn.MSELoss()
    return pipeline


def _large_velocity_history(seed: int) -> torch.Tensor:
    """Velocity large enough that the draft's extrapolation exceeds the
    Tanh-bounded [-1,1] output range, guaranteeing threshold=0 rejection
    by construction (not by chance of random model weights)."""
    torch.manual_seed(seed)
    base = torch.randn(1, 2, 3)
    velocity = torch.full((1, 1, 3), 5.0)
    steps = torch.arange(2).view(1, 2, 1).float()
    return base[:, :1, :] + steps * velocity


def main() -> int:
    print(f"Smoke test: tiny CPU LlamaModel, embed_size={EMBED_SIZE}, "
          f"fut_window={FUT_WINDOW}, gamma={GAMMA}\n")

    # --- Selector interface contract (no model needed) ---------------------
    embeddings = torch.randn(1, 10, EMBED_SIZE)
    mask = torch.ones(1, 10, dtype=torch.long)
    for selector, name in [(IdentitySelector(), "IdentitySelector"), (RecentKSelector(2), "RecentKSelector(2)")]:
        out = selector(embeddings, mask, context={"task": "viewport_prediction"})
        check(f"{name}: embeddings shape valid", out.embeddings.shape[0] == 1 and out.embeddings.shape[2] == EMBED_SIZE)
        check(f"{name}: selected_length matches output", out.embeddings.shape[1] == out.selected_length)
        check(
            f"{name}: selected_indices ascending (time-order preserved)",
            list(out.selected_indices.tolist()) == sorted(out.selected_indices.tolist()),
        )

    # --- threshold=0 equivalence gate (the core integration check) --------
    torch.manual_seed(0)
    pipeline = _make_pipeline()
    baseline = LlamaOldSelectablePipeline(pipeline)
    speculative = SpeculativeBlockVerifyPipeline(pipeline, gamma=GAMMA, acceptance_threshold=0.0)

    max_diff_seen = 0.0
    for seed in range(5):
        history = _large_velocity_history(seed)
        with torch.no_grad():
            expected = baseline.auto_regressive(history, None)
            actual = speculative.auto_regressive(history, None)

        check(f"seed={seed}: output shape == ({1},{FUT_WINDOW},3)", tuple(actual.shape) == (1, FUT_WINDOW, 3))
        max_abs_diff = (actual - expected).abs().max().item()
        max_diff_seen = max(max_diff_seen, max_abs_diff)
        check(
            f"seed={seed}: threshold=0 output matches baseline (atol=1e-5)",
            torch.allclose(actual, expected, atol=1e-5, rtol=0),
            detail=f"max abs diff {max_abs_diff:.2e}",
        )
        check(
            f"seed={seed}: target_forward_count == fut_window (no draft acceptance)",
            speculative.target_forward_count == FUT_WINDOW,
        )
        check(
            f"seed={seed}: every draft rejected at threshold=0",
            sum(speculative.accepted_per_iteration) == 0,
        )

    print(f"\nmax abs diff across all seeds at threshold=0: {max_diff_seen:.2e} "
          f"(expected inherent BLAS reassociation noise, well under 1e-5 -- see HANDOFF.md)")

    # --- threshold > 0 actually reduces target forwards --------------------
    torch.manual_seed(1)
    speculative_gen = SpeculativeBlockVerifyPipeline(pipeline, gamma=GAMMA, acceptance_threshold=10.0)
    small_history = torch.randn(1, 2, 3) * 0.1
    with torch.no_grad():
        speculative_gen.auto_regressive(small_history, None)
    check(
        "large threshold reduces target_forward_count below fut_window",
        speculative_gen.target_forward_count < FUT_WINDOW,
        detail=f"got {speculative_gen.target_forward_count}",
    )

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll checks passed.")
    if FAILURES:
        print("Failed checks:", *FAILURES, sep="\n  - ")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
