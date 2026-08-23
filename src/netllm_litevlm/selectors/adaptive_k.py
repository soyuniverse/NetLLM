from typing import Any, Dict, Optional

import torch

from .base import BaseSelector, SelectionOutput

# Matches third_party/netllm_upstream/viewport_prediction/utils/normalize.py's
# normalize_data/denormalize_data exactly (roll/pitch/yaw scaled by
# 180/90/180 respectively) -- replicated here as a fixed, dataset-
# independent affine transform rather than importing upstream, since
# denormalize_data itself ignores its `dataset` argument (same formula
# for every dataset this project uses). Needed because the `history`
# tensor the checkpoint-era pipelines pass into a selector's context is
# already normalize_data()-normalized (roughly [-1,1] per channel, the
# scale `conv1d1`/`linear_layer` expect), not the raw-degree scale
# TAIL_ANALYSIS.md's velocity thresholds are defined in.
_DENORMALIZE_SCALE_DEG = torch.tensor([180.0, 90.0, 180.0])


def _wrapped_abs_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a-b| accounting for +-180/360 wraparound (roll/yaw); matches
    scripts/experiment_phase/speculative/tail_analysis.py's
    wrapped_abs_diff_deg exactly (torch instead of numpy)."""
    raw = torch.remainder(torch.abs(a - b), 360.0)
    return torch.minimum(raw, 360.0 - raw)


def history_motion_speed(history: torch.Tensor) -> float:
    """Average per-step motion speed in degrees/step, over a [1,H,3]
    raw-degree (roll, pitch, yaw) history tensor.

    Identical definition to `tail_analysis.py`'s `motion_stats()`: the
    mean absolute step-to-step difference (wrap-corrected), averaged
    first over the 3 channels then over the H-1 steps. Using the same
    formula here (not a proxy computed from embeddings) is required so
    an AdaptiveKSelector threshold means the same thing as the
    TAIL_ANALYSIS.md thresholds it's derived from.
    """
    if history.ndim != 3 or history.shape[0] != 1 or history.shape[2] != 3:
        raise ValueError(f"history must have shape [1,H,3], got {tuple(history.shape)}")
    raw_history = history[0].to(dtype=torch.float32)  # [H,3]
    if raw_history.shape[0] < 2:
        return 0.0
    step_diffs = _wrapped_abs_diff_deg(raw_history[1:], raw_history[:-1])  # [H-1,3]
    velocity_per_step = step_diffs.mean(dim=1)  # [H-1]
    return float(velocity_per_step.mean().item())


class AdaptiveKSelector(BaseSelector):
    """Widens the RecentK history window when recent motion is fast/
    variable, narrows it when motion is slow -- the narrowly-scoped next
    step TAIL_ANALYSIS.md's "Suggested next work" section recommends: the
    degraded-sample tail under plain RecentK-2 is drawn from a high-
    motion-variance regime (top-5%-worst samples average 2.16x the
    motion speed of the rest, Mann-Whitney p=3.0e-24), and 100% of that
    tail is attributable to the selector, not speculative decoding -- so
    the selector's own history-length choice is where an intervention
    belongs, not the draft/threshold layer.

    Requires `context["history"]` to be populated with the (unembedded)
    `[1, H, 3]` roll/pitch/yaw history tensor exactly as
    `auto_regressive` receives it -- i.e. `normalize_data()`-normalized
    (roughly `[-1,1]` per channel), not raw degrees. This selector
    denormalizes internally (`_DENORMALIZE_SCALE_DEG`, the fixed inverse
    of `normalize_data`) before computing motion speed, so its threshold
    units are real degrees/step, matching TAIL_ANALYSIS.md. The context
    key addition was made to `LlamaOldSelectablePipeline.auto_regressive`
    and `SpeculativeBlockVerifyPipeline.auto_regressive` for this
    selector; every other existing selector ignores the extra context key,
    so the addition is backward compatible.

    `v_low`/`v_high` (degrees/step) are quantiles of the degraded-sample
    group's own motion-speed distribution
    (`results/speculative/consolidated/tail_analysis_stats.json`
    `top5pct_samples[*].avg_velocity_deg_per_step`, n=84, mean 3.49,
    p10=1.93, p25=2.41, median=3.16, p75=4.44, p90=5.22) -- not arbitrary
    constants. Two candidate quantile pairs were smoke-tested (see
    `docs/experiment_phase/analysis/ADAPTIVE_K_RESULTS.md`) before one was
    selected for the full-scale run.

    Mapping: `avg_velocity <= v_low` -> `k_low` (default 2, matches
    RecentKSelector(2), the config used everywhere else in this project);
    `v_low < avg_velocity <= v_high` -> `k_mid` (default 4); `avg_velocity
    > v_high` -> `k_high` (default 10 = full original history length for
    this task's `his_window=10`, i.e. functionally identical to
    IdentitySelector's output for this pipeline).
    """

    def __init__(
        self,
        v_low: float,
        v_high: float,
        k_low: int = 2,
        k_mid: int = 4,
        k_high: int = 10,
    ):
        super().__init__()
        if not (v_low < v_high):
            raise ValueError(f"v_low must be < v_high, got v_low={v_low}, v_high={v_high}")
        for name, k in (("k_low", k_low), ("k_mid", k_mid), ("k_high", k_high)):
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not (k_low <= k_mid <= k_high):
            raise ValueError(
                f"expected k_low <= k_mid <= k_high, got {k_low}, {k_mid}, {k_high}"
            )
        self.v_low = float(v_low)
        self.v_high = float(v_high)
        self.k_low = k_low
        self.k_mid = k_mid
        self.k_high = k_high
        self.last_avg_velocity: Optional[float] = None
        self.last_k: Optional[int] = None

    def _select_k(self, avg_velocity: float, original_length: int) -> int:
        if avg_velocity <= self.v_low:
            k = self.k_low
        elif avg_velocity <= self.v_high:
            k = self.k_mid
        else:
            k = self.k_high
        return min(k, original_length)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        if context is None or context.get("history") is None:
            raise ValueError(
                "AdaptiveKSelector requires context['history'] (the "
                "normalize_data()-normalized [1,H,3] roll/pitch/yaw history "
                "tensor passed to auto_regressive) to compute motion speed"
            )
        history = context["history"]
        original_length = int(embeddings.shape[1])
        if history.shape[1] != original_length:
            raise ValueError(
                "context['history'] length must match embeddings sequence "
                f"length: history={history.shape[1]}, embeddings={original_length}"
            )

        scale = _DENORMALIZE_SCALE_DEG.to(device=history.device, dtype=torch.float32)
        raw_degree_history = history.to(dtype=torch.float32) * scale
        avg_velocity = history_motion_speed(raw_degree_history)
        k = self._select_k(avg_velocity, original_length)
        self.last_avg_velocity = avg_velocity
        self.last_k = k

        start = original_length - k
        selected_indices = torch.arange(
            start, original_length, dtype=torch.long, device=embeddings.device
        )
        return SelectionOutput(
            embeddings=embeddings[:, start:, :],
            attention_mask=(
                None if attention_mask is None else attention_mask[:, start:]
            ),
            selected_indices=selected_indices,
            scores=None,
            original_length=original_length,
            selected_length=k,
            metadata={
                "selector": type(self).__name__,
                "k": k,
                "preserves_order": True,
                "selection_policy": "adaptive_recent_k_by_motion_speed",
                "avg_velocity_deg_per_step": avg_velocity,
                "v_low": self.v_low,
                "v_high": self.v_high,
                "context": {k_: v_ for k_, v_ in context.items() if k_ != "history"},
            },
        )
