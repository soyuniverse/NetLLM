from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from .acceptance import AcceptanceResult, verify_continuous_prefix
from .base import ContinuousDraftModel, DraftOutput, TargetOutput


@dataclass
class ContinuousDraftVerifyResult:
    output: torch.Tensor
    draft: DraftOutput
    target: TargetOutput
    verification: AcceptanceResult
    baseline_target_forward_count: int
    control_flow_valid: bool
    speedup_claim_valid: bool


class ContinuousDraftVerify:
    """Continuous VP draft-and-verify control flow.

    The current prototype obtains a complete target trajectory before prefix
    verification. It validates the continuous-coordinate acceptance behavior
    but does not claim target-forward reduction.
    """

    def __init__(
        self,
        draft_model: ContinuousDraftModel,
        target_predictor: Callable[
            [torch.Tensor, int, Optional[Dict[str, Any]]], TargetOutput
        ],
        threshold: float,
        baseline_target_forward_count: int,
    ):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if baseline_target_forward_count <= 0:
            raise ValueError("baseline target forward count must be positive")
        self.draft_model = draft_model
        self.target_predictor = target_predictor
        self.threshold = float(threshold)
        self.baseline_target_forward_count = baseline_target_forward_count

    def run(
        self,
        history: torch.Tensor,
        steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContinuousDraftVerifyResult:
        draft = self.draft_model(history, steps, context)
        target = self.target_predictor(history, steps, context)
        if draft.coordinates.shape != target.coordinates.shape:
            raise ValueError("draft/target output shape mismatch")
        verification = verify_continuous_prefix(
            draft.coordinates, target.coordinates, self.threshold
        )
        accepted = verification.accepted_prefix_length
        output = torch.cat(
            (
                draft.coordinates[:, :accepted, :],
                target.coordinates[:, accepted:, :],
            ),
            dim=1,
        )
        valid = (
            output.shape == target.coordinates.shape
            and bool(torch.isfinite(output).all().item())
            and target.forward_count > 0
            and draft.forward_count > 0
        )
        return ContinuousDraftVerifyResult(
            output=output,
            draft=draft,
            target=target,
            verification=verification,
            baseline_target_forward_count=self.baseline_target_forward_count,
            control_flow_valid=valid,
            speedup_claim_valid=False,
        )
