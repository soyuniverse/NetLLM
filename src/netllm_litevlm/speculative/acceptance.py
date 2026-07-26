from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class AcceptanceResult:
    accepted_prefix_length: int
    first_rejected_index: Optional[int]
    per_step_max_absolute_error: List[float]
    threshold: float


def verify_continuous_prefix(
    draft: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> AcceptanceResult:
    """Accept only the consecutive draft prefix within a coordinate threshold."""
    if draft.shape != target.shape or draft.ndim != 3:
        raise ValueError("draft and target must have identical [B,F,C] shapes")
    if draft.shape[0] != 1:
        raise ValueError("prototype currently preserves the B=1 VP contract")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not torch.isfinite(draft).all() or not torch.isfinite(target).all():
        raise ValueError("draft and target coordinates must be finite")

    errors = torch.max(torch.abs(draft - target), dim=-1).values[0]
    error_values = [float(value) for value in errors.detach().cpu()]
    accepted = 0
    for error in error_values:
        if error > threshold:
            break
        accepted += 1
    first_rejected = None if accepted == len(error_values) else accepted
    return AcceptanceResult(
        accepted_prefix_length=accepted,
        first_rejected_index=first_rejected,
        per_step_max_absolute_error=error_values,
        threshold=float(threshold),
    )
