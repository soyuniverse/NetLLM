from typing import Any, Dict, Optional

import torch

from .base import ContinuousDraftModel, DraftOutput


class RecentVelocityDraft(ContinuousDraftModel):
    """Deterministic constant-velocity extrapolation in VP coordinate space."""

    def forward(
        self,
        history: torch.Tensor,
        steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> DraftOutput:
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("history must have shape [1,H,C]")
        if history.shape[1] < 2:
            raise ValueError("recent velocity requires at least two history steps")
        if steps <= 0:
            raise ValueError("steps must be positive")
        velocity = history[:, -1, :] - history[:, -2, :]
        offsets = torch.arange(
            1, steps + 1, device=history.device, dtype=history.dtype
        ).view(1, steps, 1)
        coordinates = history[:, -1:, :] + offsets * velocity.unsqueeze(1)
        return DraftOutput(
            coordinates=coordinates,
            forward_count=1,
            metadata={
                "draft": type(self).__name__,
                "deterministic": True,
                "learned": False,
                "context": dict(context) if context is not None else {},
            },
        )
