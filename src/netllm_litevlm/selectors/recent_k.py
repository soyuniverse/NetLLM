from typing import Any, Dict, Optional

import torch

from .base import BaseSelector, SelectionOutput


class RecentKSelector(BaseSelector):
    """Keep the most recent ``k`` embeddings without changing their order."""

    def __init__(self, k: int):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        original_length = int(embeddings.shape[1])
        if self.k > original_length:
            raise ValueError(
                f"k={self.k} exceeds sequence length {original_length}"
            )

        start = original_length - self.k
        selected_indices = torch.arange(
            start,
            original_length,
            dtype=torch.long,
            device=embeddings.device,
        )
        return SelectionOutput(
            embeddings=embeddings[:, start:, :],
            attention_mask=(
                None if attention_mask is None else attention_mask[:, start:]
            ),
            selected_indices=selected_indices,
            scores=None,
            original_length=original_length,
            selected_length=self.k,
            metadata={
                "selector": type(self).__name__,
                "k": self.k,
                "preserves_order": True,
                "selection_policy": "most_recent",
                "context": dict(context) if context is not None else {},
            },
        )
