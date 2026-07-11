from typing import Any, Dict, Optional

import torch

from .base import BaseSelector, SelectionOutput


class IdentitySelector(BaseSelector):
    """Return the input sequence without pruning, reordering, or mutation."""

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        sequence_length = int(embeddings.shape[1])
        selected_indices = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=embeddings.device,
        )
        return SelectionOutput(
            embeddings=embeddings,
            attention_mask=attention_mask,
            selected_indices=selected_indices,
            scores=None,
            original_length=sequence_length,
            selected_length=sequence_length,
            metadata={
                "selector": type(self).__name__,
                "preserves_order": True,
                "context": dict(context) if context is not None else {},
            },
        )
