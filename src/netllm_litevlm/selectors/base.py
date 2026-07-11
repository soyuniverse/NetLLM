from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SelectionOutput:
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    selected_indices: torch.Tensor
    scores: Optional[torch.Tensor]
    original_length: int
    selected_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSelector(nn.Module, ABC):
    """Base interface for selecting sequence embeddings."""

    @staticmethod
    def validate_inputs(
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must have shape [B,L,E], got {tuple(embeddings.shape)}"
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape [B,L], "
                    f"got {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape != embeddings.shape[:2]:
                raise ValueError(
                    "attention_mask shape must match embeddings [B,L]: "
                    f"mask={tuple(attention_mask.shape)}, "
                    f"embeddings={tuple(embeddings.shape)}"
                )
            if attention_mask.device != embeddings.device:
                raise ValueError(
                    "attention_mask and embeddings must be on the same device"
                )

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        raise NotImplementedError
