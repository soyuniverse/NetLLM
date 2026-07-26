from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class DraftOutput:
    coordinates: torch.Tensor
    forward_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetOutput:
    coordinates: torch.Tensor
    forward_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinuousDraftModel(nn.Module, ABC):
    """Interface for continuous-coordinate VP draft models."""

    @abstractmethod
    def forward(
        self,
        history: torch.Tensor,
        steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> DraftOutput:
        raise NotImplementedError
