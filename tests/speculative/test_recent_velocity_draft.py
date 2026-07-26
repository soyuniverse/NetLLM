import torch

from netllm_litevlm.speculative.recent_velocity_draft import (
    RecentVelocityDraft,
)


def test_recent_velocity_extrapolates_deterministically():
    history = torch.tensor([[[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]])
    result = RecentVelocityDraft()(history, steps=3)
    expected = torch.tensor([[[3.0, 7.0], [4.0, 9.0], [5.0, 11.0]]])
    assert torch.equal(result.coordinates, expected)
    assert result.forward_count == 1
    assert result.metadata["learned"] is False
