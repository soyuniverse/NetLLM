import torch

from netllm_litevlm.speculative.acceptance import verify_continuous_prefix


def test_acceptance_stops_at_first_rejection():
    target = torch.zeros(1, 4, 3)
    draft = target.clone()
    draft[0, 1, 0] = 0.2
    result = verify_continuous_prefix(draft, target, threshold=0.1)
    assert result.accepted_prefix_length == 1
    assert result.first_rejected_index == 1


def test_acceptance_all_steps():
    target = torch.zeros(1, 3, 3)
    draft = torch.full_like(target, 0.05)
    result = verify_continuous_prefix(draft, target, threshold=0.1)
    assert result.accepted_prefix_length == 3
    assert result.first_rejected_index is None
