import torch

from netllm_litevlm.speculative import (
    ContinuousDraftVerify,
    RecentVelocityDraft,
    TargetOutput,
)


def test_continuous_draft_verify_builds_hybrid_after_rejection():
    history = torch.tensor([[[0.0], [1.0], [2.0]]])

    def target_predictor(history, steps, context):
        del history, context
        return TargetOutput(
            coordinates=torch.tensor([[[3.0], [4.25], [5.25]]]),
            forward_count=20,
        )

    prototype = ContinuousDraftVerify(
        RecentVelocityDraft(), target_predictor, threshold=0.1,
        baseline_target_forward_count=20,
    )
    result = prototype.run(history, steps=3)
    assert result.verification.accepted_prefix_length == 1
    assert result.verification.first_rejected_index == 1
    assert torch.equal(
        result.output, torch.tensor([[[3.0], [4.25], [5.25]]])
    )
    assert result.draft.forward_count == 1
    assert result.target.forward_count == 20
    assert result.baseline_target_forward_count == 20
    assert result.control_flow_valid is True
    assert result.speedup_claim_valid is False
