import torch

from netllm_litevlm.selectors import IdentitySelector
from netllm_litevlm.vp.llama_old_selectable_pipeline import (
    LlamaOldSelectablePipeline,
)
from test_llama_old_pipeline_contract import make_pipeline


def test_identity_matches_disabled_exactly():
    torch.manual_seed(1)
    pipeline = make_pipeline()
    history = torch.randn(1, 10, 3)
    disabled = LlamaOldSelectablePipeline(pipeline)
    identity = LlamaOldSelectablePipeline(pipeline, IdentitySelector())
    expected = disabled.auto_regressive(history, (4, 83, 30))
    actual = identity.auto_regressive(history, (4, 83, 30))
    assert torch.equal(expected, actual)
    assert identity.last_selection_output.selected_indices.tolist() == list(range(10))
    assert identity.last_trace["selector_call_count"] == 1
