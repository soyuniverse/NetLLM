import torch

from netllm_litevlm.selectors import RecentKSelector
from netllm_litevlm.vp.llama_old_selectable_pipeline import (
    LlamaOldSelectablePipeline,
)
from test_llama_old_pipeline_contract import make_pipeline


def test_recent_k_changes_only_initial_history_length():
    wrapped = LlamaOldSelectablePipeline(make_pipeline(), RecentKSelector(4))
    output = wrapped.auto_regressive(torch.randn(1, 10, 3), (4, 83, 30))
    assert output.shape == (1, 3, 3)
    assert wrapped.last_selection_output.selected_indices.tolist() == [6, 7, 8, 9]
    assert wrapped.last_trace["sequence_lengths"] == [4, 5, 6]
    assert wrapped.last_trace["feedback_selector_call_count"] == 0
