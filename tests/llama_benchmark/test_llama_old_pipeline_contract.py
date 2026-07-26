from types import SimpleNamespace

import torch
import torch.nn as nn

from netllm_litevlm.vp.llama_old_selectable_pipeline import (
    LlamaOldSelectablePipeline,
)


class DummyPLM(nn.Module):
    def forward(self, inputs_embeds, attention_mask):
        assert attention_mask.shape == inputs_embeds.shape[:2]
        logits = torch.tanh(inputs_embeds[:, -1:, :3])
        return SimpleNamespace(logits=logits, past_key_values=None)


def make_pipeline():
    old = nn.Module()
    old.plm = DummyPLM()
    old.conv1d1 = nn.Sequential(nn.Conv1d(1, 256, 3), nn.Flatten())
    old.linear_layer = nn.Linear(256, 8)
    old.embed_ln = nn.LayerNorm(8)
    old.embed_size = 8
    old.fut_window_length = 3
    pipeline = nn.Module()
    pipeline.embedding_model = old
    pipeline.using_multimodal = False
    pipeline.loss_fct = nn.MSELoss()
    return pipeline


def test_feedback_and_cache_contract():
    wrapped = LlamaOldSelectablePipeline(make_pipeline())
    pred = wrapped.auto_regressive(torch.randn(1, 4, 3), (4, 83, 30))
    assert pred.shape == (1, 3, 3)
    assert wrapped.last_trace["sequence_lengths"] == [4, 5, 6]
    assert wrapped.last_trace["processed_sequence_length_sum"] == 15
    assert wrapped.last_trace["cache_reused"] is False
    assert wrapped.last_trace["feedback_selector_call_count"] == 0
