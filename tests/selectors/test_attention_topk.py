"""AttentionTopKSelector gate tests. GPU is occupied by a real benchmark
run this session, so this uses a tiny real (CPU) LlamaModel -- small
enough to run fast on CPU while still exercising the genuine
transformers==4.34.1 LlamaDecoderLayer.forward(output_attentions=True)
code path, not a hand-rolled approximation of attention.
"""

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from netllm_litevlm.selectors import AttentionTopKSelector, IdentitySelector

SEQ_LEN = 10
EMBED_SIZE = 16


def _make_llama_model(seed: int) -> LlamaModel:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=EMBED_SIZE,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaModel(config).eval()


def test_k_equals_length_matches_identity_selector():
    llama_model = _make_llama_model(seed=0)
    torch.manual_seed(1)
    embeddings = torch.randn(1, SEQ_LEN, EMBED_SIZE)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

    attention_selector = AttentionTopKSelector(k=SEQ_LEN, llama_model=llama_model)
    identity_selector = IdentitySelector()

    with torch.no_grad():
        attention_output = attention_selector(embeddings, attention_mask, {"sample": 0})
    identity_output = identity_selector(embeddings, attention_mask, {"sample": 0})

    assert torch.equal(attention_output.embeddings, identity_output.embeddings)
    assert torch.equal(attention_output.attention_mask, identity_output.attention_mask)
    assert torch.equal(attention_output.selected_indices, identity_output.selected_indices)
    assert attention_output.original_length == identity_output.original_length
    assert attention_output.selected_length == identity_output.selected_length
    assert attention_selector.selector_forward_count == 1


def test_k_less_than_length_matches_hook_captured_attention_top_k():
    llama_model = _make_llama_model(seed=2)
    first_layer = llama_model.layers[0]
    torch.manual_seed(3)
    embeddings = torch.randn(1, SEQ_LEN, EMBED_SIZE)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    k = 4

    # Independent capture: a forward hook on the SAME first layer, driven
    # through the selector's own call, so the "expected" top-K here is
    # derived from a mechanism the selector implementation doesn't control.
    captured = {}

    def _hook(module, args, kwargs, output):
        captured["attn_weights"] = output[1]

    handle = first_layer.register_forward_hook(_hook, with_kwargs=True)
    try:
        selector = AttentionTopKSelector(k=k, llama_model=llama_model)
        with torch.no_grad():
            output = selector(embeddings, attention_mask, {"sample": 1})
    finally:
        handle.remove()

    assert "attn_weights" in captured
    hook_scores = captured["attn_weights"][0, :, -1, :].mean(dim=0)
    expected_indices, _ = torch.sort(torch.topk(hook_scores, k).indices)

    assert output.embeddings.shape == (1, k, EMBED_SIZE)
    assert output.attention_mask.shape == (1, k)
    assert output.selected_length == k
    assert output.original_length == SEQ_LEN
    assert torch.equal(output.selected_indices, expected_indices)
    assert torch.equal(output.embeddings, embeddings[:, expected_indices, :])
    # temporal order preserved: indices strictly increasing
    assert torch.equal(output.selected_indices, torch.sort(output.selected_indices).values)
    assert selector.selector_forward_count == 1


def test_rejects_invalid_k():
    llama_model = _make_llama_model(seed=4)
    try:
        AttentionTopKSelector(k=0, llama_model=llama_model)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        AttentionTopKSelector(k=True, llama_model=llama_model)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    selector = AttentionTopKSelector(k=SEQ_LEN + 1, llama_model=llama_model)
    embeddings = torch.zeros(1, SEQ_LEN, EMBED_SIZE)
    try:
        selector(embeddings)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
