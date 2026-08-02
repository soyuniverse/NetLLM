from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseSelector, SelectionOutput


class AttentionTopKSelector(BaseSelector):
    """Keep the top-K embeddings by first-decoder-layer attention weight,
    preserving temporal order.

    Importance is the last query position's attention (averaged over
    heads), from a single partial forward through only the first decoder
    layer of the underlying causal LM -- not the full stack. Verified on
    transformers==4.34.1 that a direct `LlamaDecoderLayer.forward(...,
    output_attentions=True)` call, using `llama_model
    ._prepare_decoder_attention_mask(...)` for the causal mask, gives
    attention weights bit-identical to that same layer's output inside a
    full model forward (no forward-hook fallback needed at this pinned
    version).

    Drop-in replacement for RecentKSelector: same BaseSelector /
    SelectionOutput contract, no pipeline changes required. `llama_model`
    is whatever exposes `.layers` (decoder layer list) and
    `._prepare_decoder_attention_mask(...)` -- the base LlamaModel, or
    accessible through PEFT attribute forwarding (e.g. `plm.model`) on the
    assembled checkpoint-era pipeline.
    """

    def __init__(self, k: int, llama_model: nn.Module):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.llama_model = llama_model
        self.first_layer = llama_model.layers[0]
        self.selector_forward_count = 0

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        if embeddings.shape[0] != 1:
            raise ValueError("AttentionTopKSelector currently supports B=1")
        original_length = int(embeddings.shape[1])
        if self.k > original_length:
            raise ValueError(f"k={self.k} exceeds sequence length {original_length}")

        working_mask = attention_mask
        if working_mask is None:
            working_mask = torch.ones(
                (1, original_length), dtype=torch.long, device=embeddings.device
            )

        causal_mask = self.llama_model._prepare_decoder_attention_mask(
            working_mask, (1, original_length), embeddings, 0
        )
        position_ids = torch.arange(
            original_length, device=embeddings.device
        ).unsqueeze(0)

        with torch.no_grad():
            _, attn_weights = self.first_layer(
                embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
            )
        self.selector_forward_count += 1

        # Last query position's attention to every source position,
        # averaged over heads -> one importance score per source position.
        scores = attn_weights[0, :, -1, :].mean(dim=0)  # [L]

        top_indices = torch.topk(scores, self.k).indices
        selected_indices, _ = torch.sort(top_indices)

        return SelectionOutput(
            embeddings=embeddings[:, selected_indices, :],
            attention_mask=(
                None if attention_mask is None else attention_mask[:, selected_indices]
            ),
            selected_indices=selected_indices,
            scores=scores.detach(),
            original_length=original_length,
            selected_length=self.k,
            metadata={
                "selector": type(self).__name__,
                "k": self.k,
                "preserves_order": True,
                "selection_policy": "attention_top_k",
                "selector_forward_count": self.selector_forward_count,
                "context": dict(context) if context is not None else {},
            },
        )
