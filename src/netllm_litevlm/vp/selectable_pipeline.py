from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from netllm_litevlm.selectors import BaseSelector, SelectionOutput


class SelectablePipeline(nn.Module):
    """Compose an upstream VP Pipeline with one initial-sequence selector.

    The upstream Pipeline instance and all of its modules are reused. This class
    intentionally preserves the verified batch-size-one, feedback, LayerNorm,
    cache, and final-unused-feedback behavior of the upstream autoregressive path.
    """

    def __init__(self, pipeline: nn.Module, selector: Optional[BaseSelector] = None):
        super().__init__()
        if pipeline.using_multimodal:
            raise ValueError("Phase 3A SelectablePipeline supports non-multimodal VP only")
        self.pipeline = pipeline
        self.selector = selector
        self.last_selection_output: Optional[SelectionOutput] = None
        self.last_trace: Dict[str, Any] = {}

    def set_selector(self, selector: Optional[BaseSelector]) -> None:
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be a BaseSelector instance or None")
        self.selector = selector

    def auto_regressive(
        self,
        x: torch.Tensor,
        future: torch.Tensor,
        video_user_position: Any,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(
                "SelectablePipeline preserves the upstream batch-size-one contract; "
                f"got {tuple(x.shape)}"
            )

        seq_len = x.shape[1]
        batch_embeddings = []
        for index in range(seq_len):
            batch_embeddings.append(
                self.pipeline.embed_vp(
                    self.pipeline.conv1d(x[:, index, :]).view(1, 256)
                ).unsqueeze(1)
            )
        x = torch.cat(batch_embeddings, dim=1)

        if self.pipeline.using_multimodal:
            raise RuntimeError("Multimodal selection is outside Phase 3A")

        x = self.pipeline.embed_ln(x)
        attention_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            dtype=torch.long,
            device=self.pipeline.device,
        )

        initial_embeddings = x
        initial_attention_mask = attention_mask
        selector_call_count = 0
        if self.selector is not None:
            selection = self.selector(
                x,
                attention_mask,
                context={
                    "task": "viewport_prediction",
                    "stage": "initial_history",
                    "multimodal": False,
                },
            )
            selector_call_count += 1
            x = selection.embeddings
            attention_mask = selection.attention_mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    dtype=torch.long,
                    device=x.device,
                )
            self.last_selection_output = selection
        else:
            selection = None
            self.last_selection_output = None

        if x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != self.pipeline.embed_size:
            raise ValueError(f"Selector returned invalid embeddings shape: {tuple(x.shape)}")
        if attention_mask.shape != x.shape[:2]:
            raise ValueError(
                "Selector attention mask does not match selected embeddings: "
                f"mask={tuple(attention_mask.shape)}, embeddings={tuple(x.shape)}"
            )

        output_list = []
        sequence_lengths = []
        past_key_values_passed = []
        cache_returned = []
        for _ in range(self.pipeline.fut_window_length):
            sequence_lengths.append(int(x.shape[1]))
            past_key_values_passed.append(False)
            outputs = self.pipeline.plm(
                inputs_embeds=x,
                attention_mask=attention_mask,
            )
            cache_returned.append(outputs.past_key_values is not None)
            output_list.append(outputs.logits)

            feedback_embedding = self.pipeline.embed_vp(
                self.pipeline.conv1d(outputs.logits)
            ).unsqueeze(1)
            x = torch.cat((x, feedback_embedding), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0],
                        1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )

        prediction = torch.cat(output_list, dim=1)
        self.last_trace = {
            "selector_enabled": self.selector is not None,
            "selector_class": None if self.selector is None else type(self.selector).__name__,
            "selector_call_count": selector_call_count,
            "selector_applied_to_feedback": False,
            "initial_sequence_shape_before_selection": list(initial_embeddings.shape),
            "initial_attention_mask_shape_before_selection": list(initial_attention_mask.shape),
            "initial_sequence_shape_after_selection": list(x.shape[:1])
            + [sequence_lengths[0]]
            + [x.shape[2]],
            "selected_length": sequence_lengths[0],
            "sequence_lengths": sequence_lengths,
            "sequence_lengths_after_feedback_append": [length + 1 for length in sequence_lengths],
            "plm_forward_count": len(sequence_lengths),
            "past_key_values_passed": past_key_values_passed,
            "cache_returned": cache_returned,
            "cache_reused": any(past_key_values_passed),
            "feedback_layernorm_applied": False,
            "final_unused_feedback_computed": True,
            "embeddings_same_object": selection is not None
            and selection.embeddings is initial_embeddings,
            "attention_mask_same_object": selection is not None
            and selection.attention_mask is initial_attention_mask,
        }
        return prediction

    def inference(
        self,
        batch: torch.Tensor,
        future: torch.Tensor,
        video_user_info: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.auto_regressive(batch, future, video_user_info)
        ground_truth = future.to(prediction.device)
        return prediction, ground_truth

    def forward(
        self,
        batch: torch.Tensor,
        future: torch.Tensor,
        video_user_info: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inference(batch, future, video_user_info)
