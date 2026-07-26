from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from netllm_litevlm.selectors import BaseSelector, SelectionOutput


class LlamaOldSelectablePipeline(nn.Module):
    """Composition wrapper for the checkpoint-era VP pipeline.

    The supplied ``EmbeddingForViewportPrediction`` instance and every trained
    module it owns are reused. Selection is applied exactly once to the
    LayerNorm-normalized initial history embeddings.
    """

    def __init__(
        self, pipeline: nn.Module, selector: Optional[BaseSelector] = None
    ):
        super().__init__()
        if pipeline.using_multimodal:
            raise ValueError("only the proven non-multimodal checkpoint is supported")
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be BaseSelector or None")
        self.pipeline = pipeline
        self.selector = selector
        self.last_selection_output: Optional[SelectionOutput] = None
        self.last_trace: Dict[str, Any] = {}
        self._selector_events = None

    @property
    def embedding_model(self):
        return self.pipeline.embedding_model

    def set_selector(self, selector: Optional[BaseSelector]) -> None:
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be BaseSelector or None")
        self.selector = selector

    def selector_elapsed_ms(self) -> float:
        if self._selector_events is None:
            return 0.0
        start, end = self._selector_events
        return float(start.elapsed_time(end))

    def auto_regressive(
        self, history: torch.Tensor, video_user_position: Any
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("checkpoint-era pipeline requires B=1")

        old = self.embedding_model
        embeddings = []
        for index in range(history.shape[1]):
            token = old.linear_layer(
                old.conv1d1(history[:, index, :]).view(1, 256)
            ).unsqueeze(1)
            embeddings.append(token)
        sequence = torch.cat(embeddings, dim=1)
        sequence = old.embed_ln(sequence)
        initial_sequence = sequence
        attention_mask = torch.ones(
            sequence.shape[:2],
            dtype=torch.long,
            device=sequence.device,
        )

        self.last_selection_output = None
        self._selector_events = None
        selector_calls = 0
        if self.selector is not None:
            if sequence.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            selection = self.selector(
                sequence,
                attention_mask,
                context={
                    "task": "viewport_prediction",
                    "source": "checkpoint-era",
                    "stage": "initial_history",
                },
            )
            if sequence.is_cuda:
                end.record()
                self._selector_events = (start, end)
            selector_calls = 1
            sequence = selection.embeddings
            attention_mask = selection.attention_mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    sequence.shape[:2],
                    dtype=torch.long,
                    device=sequence.device,
                )
            self.last_selection_output = selection

        if sequence.shape[0] != 1 or sequence.shape[2] != old.embed_size:
            raise ValueError("selector returned an invalid embedding shape")
        if tuple(attention_mask.shape) != tuple(sequence.shape[:2]):
            raise ValueError("selector attention mask shape mismatch")

        outputs = []
        sequence_lengths = []
        for _ in range(old.fut_window_length):
            sequence_lengths.append(int(sequence.shape[1]))
            result = old.plm(
                inputs_embeds=sequence,
                attention_mask=attention_mask,
            )
            outputs.append(result.logits)
            feedback = old.linear_layer(old.conv1d1(result.logits)).unsqueeze(1)
            sequence = torch.cat((sequence, feedback), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )

        prediction = torch.cat(outputs, dim=1)
        selected_length = sequence_lengths[0]
        self.last_trace = {
            "selector_call_count": selector_calls,
            "feedback_selector_call_count": 0,
            "initial_history_shape": list(history.shape),
            "initial_embedding_shape": list(initial_sequence.shape),
            "selected_length": selected_length,
            "sequence_lengths": sequence_lengths,
            "plm_forward_count": len(sequence_lengths),
            "processed_sequence_length_sum": sum(sequence_lengths),
            "cache_reused": False,
            "feedback_layernorm_applied": False,
            "final_unused_feedback_computed": True,
            "attention_mask_final_shape": list(attention_mask.shape),
            "prediction_shape": list(prediction.shape),
        }
        return prediction

    def inference(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        video_user_position: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.auto_regressive(history, video_user_position)
        return prediction, future.to(prediction.device)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        video_user_position: Any,
    ) -> torch.Tensor:
        prediction = self.auto_regressive(history, video_user_position)
        return self.pipeline.loss_fct(
            prediction, future.to(prediction.device)
        )
