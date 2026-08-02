from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from netllm_litevlm.selectors import BaseSelector, SelectionOutput

from .base import ContinuousDraftModel
from .recent_velocity_draft import RecentVelocityDraft

PastKeyValues = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def slice_past_key_values(
    past_key_values: PastKeyValues, keep_length: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Truncate a legacy tuple-of-(key,value) KV cache to `keep_length` positions.

    transformers==4.34.1 (pinned for this checkpoint era, see
    docs/experiment_phase/speculative/PHASE_A_DESIGN.md) predates the
    Cache/DynamicCache object and returns this tuple-of-tuples format from
    LlamaModel, each entry shaped [B, num_heads, seq_len, head_dim].
    """
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "unsupported past_key_values type for rollback: "
            f"{type(past_key_values)!r}; expected the legacy tuple-of-"
            "(key,value) format used by transformers==4.34.1"
        )
    return tuple(
        (
            key[:, :, :keep_length, :].contiguous(),
            value[:, :, :keep_length, :].contiguous(),
        )
        for key, value in past_key_values
    )


def _embed_history(old: nn.Module, history: torch.Tensor) -> torch.Tensor:
    tokens = []
    for index in range(history.shape[1]):
        token = old.linear_layer(
            old.conv1d1(history[:, index, :]).view(1, 256)
        ).unsqueeze(1)
        tokens.append(token)
    return torch.cat(tokens, dim=1)


def _embed_step(old: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Embed one [1,1,3] coordinate via the baseline feedback path
    (conv1d1 -> linear_layer -> unsqueeze, no LayerNorm)."""
    return old.linear_layer(old.conv1d1(value).view(1, 256)).unsqueeze(1)


class SpeculativeBlockVerifyPipeline(nn.Module):
    """Block-verified continuous-coordinate speculative decoding.

    Reuses the checkpoint-era embedding/task-head modules exactly as
    LlamaOldSelectablePipeline does, but replaces its per-step
    reprocess-the-whole-sequence loop with a KV-cache incremental loop
    that verifies `gamma` drafted coordinates against the target model in
    a single forward call. See
    docs/experiment_phase/speculative/PHASE_A_DESIGN.md for the design.
    """

    def __init__(
        self,
        pipeline: nn.Module,
        selector: Optional[BaseSelector] = None,
        draft_model: Optional[ContinuousDraftModel] = None,
        gamma: int = 4,
        acceptance_threshold: float = 0.0,
    ):
        super().__init__()
        if pipeline.using_multimodal:
            raise ValueError("only the proven non-multimodal checkpoint is supported")
        if selector is not None and not isinstance(selector, BaseSelector):
            raise TypeError("selector must be BaseSelector or None")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if acceptance_threshold < 0:
            raise ValueError("acceptance_threshold must be non-negative")
        self.pipeline = pipeline
        self.selector = selector
        self.draft_model = draft_model if draft_model is not None else RecentVelocityDraft()
        self.gamma = int(gamma)
        self.acceptance_threshold = float(acceptance_threshold)

        self.last_selection_output: Optional[SelectionOutput] = None
        self.last_trace: Dict[str, Any] = {}
        self.target_forward_count = 0
        self.draft_forward_count = 0
        self.accepted_per_iteration: List[int] = []

    @property
    def embedding_model(self):
        return self.pipeline.embedding_model

    def auto_regressive(
        self, history: torch.Tensor, video_user_position: Any
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("checkpoint-era pipeline requires B=1")

        old = self.embedding_model
        sequence = old.embed_ln(_embed_history(old, history))
        attention_mask = torch.ones(
            sequence.shape[:2], dtype=torch.long, device=sequence.device
        )

        self.last_selection_output = None
        self.target_forward_count = 0
        self.draft_forward_count = 0
        self.accepted_per_iteration = []
        selector_calls = 0
        if self.selector is not None:
            selection = self.selector(
                sequence,
                attention_mask,
                context={
                    "task": "viewport_prediction",
                    "source": "speculative-block",
                    "stage": "initial_history",
                },
            )
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

        # Initial warmup forward: seeds the KV cache and produces the first carry.
        result = old.plm(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.target_forward_count += 1
        cache = result.past_key_values
        cache_len = int(sequence.shape[1])
        carry = old.plm.task_head.task_head(result.hidden_states[-1][:, -1:, :])

        fut_window = old.fut_window_length
        confirmed = [carry]

        while len(confirmed) < fut_window:
            remaining = fut_window - len(confirmed)
            gamma = min(self.gamma, remaining)

            draft_history = torch.cat((history, *confirmed), dim=1)
            draft = self.draft_model(draft_history, steps=gamma)

            chunk_values = [carry] + [
                draft.coordinates[:, i : i + 1, :] for i in range(gamma)
            ]
            chunk_embeds = torch.cat(
                [_embed_step(old, value) for value in chunk_values], dim=1
            )
            full_mask = torch.ones(
                (1, cache_len + chunk_embeds.shape[1]),
                dtype=attention_mask.dtype,
                device=sequence.device,
            )

            result = old.plm(
                inputs_embeds=chunk_embeds,
                attention_mask=full_mask,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            self.target_forward_count += 1
            preds = old.plm.task_head.task_head(result.hidden_states[-1])

            accepted = 0
            for k in range(gamma):
                error = torch.linalg.vector_norm(
                    (preds[:, k, :] - draft.coordinates[:, k, :]).float(), ord=2
                )
                if error.item() > self.acceptance_threshold:
                    break
                accepted += 1
            self.accepted_per_iteration.append(accepted)

            for k in range(accepted):
                confirmed.append(preds[:, k : k + 1, :])

            commit_len = 1 + accepted
            if accepted < gamma:
                bonus = preds[:, accepted : accepted + 1, :]
                confirmed.append(bonus)
                carry = bonus
            else:
                carry = preds[:, gamma : gamma + 1, :]

            cache = slice_past_key_values(result.past_key_values, cache_len + commit_len)
            cache_len += commit_len

        prediction = torch.cat(confirmed, dim=1)
        self.last_trace = {
            "selector_call_count": selector_calls,
            "target_forward_count": self.target_forward_count,
            "draft_forward_count": self.draft_forward_count,
            "accepted_per_iteration": list(self.accepted_per_iteration),
            "gamma": self.gamma,
            "acceptance_threshold": self.acceptance_threshold,
            "cache_reused": True,
            "final_cache_length": cache_len,
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
        return self.pipeline.loss_fct(prediction, future.to(prediction.device))
