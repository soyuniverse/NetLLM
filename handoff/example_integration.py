#!/usr/bin/env python3
"""Copy-paste integration examples for the selector + speculative-decoding
modules. Three self-contained sections:

  (a) swap in a custom (e.g. patch-selection) selector
  (b) compare speculative decoding on vs. off with the same selector
  (c) load a different adapter checkpoint (only the path changes)

None of these require this specific repo instance's absolute paths --
only `<repo_root>/src` on sys.path, resolved relative to this file, so
this still works after `handoff/` is copied into a different clone of
the same repo. Sections (b) and (c) need a real checkpoint + dataset to
actually execute; without them, running this file just prints what each
section would do (see `if __name__ == "__main__"` at the bottom).
Section (a) prints its structural check with no checkpoint needed.

For a guaranteed-runnable check (no checkpoint, no dataset, ~1 minute),
use `handoff/smoke_test.py` instead.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from netllm_litevlm.selectors import BaseSelector, RecentKSelector, SelectionOutput  # noqa: E402
from netllm_litevlm.speculative import SpeculativeBlockVerifyPipeline  # noqa: E402
from netllm_litevlm.vp.checkpoint_era_runtime import (  # noqa: E402
    DEFAULT_BASE_MODEL_PATH,
    load_checkpoint_era_model,
)
from netllm_litevlm.vp.llama_old_selectable_pipeline import LlamaOldSelectablePipeline  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Custom selector skeleton -- e.g. wrapping a teammate's patch-selection
#     module in the BaseSelector contract (see INTERFACE_SPEC.md §1).
# ---------------------------------------------------------------------------
class PatchSelectionSelector(BaseSelector):
    """Skeleton: replace `_choose_indices` with the real patch-selection
    policy. Everything else (shape bookkeeping, order preservation,
    SelectionOutput construction) is boilerplate every selector needs."""

    def __init__(self, keep_count: int):
        super().__init__()
        if keep_count <= 0:
            raise ValueError("keep_count must be positive")
        self.keep_count = keep_count

    def _choose_indices(
        self, embeddings: torch.Tensor, context: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Replace this with the real patch-importance policy. Must return
        a 1-D LongTensor of ASCENDING indices into embeddings.shape[1] --
        see INTERFACE_SPEC.md §1's time-order-preservation requirement.
        This placeholder just keeps the most recent `keep_count`, i.e. it
        behaves like RecentKSelector until you fill in real logic."""
        length = embeddings.shape[1]
        k = min(self.keep_count, length)
        return torch.arange(length - k, length, dtype=torch.long, device=embeddings.device)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectionOutput:
        self.validate_inputs(embeddings, attention_mask)
        original_length = int(embeddings.shape[1])
        indices = self._choose_indices(embeddings, context)
        return SelectionOutput(
            embeddings=embeddings[:, indices, :],
            attention_mask=None if attention_mask is None else attention_mask[:, indices],
            selected_indices=indices,
            scores=None,
            original_length=original_length,
            selected_length=int(indices.shape[0]),
            metadata={
                "selector": type(self).__name__,
                "preserves_order": True,
                "context": dict(context) if context is not None else {},
            },
        )


def demo_custom_selector() -> None:
    """Structural check only -- no checkpoint needed. Confirms the custom
    selector satisfies the interface contract on a synthetic sequence."""
    embeddings = torch.randn(1, 10, 4096)
    mask = torch.ones(1, 10, dtype=torch.long)
    selector = PatchSelectionSelector(keep_count=3)
    out = selector(embeddings, mask, context={"task": "viewport_prediction"})
    assert out.embeddings.shape == (1, 3, 4096)
    assert torch.equal(out.selected_indices, torch.tensor([7, 8, 9]))
    assert list(out.selected_indices) == sorted(out.selected_indices.tolist()), (
        "selected_indices must be ascending (time-order preserved)"
    )
    print(f"(a) custom selector OK: kept indices {out.selected_indices.tolist()} of 10")


# ---------------------------------------------------------------------------
# (b) Speculative decoding on/off comparison with the SAME selector instance
#     -- this is exactly what run_speculative_benchmark.py does internally.
# ---------------------------------------------------------------------------
def demo_speculative_on_off(checkpoint_path: str, dataset_path: str, device: str = "cuda:0") -> None:
    model, checkpoint_loaded = load_checkpoint_era_model(
        base_model_path=DEFAULT_BASE_MODEL_PATH,
        device=device,
        dtype=torch.float16,
        checkpoint_path=checkpoint_path,
    )
    assert checkpoint_loaded

    selector = RecentKSelector(k=2)  # swap for PatchSelectionSelector(...) freely
    baseline = LlamaOldSelectablePipeline(model, selector=selector)
    speculative = SpeculativeBlockVerifyPipeline(
        model, selector=selector, gamma=8, acceptance_threshold=0.35
    )

    sys.path.insert(0, str(REPO_ROOT / "third_party" / "netllm_upstream" / "viewport_prediction"))
    from config import cfg
    from dataset.load_dataset import create_dataset
    from utils.normalize import denormalize_data, normalize_data

    cfg.dataset["Jin2022"] = dataset_path
    test_dataset = create_dataset(
        "Jin2022", his_window=10, fut_window=20, trim_head=30, trim_tail=60,
        frequency=5, step=15, include=["test"],
    )[0]
    history_np, future_np, info = test_dataset[0]
    history = normalize_data(
        torch.from_numpy(history_np).unsqueeze(0), "Jin2022"
    ).to(device, dtype=torch.float16)
    future = torch.from_numpy(future_np).unsqueeze(0)

    with torch.inference_mode():
        pred_off, _ = baseline.inference(history, future, info) if hasattr(
            baseline, "inference"
        ) else (baseline.auto_regressive(history, info), None)
        pred_on, _ = speculative.inference(history, future, info)

    pred_off_deg = denormalize_data(pred_off.float(), "Jin2022")
    pred_on_deg = denormalize_data(pred_on.float(), "Jin2022")
    print(f"(b) speculative OFF: target_forward_count=20 (baseline always recomputes)")
    print(
        f"(b) speculative ON:  target_forward_count={speculative.target_forward_count}, "
        f"accepted_per_iteration={speculative.accepted_per_iteration}"
    )
    print(f"(b) max |pred diff| (informational, not an equivalence claim at threshold=0.35): "
          f"{(pred_on_deg - pred_off_deg).abs().max().item():.4f} deg")


# ---------------------------------------------------------------------------
# (c) Loading a different adapter checkpoint -- only the path changes.
# ---------------------------------------------------------------------------
def demo_alternate_checkpoint(checkpoint_path: str, device: str = "cuda:0") -> None:
    """Swap in an AdaLoRA (or any other PEFT-adapter) checkpoint by pointing
    at its directory -- same call, no code change needed downstream, AS LONG
    AS `checkpoint_era_runtime.py`'s `peft_model(...)` call was itself
    updated to build an AdaLoraConfig instead of LoraConfig (see
    INTERFACE_SPEC.md §3). This function doesn't care which PEFT method
    produced the checkpoint directory -- `model.plm.load_adapter(...)` is
    the same call either way."""
    model, checkpoint_loaded = load_checkpoint_era_model(
        base_model_path=DEFAULT_BASE_MODEL_PATH,
        device=device,
        dtype=torch.float16,
        checkpoint_path=checkpoint_path,  # <-- only this changes for a new adapter
    )
    assert checkpoint_loaded
    print(f"(c) loaded checkpoint from {checkpoint_path}: strict-load passed")
    return model


if __name__ == "__main__":
    demo_custom_selector()

    example_checkpoint = "/root/NetLLM-assets/checkpoints/try_llama2_7b"
    example_dataset = str(REPO_ROOT.parent / "NetLLM-source" / "viewport_prediction" / "data")
    if Path(example_checkpoint).exists() and Path(example_dataset).exists():
        demo_speculative_on_off(example_checkpoint, example_dataset)
        demo_alternate_checkpoint(example_checkpoint)
    else:
        print(
            "(b)/(c) skipped: no checkpoint/dataset found at the default example "
            f"paths ({example_checkpoint}, {example_dataset}). Call "
            "demo_speculative_on_off(checkpoint_path, dataset_path) / "
            "demo_alternate_checkpoint(checkpoint_path) directly with your own "
            "paths once you have real assets -- see HANDOFF.md's asset table."
        )
