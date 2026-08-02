"""Assembles the real checkpoint-era Llama VP model.

Combines the vendored, unmodified upstream source
(third_party/netllm_upstream/viewport_prediction, see PROVENANCE.md there)
with this project's own weights. Reused by both the 7B integration smoke
(scripts/experiment_phase/speculative/run_llama_7b_speculative_smoke.py)
and the speculative benchmark harness
(scripts/experiment_phase/speculative/run_speculative_benchmark.py) so the
assembly logic -- which exactly mirrors run_old.py's own load_plm ->
peft_model -> set_task_head -> EmbeddingForViewportPrediction order --
exists in exactly one place.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_VP_ROOT = (
    PROJECT_ROOT / "third_party" / "netllm_upstream" / "viewport_prediction"
)

# Base Llama2-7b weights location. Updated by the Task 3 repo cleanup pass
# (docs/REPO_LAYOUT.md) -- keep this the single place that path is defined.
DEFAULT_BASE_MODEL_PATH = Path("/root/llama2-7b-base")

_upstream_on_path = False


def _import_upstream():
    global _upstream_on_path
    if not _upstream_on_path:
        if str(UPSTREAM_VP_ROOT) not in sys.path:
            sys.path.insert(0, str(UPSTREAM_VP_ROOT))
        _upstream_on_path = True


def load_checkpoint_era_model(
    base_model_path: Optional[Path] = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    rank: int = 32,
    fut_window: int = 20,
    embed_size: int = 4096,
    frequency: int = 5,
    dataset: str = "Jin2022",
    checkpoint_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Tuple["torch.nn.Module", bool]:
    """Assemble EmbeddingForViewportPrediction exactly as run_old.py does.

    If `checkpoint_path` is None, the LoRA adapter keeps PEFT's own default
    init (B=0, i.e. zero effect until trained) and the task head keeps
    nn.Linear's default random init -- valid for structural/control-flow
    smoke testing only; never treat its predictions as an accuracy result.
    If `checkpoint_path` is given, the fine-tuned adapter and non-PLM
    modules are strict-loaded from it (missing/unexpected keys must be 0).

    Returns (model, checkpoint_loaded).
    """
    _import_upstream()
    from models.low_rank import peft_model
    from models.old.llama import LlamaTaskHeadModel2
    from models.old.networking_head import SimpleLinearTaskHead
    from models.old.pipeline import EmbeddingForViewportPrediction
    from transformers import LlamaConfig

    if seed is not None:
        torch.manual_seed(seed)

    resolved_base = (
        Path(base_model_path) if base_model_path is not None else DEFAULT_BASE_MODEL_PATH
    )
    if not resolved_base.exists():
        raise FileNotFoundError(f"base Llama2-7b weights not found at {resolved_base}")

    config = LlamaConfig.from_pretrained(str(resolved_base), local_files_only=True)
    base = LlamaTaskHeadModel2.from_pretrained(
        str(resolved_base),
        config=config,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    plm = peft_model(base, "llama", rank)
    plm.set_task_head(SimpleLinearTaskHead(embed_size, 3, fut_window).to(device))

    model = EmbeddingForViewportPrediction(
        plm,
        fut_window=fut_window,
        device=device,
        embed_size=embed_size,
        frequency=frequency,
        using_teaching_forcing=False,
        using_multimodal=False,
        dataset=dataset,
    )

    checkpoint_loaded = False
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found at {checkpoint_path}")
        model.plm.load_adapter(str(checkpoint_path), adapter_name="default")
        model.plm.set_adapter("default")
        state = torch.load(
            checkpoint_path / "modules_except_plm.bin",
            map_location="cpu",
            weights_only=True,
        )
        incompatible = model.embedding_model.modules_except_plm.load_state_dict(
            state, strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict non-PLM load mismatch")
        checkpoint_loaded = True

    model.half().eval()
    return model, checkpoint_loaded
