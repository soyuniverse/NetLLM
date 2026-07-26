import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass(frozen=True)
class RuntimeBenchmarkResult:
    latency_median_ms: float
    latency_p95_ms: float
    peak_allocated_mib: Optional[float]
    peak_reserved_mib: Optional[float]
    repetitions: int
    warmup_repetitions: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_callable(
    operation: Callable[[], Any],
    repetitions: int = 20,
    warmup_repetitions: int = 5,
    device: Optional[torch.device] = None,
) -> RuntimeBenchmarkResult:
    if repetitions <= 0 or warmup_repetitions < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")

    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    use_cuda = resolved_device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")

    for _ in range(warmup_repetitions):
        operation()
    if use_cuda:
        torch.cuda.synchronize(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    elapsed_ms = []
    with torch.inference_mode():
        for _ in range(repetitions):
            if use_cuda:
                torch.cuda.synchronize(resolved_device)
            started = time.perf_counter()
            operation()
            if use_cuda:
                torch.cuda.synchronize(resolved_device)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    allocated = None
    reserved = None
    if use_cuda:
        allocated = torch.cuda.max_memory_allocated(resolved_device) / (1024.0 ** 2)
        reserved = torch.cuda.max_memory_reserved(resolved_device) / (1024.0 ** 2)

    return RuntimeBenchmarkResult(
        latency_median_ms=float(statistics.median(elapsed_ms)),
        latency_p95_ms=float(_percentile(elapsed_ms, 0.95)),
        peak_allocated_mib=None if allocated is None else float(allocated),
        peak_reserved_mib=None if reserved is None else float(reserved),
        repetitions=repetitions,
        warmup_repetitions=warmup_repetitions,
    )


def benchmark_selector(
    selector,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    repetitions: int = 100,
    warmup_repetitions: int = 10,
) -> RuntimeBenchmarkResult:
    return benchmark_callable(
        lambda: selector(embeddings, attention_mask),
        repetitions=repetitions,
        warmup_repetitions=warmup_repetitions,
        device=embeddings.device,
    )
