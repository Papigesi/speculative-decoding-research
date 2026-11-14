"""Wall-clock inference timings with explicit synchronization on CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass(frozen=True) # Immutable dataclass for reliable inference metrics
class InferenceMetrics:
    elapsed_seconds: float
    generated_tokens: int
    forward_calls: int
    peak_gpu_memory_bytes: int | None

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


class MetricsCollector:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.start_time: float | None = None

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        self._synchronize()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = perf_counter()

    def finish(self, *, generated_tokens: int, forward_calls: int) -> InferenceMetrics:
        if self.start_time is None:
            raise RuntimeError("Call MetricsCollector.start() before finish()")
        
        self._synchronize()
        elapsed = perf_counter() - self.start_time # Elapsed seconds from start
        peak = (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        )

        return InferenceMetrics(
            elapsed_seconds=elapsed,
            generated_tokens=generated_tokens,
            forward_calls=forward_calls,
            peak_gpu_memory_bytes=peak,
        )
