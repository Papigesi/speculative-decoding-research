"""
Sampling policies shared by autoregressive and speculative decoders.

Speculative sampling must compare p and q after applying the same
sampling-policy transformations to the target and draft logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingConfig:
    mode: Literal["greedy", "sample"] = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature == 0.0:
            self.temperature = 1.0
            print("Temperature value cannot be 0.0, that's why value converted to 1.0")
        
        if self.mode not in ("greedy", "sample"):
            raise ValueError("mode must be 'greedy' or 'sample'")
        if not (self.temperature < float("inf")):
            raise ValueError("temperature must be finite")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be integer with greater than 0 or None")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")


def logits_to_probs(logits: Tensor, config: SamplingConfig) -> Tensor:
    """
    Return normalized probabilities after temperature, top-k and top-p.

    Accepts logits of shape [..., vocab]. Converts to float32 to reduce
    numerical issues with half precision inference.
    """
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must end with a nonempty vocab dimension")

    scores = logits.float() / config.temperature
    vocab_size = scores.shape[-1]

    if config.top_k is not None and config.top_k < vocab_size:
        threshold = torch.topk(scores, config.top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)

    if config.top_p < 1.0:
        # Keep the smallest descending-probability prefix with mass >= top_p.
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # 'Before this token' cumulative mass is still below top_p.
        to_remove = cumulative - sorted_probs >= config.top_p
        sorted_scores = sorted_scores.masked_fill(to_remove, -torch.inf)
        scores = torch.empty_like(scores).scatter(-1, sorted_indices, sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all():
        raise ValueError("non-finite probability distribution; check model logits")
    
    return probs


def select_next_token(
    logits: Tensor,
    config: SamplingConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per distribution, keeping a trailing size-1 dimension.

    Expected common input: [batch, vocab], output: [batch, 1].
    """
    if logits.ndim != 2:
        raise ValueError("select_next_token expects [batch, vocab] logits")
    if config.mode == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = logits_to_probs(logits, config)
    return torch.multinomial(probs, num_samples=1, generator=generator)
