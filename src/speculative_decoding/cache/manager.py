"""
Minimal KV-cache bookkeeping for single-sequence autoregressive decoding.

The latest sampled token is not cached until it is fed through a forward pass.
I explicitly track that distinction for a later speculative rollback manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheState:
    past_key_values: Any | None = None
    cached_length: int = 0

    def update(self, past_key_values: Any, *, processed_tokens: int) -> None:
        if processed_tokens < 1:
            raise ValueError("processed_tokens must be >= 1")
        if past_key_values is None:
            raise RuntimeError("Model returned no KV cache with use_cache=True")
        self.past_key_values = past_key_values
        self.cached_length += processed_tokens
