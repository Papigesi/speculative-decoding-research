from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection

import torch
from torch import Tensor, nn

from speculative_decoding.cache.manager import CacheState
from speculative_decoding.decoding.sampling import SamplingConfig, select_next_token
from speculative_decoding.metrics.collector import InferenceMetrics, MetricsCollector


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    use_cache: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens field must be greater than 0")


@dataclass(frozen=True)
class GenerationResult:
    sequences: Tensor            # [1, prompt_length + generated_length]
    new_token_ids: Tensor         # [1, generated_length]
    stop_reason: str              # 'eos' or 'length'
    metrics: InferenceMetrics
    cached_length: int           # 0 with use_cache=False


class AutoregressiveDecoder:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
        *,
        eos_token_id: int | Collection[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> GenerationResult:
        
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("v0.1 supports unpadded input_ids with shape [1, seq_len>=1]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")

        device = next(self.model.parameters()).device
        if input_ids.device != device:
            raise ValueError(f"input_ids are on {input_ids.device}, but model is on {device}")

        if eos_token_id is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = {int(token) for token in eos_token_id}

        sequences = input_ids.clone()
        original_length = sequences.shape[1]
        cache_state = CacheState()
        collector = MetricsCollector(device)
        forward_calls = 0
        reason = "length"
        
        collector.start()
        for _ in range(config.max_new_tokens):
            if config.use_cache and cache_state.past_key_values is not None:
                # Only the final, not-yet-cached token should be processed.
                model_input = sequences[:, -1:]
                assert cache_state.cached_length == sequences.shape[1] - 1
            else:
                # Prefill the entire prompt; or recompute the full prefix without KV cache.
                model_input = sequences

            model_kwargs: dict = {
                "input_ids": model_input,
                "attention_mask": torch.ones_like(sequences, dtype=torch.long),
                "use_cache": config.use_cache,
                "return_dict": True,
            }
            if config.use_cache and cache_state.past_key_values is not None:
                model_kwargs["past_key_values"] = cache_state.past_key_values

            outputs = self.model(**model_kwargs)
            forward_calls += 1
            if config.use_cache:
                cache_state.update(
                    outputs.past_key_values, processed_tokens=model_input.shape[1]
                )

            last_logits = outputs.logits[:, -1, :]
            next_token = select_next_token(
                last_logits, config.sampling, generator=generator
            )
            sequences = torch.cat((sequences, next_token), dim=-1)

            # CPU transfer intentionally occurs only once per generated token.
            if int(next_token.item()) in eos_ids:
                reason = "eos"
                break

        generated = sequences[:, original_length:]
        metrics = collector.finish(
            generated_tokens=generated.shape[1], forward_calls=forward_calls
        )
        
        return GenerationResult(
            sequences=sequences,
            new_token_ids=generated,
            stop_reason=reason,
            metrics=metrics,
            cached_length=cache_state.cached_length,
        )
