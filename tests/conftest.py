"""A deterministic causal LM fake for offline algorithm tests (no model download)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class ToyCausalLM(nn.Module):
    """Predict (current_token + 1) % vocab_size with overwhelmingly high confidence.

    past_key_values simply contains the processed token IDs. This deliberately
    checks the same cache input/mask length invariants as a real causal LM.
    """

    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.seen_input_lengths: list[int] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        assert return_dict is True
        self.seen_input_lengths.append(input_ids.shape[1])
        prior = past_key_values
        full_sequence = input_ids if prior is None else torch.cat((prior, input_ids), dim=1)
        assert attention_mask.shape == full_sequence.shape
        assert torch.all(attention_mask == 1)

        preferred_ids = (input_ids + 1) % self.vocab_size
        logits = torch.full(
            (1, input_ids.shape[1], self.vocab_size), -12.0,
            device=input_ids.device,
        )
        logits.scatter_(2, preferred_ids.unsqueeze(-1), 12.0)
        return SimpleNamespace(
            logits=logits,
            past_key_values=full_sequence if use_cache else None,
        )


@pytest.fixture
def toy_model() -> ToyCausalLM:
    return ToyCausalLM()
