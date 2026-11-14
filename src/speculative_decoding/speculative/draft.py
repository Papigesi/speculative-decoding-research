"""
Sequential draft-token proposals without KV caching.

This intentionally recomputes the full prefix for every proposal. The goal is
an easy-to-audit reference implementation, not inference speed yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DraftProposal:
    token_ids: Tensor  # [1, k]
    forward_calls: int


class DraftGenerator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @torch.inference_mode()
    def propose(self, prefix_ids: Tensor, *, num_tokens: int) -> DraftProposal:
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 1:
            raise ValueError("prefix_ids must have shape [1, nonempty_length]")
        if num_tokens < 1:
            raise ValueError("num_tokens field must be greater than 0")

        context = prefix_ids
        proposed: list[Tensor] = []

        for _ in range(num_tokens):
            outputs = self.model(
                input_ids=context,
                attention_mask=torch.ones_like(context, dtype=torch.long),
                use_cache=False,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            proposed.append(next_token)
            context = torch.cat((context, next_token), dim=1)

        return DraftProposal(token_ids=torch.cat(proposed, dim=1), forward_calls=num_tokens)
