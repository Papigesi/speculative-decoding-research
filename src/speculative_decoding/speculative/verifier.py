"""
Check all draft tokens in a single full-prefix target forward pass.

For a prefix of length L and k draft tokens, the target's logits at positions
L-1 ... L+k-2 predict the draft tokens and position L+k-1 predicts a bonus token.
The -1 offset matters: logits at token i predict the next token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GreedyVerification:
    target_token_ids: Tensor  # [1, k], target's argmax at each draft position
    bonus_token_id: Tensor    # [1, 1], target's argmax after all k draft tokens


class TargetVerifier:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @torch.inference_mode()
    def verify(self, prefix_ids: Tensor, draft_token_ids: Tensor) -> GreedyVerification:
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 1:
            raise ValueError("prefix_ids must have shape [1, nonempty_length]")
        if (draft_token_ids.ndim != 2 or draft_token_ids.shape[0] != 1
                or draft_token_ids.shape[1] < 1):
            raise ValueError("draft_token_ids must have shape [1, k>=1]")
        if prefix_ids.device != draft_token_ids.device:
            raise ValueError("prefix and draft tokens must be on the same device")

        prefix_len = prefix_ids.shape[1]
        k = draft_token_ids.shape[1]
        full_input = torch.cat((prefix_ids, draft_token_ids), dim=1)
        outputs = self.model(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input, dtype=torch.long),
            use_cache=False,
            return_dict=True,
        )
        # [1, k+1, vocab]: target distributions for y1..yk, then bonus.
        relevant_logits = outputs.logits[:, prefix_len - 1 : prefix_len + k, :]
        if relevant_logits.shape[1] != k + 1:
            raise RuntimeError("Target returned insufficient logits for verification")
        choices = relevant_logits.argmax(dim=-1)
        return GreedyVerification(
            target_token_ids=choices[:, :k],
            bonus_token_id=choices[:, k : k + 1],
        )
