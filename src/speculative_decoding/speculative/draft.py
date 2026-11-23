from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from speculative_decoding.decoding.sampling import SamplingConfig, logits_to_probs


@dataclass(frozen=True)
class DraftProposal:
    token_ids: Tensor  # [1, k]
    forward_calls: int


@dataclass(frozen=True)
class SamplingDraftProposal:
    token_ids: Tensor  # [1, k]
    logprobs: Tensor  # [1, k, vocab] full q_i needed on rejection
    forward_calls: int


class DraftGenerator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @staticmethod
    def _validate(prefix_ids: Tensor, num_tokens: int) -> None:
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 1:
            raise ValueError("prefix_ids must have shape [1, nonempty_length]")
        if prefix_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("prefix_ids must contain integer token IDs")
        if num_tokens < 1:
            raise ValueError("num_tokens field must be greater than 0")

    @torch.inference_mode()
    def propose(self, prefix_ids: Tensor, *, num_tokens: int) -> DraftProposal:
        self._validate(prefix_ids, num_tokens)
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

        @torch.inference_mode()
        def propose_with_sampling(
            self, 
            prefix_ids: Tensor, 
            *, 
            num_tokens: int,
            sampling: SamplingConfig,
            generator: torch.Generator | None = None
            ) -> SamplingDraftProposal:

            self._validate(prefix_ids, num_tokens)
            if sampling.mode != "sample":
                return ValueError("propose_with_sampling only supports SamplingConfig(mode='sample').")

            context = prefix_ids
            tokens: list[Tensor] = []
            distributions: list[Tensor] = []

            for _ in range(num_tokens):
                outputs = self.model(
                    input_ids=context,
                    attention_mask=torch.ones_like(context, dtype=torch.long),
                    use_cache=False,
                    return_dict=True,
                )
                probs = logits_to_probs(outputs.logits[:, -1, :], sampling)
                token = torch.multinomial(probs, num_samples=1, generator=generator)
                distribution.append(probs.unsqueeze(1))
                tokens.append(token)
                context = torch.cat((context, token), dim=1)

            return SamplingDraftProposal(
                token_ids=torch.cat(tokens, dim=1),
                logprobs=torch.cat(distributions, dim=1),
                forward_calls=num_tokens
            )
