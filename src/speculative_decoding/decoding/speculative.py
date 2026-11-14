"""
Correctness-first greedy speculative decoding, no KV cache.

One prompt (batch=1), same vocabulary/token IDs for draft and target.
I deliberately use full-prefix forward passes so any mismatch with the
autoregressive greedy baseline can be investigated without cache state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import torch
from torch import Tensor, nn

from speclab.metrics.collector import InferenceMetrics, MetricsCollector
from speclab.speculative.acceptance import accept_greedy
from speclab.speculative.draft import DraftGenerator
from speclab.speculative.verifier import TargetVerifier


@dataclass(frozen=True)
class SpeculativeGreedyConfig:
    max_new_tokens: int = 64
    draft_k: int = 4

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.draft_k < 1:
            raise ValueError("draft_k must be >= 1")


@dataclass(frozen=True)
class SpeculativeGreedyResult:
    sequences: Tensor
    new_token_ids: Tensor
    stop_reason: str
    metrics: InferenceMetrics
    target_forward_calls: int
    draft_forward_calls: int
    rounds: int
    drafted_tokens: int
    accepted_draft_tokens: int
    rejection_rounds: int
    bonus_tokens: int

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_draft_tokens / self.drafted_tokens
                if self.drafted_tokens else 0.0)


class SpeculativeGreedyDecoder:
    def __init__(self, target_model: nn.Module, draft_model: nn.Module) -> None:
        self.target_model = target_model.eval()
        self.draft_model = draft_model.eval()
        self.draft = DraftGenerator(self.draft_model)
        self.verifier = TargetVerifier(self.target_model)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        config: SpeculativeGreedyConfig,
        *,
        eos_token_id: int | Collection[int] | None = None,
    ) -> SpeculativeGreedyResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("v0.2a expects unpadded input_ids [1, seq_len>=1]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")

        target_device = next(self.target_model.parameters()).device
        draft_device = next(self.draft_model.parameters()).device
        if input_ids.device != target_device or input_ids.device != draft_device:
            raise ValueError("Input, draft model and target model must share a device")
        # Equal vocab sizes are necessary, not sufficient: callers must also
        # ensure token ID mappings are identical (see the example script).
        target_vocab = getattr(getattr(self.target_model, "config", None), "vocab_size", None)
        draft_vocab = getattr(getattr(self.draft_model, "config", None), "vocab_size", None)
        if target_vocab is not None and draft_vocab is not None and target_vocab != draft_vocab:
            raise ValueError("Target and draft model vocabulary sizes differ")

        if eos_token_id is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = {int(x) for x in eos_token_id}

        sequences = input_ids.clone()
        prompt_len = input_ids.shape[1]
        collector = MetricsCollector(target_device)
        collector.start()
        target_calls = draft_calls = rounds = drafted = accepted = rejected_rounds = bonus = 0
        stop_reason = "length"

        while sequences.shape[1] - prompt_len < config.max_new_tokens:
            remaining = config.max_new_tokens - (sequences.shape[1] - prompt_len)
            k = min(config.draft_k, remaining)

            proposal = self.draft.propose(sequences, num_tokens=k)
            draft_calls += proposal.forward_calls
            drafted += k

            verification = self.verifier.verify(sequences, proposal.token_ids)
            target_calls += 1
            rounds += 1

            decision = accept_greedy(
                proposal.token_ids, verification,
                remaining_tokens=remaining, eos_token_ids=eos_ids,
            )
            accepted += decision.accepted_draft_tokens
            rejected_rounds += int(decision.rejected)
            bonus += int(decision.used_bonus_token)
            sequences = torch.cat((sequences, decision.committed_token_ids), dim=1)

            if decision.stop_reason is not None:
                stop_reason = decision.stop_reason
                break

        generated = sequences[:, prompt_len:]
        metrics = collector.finish(
            generated_tokens=generated.shape[1],
            forward_calls=target_calls + draft_calls,
        )
        return SpeculativeGreedyResult(
            sequences=sequences,
            new_token_ids=generated,
            stop_reason=stop_reason,
            metrics=metrics,
            target_forward_calls=target_calls,
            draft_forward_calls=draft_calls,
            rounds=rounds,
            drafted_tokens=drafted,
            accepted_draft_tokens=accepted,
            rejection_rounds=rejected_rounds,
            bonus_tokens=bonus,
        )
