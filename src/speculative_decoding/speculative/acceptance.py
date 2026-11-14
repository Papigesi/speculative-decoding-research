"""
Exact agreement rule for greedy speculative decoding.
No probability ratios or residual distribution are used for now
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Literal

import torch
from torch import Tensor

from speclab.speculative.verifier import GreedyVerification


@dataclass(frozen=True)
class GreedyAcceptanceResult:
    committed_token_ids: Tensor  # [1, n], 1 <= n <= k+1
    accepted_draft_tokens: int
    rejected: bool
    used_bonus_token: bool
    stop_reason: Literal["eos", "length"] | None


def accept_greedy(
    draft_token_ids: Tensor,
    verification: GreedyVerification,
    *,
    remaining_tokens: int,
    eos_token_ids: Collection[int] = (),
) -> GreedyAcceptanceResult:
    """Commit the longest agreeing draft prefix and one target continuation.

    At the first mismatch, replace the draft token with the target's argmax.
    If all draft tokens match, append the target's bonus token. Enforce EOS and
    remaining token budget *after* resolving the proposal.
    """
    targets = verification.target_token_ids
    bonus = verification.bonus_token_id
    if (draft_token_ids.ndim != 2 or draft_token_ids.shape[0] != 1
            or draft_token_ids.shape[1] == 0 or targets.shape != draft_token_ids.shape
            or bonus.shape != (1, 1)):
        raise ValueError("Expected draft and target [1, k>=1], bonus [1, 1]")
    if remaining_tokens < 1:
        raise ValueError("remaining_tokens must be >= 1")
    if draft_token_ids.device != targets.device or draft_token_ids.device != bonus.device:
        raise ValueError("All token tensors must be on the same device")

    mismatch_indices = torch.nonzero(draft_token_ids[0] != targets[0]).flatten()
    first_mismatch = int(mismatch_indices[0].item()) if mismatch_indices.numel() else None
    accepted = draft_token_ids.shape[1] if first_mismatch is None else first_mismatch
    rejected = first_mismatch is not None

    if rejected:
        # Keep only agreeing y1..y_{i-1}; replace yi by target's greedy choice.
        continuation = targets[:, first_mismatch : first_mismatch + 1]
    else:
        # Every draft token matched: we get one extra target token for free.
        continuation = bonus
    candidate = torch.cat((draft_token_ids[:, :accepted], continuation), dim=1)
    committed = candidate[:, :remaining_tokens]

    reason: Literal["eos", "length"] | None = None
    eos_set = set(eos_token_ids)
    if eos_set:
        # Small transfer to CPU, acceptable in correctness-first v0.2a.
        for index, token_id in enumerate(committed[0].tolist()):
            if token_id in eos_set:
                committed = committed[:, : index + 1]
                reason = "eos"
                break
    if reason is None and committed.shape[1] >= remaining_tokens:
        reason = "length"

    accepted_committed = min(accepted, committed.shape[1])

    return GreedyAcceptanceResult(
        committed_token_ids=committed,
        accepted_draft_tokens=accepted_committed,
        rejected=rejected,
        used_bonus_token=(not rejected and committed.shape[1] > accepted),
        stop_reason=reason,
    )
