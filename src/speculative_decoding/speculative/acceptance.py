"""
Exact agreement rule for greedy speculative decoding.
No probability ratios or residual distribution are used for now
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Literal

import torch
from torch import Tensor

from speculative_decoding.speculative.verifier import GreedyVerification, SamplingVerification


@dataclass(frozen=True)
class GreedyAcceptanceResult:
    committed_token_ids: Tensor  # [1, n], 1 <= n <= k+1
    accepted_draft_tokens: int
    rejected: bool
    used_bonus_token: bool
    stop_reason: Literal["eos", "length"] | None


@dataclass(frozen=True)
class SamplingAcceptanceResult:
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

    @torch.inference_mode()
    def accept_with_sampling(
        draft_token_ids: Tensor,
        draft_probs: Tensor,
        verification: SamplingVerification,
        *,
        remaining_tokens: int,
        eos_token_ids: Collection[int] = (),
        generator: torch.Generator | None = None
    ) -> SamplingAcceptanceResult:
        """Exact speculative sampling for batch = 1 using the same p/q policy"""
        if draft_token.ids.ndim != 2 or draft_token_ids.shape[0] != 1:
            raise ValueError("Expected tokens [1, k] and draft probs [1, k>=1]")
        k = draft_token_ids.shape[1]
        p = verification.target_probs
        q = draft_probs

        if k < 1 or q.ndim != 3 or p.ndim != 3 or q.shape[:2] != (1, k):
            raise ValueError("Expected target probs [1, k + 1, vocab]")
        if remaining_tokens < 1:
            raise ValueError("remaining_tokens must be >= 1")
        if any(t.device != draft_token_ids.device for t in (p, q)):
            raise ValueError("All token tensors must be on the same device" )
        if draft_token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("draft_token_ids must contain integer token IDs")
        if not (torch.isinfinite(p).all() and torch.isinfinite(q).all() and (p >= 0).all() and (q >= 0).all()):
            raise ValueError("Target and draft probabilities must be finite and nonnegative")
        if not (torch.allclose(p.sum(dim=-1), torch.ones_like(p[..., 0]), atol=1e-5)
            and torch.allclose(q.sum(dim=-1), torch.ones_like(q[..., 0]), atol=1e-5)):
            raise ValueError("Probability distributions must sum to 1")

        eos = set(eos_token_ids)
        comitted: list[Tensor] = []
        accepted = 0
        rejected = False
        used_bonus = False
        reason: Literal["eos", "length"] | None = None

        # Iterating only through the possible comitted prefix
        for i in range(min(k, remaining_tokens)):
            token = draft_token_ids[:, i : i + 1]
            token_id = int(token.item())
            p_i = p[0, i]
            q_i = q[0, i]
            q_y = float(q_i[token_id].item())
            if q_y <= 0:
                raise ValueError(f"A proposed token cannot have q_i(y_i) == 0, got {q_y} at position {i}")

            ratio = min(1.0, float(p_i[token_id].item()) / q_y)
            if ratio == 1.0 or bool(torch.rand((), device=p.device, generator=generator) < ratio):
                comitted.append(token)
                accepted += 1
                if token_id in eos:
                    reason = "eos"
                    break
                if len(comitted) == remaining_tokens:
                    reason = "length"
                    break
                continue

            rejected = True
            # No fallback to sampling from p; that would change the distribution of the comitted prefix. Instead, we stop at the first mismatch and append the target's greedy choice.
            residual = (p_i - q_i).clamp_min(0)
            residual_mass = residual.sum()
            if float(residual_mass.item()) <= 0:
                raise ValueError(f"Rejection occurred but the residual mass is zero at position {i}")

            replacement = torch.multinomial((residual / residual_mass).unsqueeze(0), num_samples=1, generator=generator)
            comitted.append(replacement)
            if int(replacement.item()) in eos:
                reason = "eos"
            elif len(comitted) == remaining_tokens:
                reason = "length"
            break

        if not rejected and reason is None:
            # All k proposals accepted so p_(k + 1) is used as the bonus token.
            bonus = torch.multinomial(p[:, k, :], num_samples=1, generator=generator)
            comitted.append(bonus)
            used_bonus = True
            if int(bonus.item()) in eos:
                reason = "eos"
            elif len(comitted) == remaining_tokens:
                reason = "length"

        return SamplingAcceptanceResult(
            committed_token_ids=torch.cat(comitted, dim=1),
            accepted_draft_tokens=accepted,
            rejected=rejected,
            used_bonus_token=used_bonus,
            stop_reason=reason
        )