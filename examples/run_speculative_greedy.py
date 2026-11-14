"""
Comparing greedy speculative decoding with baseline.
"""

from __future__ import annotations

import argparse

import torch

from speclab.decoding.autoregressive import AutoregressiveDecoder, GenerationConfig
from speclab.decoding.speculative import SpeculativeGreedyConfig, SpeculativeGreedyDecoder
from speclab.models.adapter import HFModelAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="SpecLab v0.2a: greedy speculative decoding")
    parser.add_argument("--target", default="HuggingFaceTB/SmolLM-1.7B")
    parser.add_argument("--draft", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--prompt", default="Artificial intelligence is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="auto")
    args = parser.parse_args()

    target = HFModelAdapter.from_pretrained(args.target, device=args.device, dtype=args.dtype)
    draft = HFModelAdapter.from_pretrained(args.draft, device=args.device, dtype=args.dtype)

    # Matching vocab sizes alone are NOT enough: IDs must mean the same tokens.
    if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
        raise ValueError("Target/draft tokenizers have different token -> ID mappings")
    if target.tokenizer.eos_token_id != draft.tokenizer.eos_token_id:
        raise ValueError("Target/draft EOS token IDs differ")

    input_ids = target.encode(args.prompt)
    config = SpeculativeGreedyConfig(
        max_new_tokens=args.max_new_tokens, draft_k=args.draft_k
    )
    speculative = SpeculativeGreedyDecoder(target.model, draft.model).generate(
        input_ids, config, eos_token_id=target.tokenizer.eos_token_id
    )
    baseline = AutoregressiveDecoder(target.model).generate(
        input_ids,
        GenerationConfig(max_new_tokens=args.max_new_tokens, use_cache=False),
        eos_token_id=target.tokenizer.eos_token_id,
    )
    if not torch.equal(speculative.new_token_ids, baseline.new_token_ids):
        raise AssertionError("Speculative tokens differ from the greedy autoregressive baseline")

    print("\n--- Generated completion (greedy) ---")
    print(target.decode(speculative.new_token_ids))
    print("\n--- v0.2a statistics ---")
    print(f"Correctness:               PASS (same token IDs as baseline)")
    print(f"Speculative rounds:        {speculative.rounds}")
    print(f"Draft token proposals:     {speculative.drafted_tokens}")
    print(f"Accepted draft tokens:     {speculative.accepted_draft_tokens}")
    print(f"Draft acceptance rate:     {speculative.acceptance_rate:.1%}")
    print(f"Rejected rounds:           {speculative.rejection_rounds}")
    print(f"Committed bonus tokens:    {speculative.bonus_tokens}")
    print(f"Target forward calls:      {speculative.target_forward_calls}")
    print(f"Draft forward calls:       {speculative.draft_forward_calls}")
    print(f"Speculative tokens/sec:    {speculative.metrics.tokens_per_second:.2f}")
    print(f"Baseline tokens/sec:       {baseline.metrics.tokens_per_second:.2f}")
    print("Note: both run without KV cache. This version prioritizes correctness, not speed.")


if __name__ == "__main__":
    main()
