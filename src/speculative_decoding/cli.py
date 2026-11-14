"""CLI entry point"""

from __future__ import annotations

import argparse

import torch

from speclab.decoding.autoregressive import AutoregressiveDecoder, GenerationConfig
from speclab.decoding.sampling import SamplingConfig
from speclab.models.adapter import HFModelAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpecLab v0.1 autoregressive decoding")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"],
                        default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    adapter = HFModelAdapter.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    input_ids = adapter.encode(args.prompt)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_cache,
        sampling=SamplingConfig(
            mode=args.mode, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
        ),
    )
    result = AutoregressiveDecoder(adapter.model).generate(
        input_ids, config, eos_token_id=adapter.tokenizer.eos_token_id
    )

    print("\n--- Completion ---")
    print(adapter.decode(result.new_token_ids))
    print("\n--- Metrics ---")
    print(f"Tokens:           {result.metrics.generated_tokens}")
    print(f"Elapsed time:     {result.metrics.elapsed_seconds:.3f} s")
    print(f"Tokens/sec:       {result.metrics.tokens_per_second:.2f}")
    print(f"Forward passes:   {result.metrics.forward_calls}")
    print(f"Cache length:     {result.cached_length}")
    print(f"Stop reason:      {result.stop_reason}")
    if result.metrics.peak_gpu_memory_bytes is not None:
        print(f"Peak GPU allocated: {result.metrics.peak_gpu_memory_bytes / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
