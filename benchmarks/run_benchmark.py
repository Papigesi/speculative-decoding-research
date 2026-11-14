"""
Example: python benchmarks/run_benchmark.py --repeats 5 --max-new-tokens 64

Simple wall-clock benchmark. Both modes generate identical greedy
sequences, but differ in whether previously seen tokens are recomputed.
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from statistics import median

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speculative_decoding.decoding.autoregressive import AutoregressiveDecoder, GenerationConfig
from speculative_decoding.models.adapter import HFModelAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None, help="Write results as JSON")

    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be >=1 and warmups must be >=0")

    adapter = HFModelAdapter.from_pretrained(args.model, device=args.device)
    decoder = AutoregressiveDecoder(adapter.model)

    prompt_ids = adapter.encode(args.prompt)
    eos_id = adapter.tokenizer.eos_token_id

    results: dict = {"model": args.model, "prompt": args.prompt,
                     "max_new_tokens": args.max_new_tokens, "modes": {}}
    
    reference_tokens = None
    for label, use_cache in (("no_cache", False), ("with_cache", True)):
        config = GenerationConfig(max_new_tokens=args.max_new_tokens, use_cache=use_cache)

        for _ in range(args.warmups):
            decoder.generate(prompt_ids, config, eos_token_id=eos_id)

        samples = [decoder.generate(prompt_ids, config, eos_token_id=eos_id)
                   for _ in range(args.repeats)]
        tokens = samples[0].new_token_ids
        if reference_tokens is None:
            reference_tokens = tokens
        elif not torch.equal(reference_tokens, tokens):
            raise AssertionError("Greedy output differs between cache modes")
        
        record = {
            "median_latency_seconds": median(x.metrics.elapsed_seconds for x in samples),
            "median_tokens_per_second": median(x.metrics.tokens_per_second for x in samples),
            "generated_tokens": int(tokens.shape[1]),
            "forward_calls": samples[0].metrics.forward_calls,
            "peak_gpu_memory_bytes": max(
                (x.metrics.peak_gpu_memory_bytes or 0) for x in samples
            ) if adapter.device.type == "cuda" else None,
        }
        results["modes"][label] = record
        print(f"{label:>10}: {record['median_latency_seconds']:.3f} s | "
              f"{record['median_tokens_per_second']:.2f} tok/s")

    old = results["modes"]["no_cache"]["median_latency_seconds"]
    new = results["modes"]["with_cache"]["median_latency_seconds"]
    results["cache_speedup"] = old / new if new else None
    print(f"Cache speedup: {results['cache_speedup']:.2f}x")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
