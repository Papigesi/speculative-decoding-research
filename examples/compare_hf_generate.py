"""
Offline-friendly structure; real model weights must be downloaded to run.

python examples/compare_hf_generate.py
"""

import sys
from pathlib import Path

# Allow running this file directly from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from speculative_decoding.decoding.autoregressive import AutoregressiveDecoder, GenerationConfig
from speculative_decoding.models.adapter import HFModelAdapter


def main() -> None:
    adapter = HFModelAdapter.from_pretrained("HuggingFaceTB/SmolLM-135M")
    prompt_ids = adapter.encode("Artificial intelligence will")
    cfg = GenerationConfig(max_new_tokens=30, use_cache=True)

    ours = AutoregressiveDecoder(adapter.model).generate(
        prompt_ids, cfg, eos_token_id=adapter.tokenizer.eos_token_id
    )
    with torch.inference_mode():
        reference = adapter.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=adapter.tokenizer.eos_token_id,
            pad_token_id=adapter.tokenizer.eos_token_id,
        )

    same = torch.equal(ours.sequences, reference)
    print("Greedy agreement with HF generate():", same)
    if not same:
        raise AssertionError("Greedy outputs do not match the HF reference")


if __name__ == "__main__":
    main()
