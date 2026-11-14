"""Hugging Face adapter: load the weights, then expose raw model.forward()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class HFModelAdapter:
    model: nn.Module
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype = torch.float16

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "HFModelAdapter":
        # Lazy import makes unit tests for our sampling/decoding engine offline.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        selected_device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch build cannot use CUDA")
        
        if dtype == "auto":
            selected_dtype = torch.float16 if selected_device.type == "cuda" else torch.float32
        else:
            choices = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}
            if dtype not in choices:
                raise ValueError(f"dtype must be one of {list(choices)} or 'auto'")
            selected_dtype = choices[dtype]

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=selected_dtype, trust_remote_code=False
        )
        model.to(selected_device)
        model.eval()

        return cls(model=model, tokenizer=tokenizer, device=selected_device, dtype=selected_dtype)

    def encode(self, prompt: str) -> Tensor:
        if not prompt:
            raise ValueError("prompt can not be empty!")
        
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        return tokenized["input_ids"].to(self.device)

    def decode(self, token_ids: Tensor) -> str:
        return self.tokenizer.decode(
            token_ids.squeeze(0).tolist(), skip_special_tokens=True
        )
