import pytest
import torch

from speclab.decoding.autoregressive import AutoregressiveDecoder, GenerationConfig
from speclab.decoding.sampling import SamplingConfig


def test_cache_and_no_cache_match(toy_model) -> None:
    prompt = torch.tensor([[2, 3, 4]])
    decoder = AutoregressiveDecoder(toy_model)

    cached = decoder.generate(prompt, GenerationConfig(max_new_tokens=4, use_cache=True))
    assert toy_model.seen_input_lengths == [3, 1, 1, 1]
    assert cached.new_token_ids.tolist() == [[5, 6, 7, 0]]
    assert cached.metrics.forward_calls == 4
    assert cached.cached_length == cached.sequences.shape[1] - 1

    toy_model.seen_input_lengths.clear()
    uncached = decoder.generate(prompt, GenerationConfig(max_new_tokens=4, use_cache=False))
    assert toy_model.seen_input_lengths == [3, 4, 5, 6]
    assert uncached.cached_length == 0
    torch.testing.assert_close(cached.sequences, uncached.sequences)
    torch.testing.assert_close(prompt, torch.tensor([[2, 3, 4]]))


def test_stops_at_eos(toy_model) -> None:
    result = AutoregressiveDecoder(toy_model).generate(
        torch.tensor([[0, 1]]), GenerationConfig(max_new_tokens=10), eos_token_id=3
    )
    assert result.new_token_ids.tolist() == [[2, 3]]
    assert result.stop_reason == "eos"
    assert result.metrics.generated_tokens == 2


def test_stop_at_length(toy_model) -> None:
    result = AutoregressiveDecoder(toy_model).generate(
        torch.tensor([[2]]), GenerationConfig(max_new_tokens=2), eos_token_id=[0, 1]
    )
    assert result.new_token_ids.tolist() == [[3, 4]]
    assert result.stop_reason == "length"


def test_sampling_mode_does_not_require_generate(toy_model) -> None:
    config = GenerationConfig(
        max_new_tokens=3,
        sampling=SamplingConfig(mode="sample", temperature=1.0, top_k=1),
    )
    result = AutoregressiveDecoder(toy_model).generate(torch.tensor([[1, 2]]), config)
    assert result.new_token_ids.tolist() == [[3, 4, 5]]


@pytest.mark.parametrize("bad_shape", [torch.tensor([1, 2]), torch.empty((1, 0), dtype=torch.long),
                                          torch.tensor([[1], [2]])])
def test_rejects_unsupported_shapes(toy_model, bad_shape) -> None:
    with pytest.raises(ValueError):
        AutoregressiveDecoder(toy_model).generate(bad_shape, GenerationConfig())


def test_rejects_invalid_length() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)


def test_cache_preserves_context_dependent_logits() -> None:
    """A context-sensitive fake catches accidental loss of earlier tokens."""
    from types import SimpleNamespace
    from torch import nn

    class PrefixSumLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids, attention_mask, use_cache, return_dict,
                    past_key_values=None):
            full = input_ids if past_key_values is None else torch.cat(
                [past_key_values, input_ids], dim=-1
            )
            assert attention_mask.shape == full.shape
            n = input_ids.shape[1]
            cumulative = full.cumsum(dim=-1)[:, -n:]
            preferred = (cumulative + 1) % 11
            scores = torch.full((1, n, 11), -100.0)
            scores.scatter_(2, preferred.unsqueeze(-1), 100.0)
            return SimpleNamespace(
                logits=scores,
                past_key_values=full if use_cache else None,
            )

    model = PrefixSumLM()
    decoder = AutoregressiveDecoder(model)
    prompt = torch.tensor([[1, 2, 3]])
    cached = decoder.generate(prompt, GenerationConfig(max_new_tokens=6))
    uncached = decoder.generate(prompt, GenerationConfig(max_new_tokens=6, use_cache=False))
    torch.testing.assert_close(cached.sequences, uncached.sequences)
