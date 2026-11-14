import pytest
import torch

from speclab.decoding.sampling import SamplingConfig, logits_to_probs, select_next_token


def test_greedy_uses_argmax() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.2], [1.0, 0.3, 0.4]])
    actual = select_next_token(logits, SamplingConfig(mode="greedy"))
    assert actual.tolist() == [[1], [0]]


def test_temperature_changes_concentration() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    cold = logits_to_probs(logits, SamplingConfig(mode="sample", temperature=0.5))
    hot = logits_to_probs(logits, SamplingConfig(mode="sample", temperature=2.0))
    assert cold[0, -1] > hot[0, -1]
    torch.testing.assert_close(cold.sum(-1), torch.ones(1))


def test_top_k_masks_everything_else() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    probs = logits_to_probs(logits, SamplingConfig(mode="sample", top_k=2))
    assert probs[0, :2].tolist() == [0, 0]
    assert (probs[0, 2:] > 0).all()


def test_top_p_keeps_smallest_sufficient_prefix() -> None:
    # Softmax(log([0.6, 0.25, 0.15])) gives those exact proportions.
    logits = torch.tensor([[0.6, 0.25, 0.15]]).log()
    config = SamplingConfig(mode="sample", top_p=0.7)
    probs = logits_to_probs(logits, config)
    assert probs[0, 0] > 0 and probs[0, 1] > 0
    assert probs[0, 2].item() == 0
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))


def test_vectorized_sampling_matches_distribution() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.7]]).log().expand(5000, -1)
    generator = torch.Generator().manual_seed(123)
    tokens = select_next_token(
        logits, SamplingConfig(mode="sample"), generator=generator
    ).squeeze(-1)
    observed = torch.bincount(tokens, minlength=3).float() / tokens.numel()
    torch.testing.assert_close(observed, torch.tensor([0.1, 0.2, 0.7]), atol=0.025, rtol=0)


@pytest.mark.parametrize(
    "kwargs", [{"mode": "other"}, {"temperature": 0},
               {"temperature": float("inf")}, {"top_k": 0},
               {"top_p": 0}, {"top_p": 1.1}]
)
def test_invalid_sampling_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)
