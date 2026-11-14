import pytest

from speclab.cache.manager import CacheState


def test_cache_state_tracks_processed_not_sampled_tokens() -> None:
    state = CacheState()
    state.update(object(), processed_tokens=12)
    assert state.cached_length == 12
    state.update(object(), processed_tokens=1)
    assert state.cached_length == 13


def test_cache_requires_model_kv_output() -> None:
    with pytest.raises(RuntimeError):
        CacheState().update(None, processed_tokens=1)
