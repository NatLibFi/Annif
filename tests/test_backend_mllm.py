"""Unit tests for the MLLM backend in Annif"""

import annif
import annif.backend
from annif.backend.mllm import TokenSet


def test_mllm_tokenset():
    tokens = [1, 3, 5]
    tset = TokenSet(tokens)
    assert tset.subject_id is None
    assert not tset.is_pref
    assert len(tset) == len(tokens)
    assert sorted(list(tset)) == sorted(tokens)
    assert tset.contains(TokenSet(tokens))
    assert tset.contains(TokenSet([1]))
    assert not tset.contains(TokenSet([0]))
    assert tset.sample() in tokens


def test_mllm_tokenset_empty_sample():
    assert TokenSet([]).sample() is None


def test_mllm_default_params(project):
    mllm_type = annif.backend.get_backend("mllm")
    mllm = mllm_type(
        backend_id='mllm',
        config_params={},
        project=project)

    expected_default_params = {
        'limit': 100  # From AnnifBackend class
    }
    actual_params = mllm.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val
