"""Unit tests for the token set index"""

from annif.lexical.tokenset import TokenSet, TokenSetIndex


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


def test_mllm_tokensetindex():
    index = TokenSetIndex()
    assert len(index) == 0
    tset13 = TokenSet([1, 3], subject_id=1)
    index.add(tset13)
    assert len(index) == 1
    index.add(TokenSet([]))  # add empty
    assert len(index) == 1
    tset2 = TokenSet([2])
    index.add(tset2)
    tset23 = TokenSet([2, 3], subject_id=2)
    index.add(tset23)
    tset3 = TokenSet([3], subject_id=3, is_pref=True)
    index.add(tset3)
    tset34 = TokenSet([3, 4], subject_id=3, is_pref=False)
    index.add(tset34)
    tset5 = TokenSet([5])
    index.add(tset5)

    result = index.search(TokenSet([1, 2, 3, 4]))
    assert len(result) == 4

    assert (tset13, 0) in result
    assert (tset2, 1) in result
    assert (tset23, 0) in result
    assert (tset3, 2) in result

    assert tset34 not in [r[0] for r in result]
    assert tset5 not in [r[0] for r in result]
