"""Unit tests for simplemma analyzer in Annif"""

import pickle

import pytest

import annif.analyzer
from annif.analyzer.simplemma import SimplemmaAnalyzer

simplemma = pytest.importorskip("annif.analyzer.simplemma")


def test_simplemma_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("simplemma(fi)")
    assert analyzer._normalize_word("xyzzy") == "xyzzy"
    assert analyzer._normalize_word("vanhat") == "vanha"
    assert analyzer._normalize_word("koirien") == "koira"


def test_simplemma_analyzer_is_picklable():
    analyzer = annif.analyzer.get_analyzer("simplemma(fi)")
    assert isinstance(analyzer, SimplemmaAnalyzer)
    # force the lemmatizer to be created, then verify it is excluded
    # from the pickled state and that the unpickled instance works
    analyzer._normalize_word("vanhat")
    state = analyzer.__getstate__()
    assert state["_lemmatizer"] is None
    unpickled = pickle.loads(pickle.dumps(analyzer))
    assert unpickled._lemmatizer is None
    assert unpickled._normalize_word("vanhat") == "vanha"
