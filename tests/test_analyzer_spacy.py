"""Unit tests for spacy analyzer in Annif"""

import pytest
import annif.analyzer

spacy = pytest.importorskip("annif.analyzer.spacy")


def test_spacy_english_tokenize_words():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    words = analyzer.tokenize_words("""
        The quick brown foxes jumped over the lazy dogs.
        """)
    print(words)


def test_spacy_english_normalize_word():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    assert analyzer.normalize_word("xyzzy") == "xyzzy"
    assert analyzer.normalize_word("older") == "old"
    assert analyzer.normalize_word("dogs") == "dog"
