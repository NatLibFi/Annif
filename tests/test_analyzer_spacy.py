"""Unit tests for spacy analyzer in Annif"""

import pytest
import annif.analyzer
from annif.exception import OperationFailedException

spacy = pytest.importorskip("spacy")


def test_spacy_model_not_found():
    with pytest.raises(OperationFailedException) as excinfo:
        annif.analyzer.get_analyzer("spacy(not_found)")
    assert "Loading spaCy model 'not_found' failed" in str(excinfo.value)


def test_spacy_english_tokenize_words():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    words = analyzer.tokenize_words("""
        The quick brown foxes jumped over the lazy dogs in Paris.
        """)
    assert words == ['the', 'quick', 'brown', 'fox',
                     'jump', 'over', 'the', 'lazy', 'dog', 'Paris']


def test_spacy_english_tokenize_words_no_filter():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    words = analyzer.tokenize_words("""
        The quick brown foxes jumped over the lazy dogs in Paris.
        """, filter=False)
    assert words == ['the', 'quick', 'brown', 'fox',
                     'jump', 'over', 'the', 'lazy', 'dog', 'in', 'Paris', '.']


def test_spacy_english_tokenize_words_lowercase():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm,lowercase=1)")
    words = analyzer.tokenize_words("""
        The quick brown foxes jumped over the lazy dogs in Paris.
        """)
    assert words == ['the', 'quick', 'brown', 'fox',
                     'jump', 'over', 'the', 'lazy', 'dog', 'paris']
