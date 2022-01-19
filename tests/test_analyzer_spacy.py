"""Unit tests for spacy analyzer in Annif"""

import pytest
import annif.analyzer

spacy = pytest.importorskip("annif.analyzer.spacy")


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


def test_spacy_english_normalize_word():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    assert analyzer.normalize_word("xyzzy") == "xyzzy"
    assert analyzer.normalize_word("older") == "old"
    assert analyzer.normalize_word("dogs") == "dog"
    assert analyzer.normalize_word("Paris") == "Paris"


def test_spacy_english_normalize_word_lowercase():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm,lowercase=1)")
    assert analyzer.normalize_word("Paris") == "paris"