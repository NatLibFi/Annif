"""Unit tests for spacy analyzer in Annif"""

import pytest
import annif.analyzer

spacy = pytest.importorskip("annif.analyzer.spacy")


def test_spacy_english_tokenize_sentences():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    sentences = analyzer.tokenize_sentences("""
        The quick brown fox jumps over the lazy dog.
        The five boxing wizards jump quickly.
        Pack my box with five dozen liquor jugs.
        """.strip())
    assert len(sentences) == 3
    assert sentences[0].text.strip() == \
        'The quick brown fox jumps over the lazy dog.'
    assert sentences[1].text.strip() == \
        'The five boxing wizards jump quickly.'
    assert sentences[2].text.strip() == \
        'Pack my box with five dozen liquor jugs.'


def test_spacy_english_tokenize_words():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    words = analyzer.tokenize_words("""
        The quick brown foxes jumped over the lazy dogs.
        """)
    assert words == ['the', 'quick', 'brown', 'fox',
                     'jump', 'over', 'the', 'lazy', 'dog']


def test_spacy_english_normalize_word():
    analyzer = annif.analyzer.get_analyzer("spacy(en_core_web_sm)")
    assert analyzer.normalize_word("xyzzy") == "xyzzy"
    assert analyzer.normalize_word("older") == "old"
    assert analyzer.normalize_word("dogs") == "dog"
