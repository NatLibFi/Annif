"""Unit tests for analyzers in Annif"""

import pytest
import annif.analyzer


def test_get_analyzer_nonexistent():
    with pytest.raises(ValueError):
        annif.analyzer.get_analyzer("nonexistent")


def test_get_analyzer_badspec():
    with pytest.raises(ValueError):
        annif.analyzer.get_analyzer("()")


def test_english_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("snowball(english)")
    assert analyzer._normalize_word("running") == "run"
    assert analyzer._normalize_word("words") == "word"


def test_english_tokenize_sentences():
    analyzer = annif.analyzer.get_analyzer("snowball(english)")

    text = """But I must explain to you how all this mistaken idea of
    denouncing pleasure and praising pain was born and I will give you a
    complete account of the system, and expound the actual teachings of the
    great explorer of the truth, the master-builder of human happiness. No
    one rejects, dislikes, or avoids pleasure itself, because it is
    pleasure, but because those who do not know how to pursue pleasure
    rationally encounter consequences that are extremely painful. Nor again
    is there anyone who loves or pursues or desires to obtain pain of
    itself, because it is pain, but because occasionally circumstances occur
    in which toil and pain can procure him some great pleasure. To take a
    trivial example, which of us ever undertakes laborious physical
    exercise, except to obtain some advantage from it? But who has any right
    to find fault with a man who chooses to enjoy a pleasure that has no
    annoying consequences, or one who avoids a pain that produces no
    resultant pleasure?"""

    sentences = analyzer.tokenize_sentences(text)
    assert len(sentences) == 5


def test_english_tokenize_words():
    analyzer = annif.analyzer.get_analyzer("snowball(english)")
    text = """To take a trivial example, which of us ever undertakes
    laborious physical exercise, except to obtain some advantage from it?"""
    words = analyzer.tokenize_words(text)
    assert len(words) == 14


def test_english_tokenize_words_no_filter():
    analyzer = annif.analyzer.get_analyzer("snowball(english)")
    text = """To take a trivial example, which of us ever undertakes
    laborious physical exercise, except to obtain some advantage from it?"""
    words = analyzer.tokenize_words(text, filter=False)
    assert len(words) == 23


def test_english_filter_words_min_token():
    analyzer = annif.analyzer.get_analyzer(
        "snowball(english,token_min_length=2)")
    text = """Since 2000, a 3D printer can be used to print
    3 kinds of objects."""
    words = analyzer.tokenize_words(text)
    assert len(words) == 11
    assert '2000' not in words
    assert 'sinc' in words
    assert 'object' in words
    assert 'a' not in words


def test_swedish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("snowball(swedish)")
    assert analyzer._normalize_word("gamla") == "gaml"
    assert analyzer._normalize_word("hundar") == "hund"


def test_snowball_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("snowball(finnish)")
    assert analyzer._normalize_word("vanhat") == "vanh"
    assert analyzer._normalize_word("koirien") == "koir"


def test_simple_analyzer():
    analyzer = annif.analyzer.get_analyzer("simple")
    assert analyzer._normalize_word("Big") == "big"


def test_simple_analyzer_token_size():
    analyzer = annif.analyzer.get_analyzer("simple(token_min_length=2)")
    text = 'I do stuff'
    tokens = analyzer.tokenize_words(text)
    assert len(tokens) == 2
