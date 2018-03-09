"""Unit tests for analyzers in Annif"""

import pytest
import annif.analyzer

def test_get_analyzer_nonexistent():
    with pytest.raises(ValueError):
        annif.analyzer.get_analyzer("nonexistent")

def test_english_analyzer():
    analyzer = annif.analyzer.get_analyzer("english")
    assert analyzer.normalize_word("running") == "run"
    assert analyzer.normalize_word("words") == "word"

def test_english_tokenize_sentences():
    analyzer = annif.analyzer.get_analyzer("english")

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
    analyzer = annif.analyzer.get_analyzer("english")
    text = """To take a trivial example, which of us ever undertakes
    laborious physical exercise, except to obtain some advantage from it?"""
    words = analyzer.tokenize_words(text)
    assert len(words) == 23

    
