"""Unit tests for simplemma analyzer in Annif"""

import pytest
import annif.analyzer

simplemma = pytest.importorskip("annif.analyzer.simplemma")


def test_simplemma_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("simplemma(fi)")
    assert analyzer._normalize_word("xyzzy") == "xyzzy"
    assert analyzer._normalize_word("vanhat") == "vanha"
    assert analyzer._normalize_word("koirien") == "koira"
