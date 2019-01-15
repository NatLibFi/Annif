"""Unit tests for voikko analyzer in Annif"""

import pytest
import annif.analyzer

voikko = pytest.importorskip("voikko")


def test_voikko_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    assert analyzer.normalize_word("vanhat") == "vanha"
    assert analyzer.normalize_word("koirien") == "koira"
