"""Unit tests for voikko analyzer in Annif"""

import pytest
import annif.analyzer

voikko = pytest.importorskip("annif.analyzer.voikko")


def test_voikko_getstate():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    state = analyzer.__getstate__()
    assert state == {'param': 'fi', 'voikko': None}


def test_voikko_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    assert analyzer.normalize_word("xyzzy") == "xyzzy"
    assert analyzer.normalize_word("vanhat") == "vanha"
    assert analyzer.normalize_word("koirien") == "koira"
