"""Unit tests for voikko analyzer in Annif"""

import pytest

import annif.analyzer
import annif.analyzer.voikko

pytestmark = pytest.mark.skipif(
    not annif.analyzer.voikko.VoikkoAnalyzer.is_available(), reason="voikko is required"
)


def test_voikko_getstate():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    state = analyzer.__getstate__()
    assert state == {"param": "fi", "voikko": None}


def test_voikko_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    assert analyzer._normalize_word("xyzzy") == "xyzzy"
    assert analyzer._normalize_word("vanhat") == "vanha"
    assert analyzer._normalize_word("koirien") == "koira"
