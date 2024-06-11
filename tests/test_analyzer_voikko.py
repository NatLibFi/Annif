"""Unit tests for voikko analyzer in Annif"""

from unittest import mock

import pytest

import annif.analyzer
from annif.exception import OperationFailedException

voikko = pytest.importorskip("annif.analyzer.voikko")


def test_voikko_getstate():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    state = analyzer.__getstate__()
    assert state == {"param": "fi", "voikko": None}


def test_voikko_finnish_analyzer_normalize_word():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    assert analyzer._normalize_word("xyzzy") == "xyzzy"
    assert analyzer._normalize_word("vanhat") == "vanha"
    assert analyzer._normalize_word("koirien") == "koira"


def test_voikko_analyze_valueerror():
    analyzer = annif.analyzer.get_analyzer("voikko(fi)")
    with mock.patch(
        "voikko.libvoikko.Voikko.analyze",
        side_effect=ValueError,
    ):
        with pytest.raises(
            OperationFailedException, match="Voikko error in analysis of word 'kissa'"
        ):
            assert analyzer._normalize_word("kissa")
