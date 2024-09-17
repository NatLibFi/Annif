"""Unit tests for Simplemma utility functions"""

import pytest

from annif.simplemma_util import detect_language, get_language_detector


def test_get_language_detector():
    detector = get_language_detector("en")
    text = "She said 'au revoir' and left"
    proportion = detector.proportion_in_target_languages(text)
    assert proportion == pytest.approx(0.75)


def test_get_language_detector_many():
    detector = get_language_detector(("en", "fr"))
    text = "She said 'au revoir' and left"
    proportion = detector.proportion_in_target_languages(text)
    assert proportion == pytest.approx(1.0)


def test_detect_language():
    text = "She said 'au revoir' and left"
    languages = ("fr", "en")
    proportions = detect_language(text, languages)
    assert proportions["en"] == pytest.approx(0.75)
    assert proportions["fr"] == pytest.approx(0.25)
    assert list(proportions.keys())[0] == "en"
