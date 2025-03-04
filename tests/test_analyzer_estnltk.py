"""Unit tests for EstNLTK analyzer in Annif"""

import pytest

import annif.analyzer
import annif.analyzer.estnltk

pytestmark = pytest.mark.skipif(
    not annif.analyzer.estnltk.EstNLTKAnalyzer.is_available(),
    reason="EstNLTK is required",
)


def test_estnltk_tokenize_words():
    analyzer = annif.analyzer.get_analyzer("estnltk")
    words = analyzer.tokenize_words(
        """
        Aga kõik juhtus iseenesest. Ka köögis oli kõik endine.
        """
    )
    assert words == [
        "aga",
        "kõik",
        "juhtuma",
        "iseenesest",
        "köök",
        "olema",
        "kõik",
        "endine",
    ]


def test_estnltk_tokenize_words_no_filter():
    analyzer = annif.analyzer.get_analyzer("estnltk")
    words = analyzer.tokenize_words(
        """
        Aga kõik juhtus iseenesest. Ka köögis oli kõik endine.
        """,
        filter=False,
    )
    assert words == [
        "aga",
        "kõik",
        "juhtuma",
        "iseenesest",
        ".",
        "ka",
        "köök",
        "olema",
        "kõik",
        "endine",
        ".",
    ]
