"""Unit tests for vocabulary exclude rules in Annif"""

from annif.vocab import kwargs_to_exclude_uris


def test_vocab_rules_exclude(vocabulary):
    uris = kwargs_to_exclude_uris(vocabulary, {"exclude": "https://example.org/"})
    assert uris == {"https://example.org/"}


def test_vocab_rules_exclude_many(vocabulary):
    uris = kwargs_to_exclude_uris(
        vocabulary,
        {"exclude": "https://example.org/1|https://example.org/2"},
    )
    assert uris == {"https://example.org/1", "https://example.org/2"}


def test_vocab_rules_exclude_all(vocabulary):
    uris = kwargs_to_exclude_uris(vocabulary, {"exclude": "*"})
    assert len(uris) == 130


def test_vocab_rules_exclude_type(vocabulary):
    uris = kwargs_to_exclude_uris(
        vocabulary,
        {"exclude_type": "http://www.yso.fi/onto/yso-meta/Individual"},
    )
    # there are 4 concepts of type Individual in yso-archaeology
    assert len(uris) == 4
    assert "http://www.yso.fi/onto/yso/p19180" in uris


def test_vocab_rules_exclude_scheme(vocabulary):
    uris = kwargs_to_exclude_uris(
        vocabulary,
        {"exclude_scheme": "http://www.yso.fi/onto/yso/test-scheme"},
    )
    assert len(uris) == 1
    assert "http://www.yso.fi/onto/yso/p1265" in uris


def test_vocab_rules_exclude_collection(vocabulary):
    uris = kwargs_to_exclude_uris(
        vocabulary,
        {"exclude_collection": "http://www.yso.fi/onto/yso/p26569"},
    )
    assert len(uris) == 3
    assert "http://www.yso.fi/onto/yso/p7141" in uris


def test_vocab_rules_exclude_include_combination(vocabulary):
    uris = kwargs_to_exclude_uris(
        vocabulary,
        {
            "exclude": "*",
            "include": "|".join(
                [
                    "http://www.yso.fi/onto/yso/p6436",
                    "http://www.yso.fi/onto/yso/p11052",
                ]
            ),
            "include_type": "http://www.yso.fi/onto/yso-meta/Individual",
            "include_scheme": "http://www.yso.fi/onto/yso/test-scheme",
            "include_collection": "http://www.yso.fi/onto/yso/p26569",
        },
    )
    assert len(uris) == (130 - 10)
