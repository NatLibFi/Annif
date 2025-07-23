"""Unit tests for vocabulary exclude rules in Annif"""

from annif.vocab import kwargs_to_exclude_uris


def test_vocab_rules_exclude():
    uris = kwargs_to_exclude_uris({'exclude': 'https://example.org/'})
    assert uris == {'https://example.org/'}
