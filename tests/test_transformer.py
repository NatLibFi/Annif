"""Unit tests for the input-transformers in Annif"""

import pytest
import annif.transformer
from annif.exception import ConfigurationException
from annif.transformer import parse_specs


def test_parse_specs():
    parsed = parse_specs('foo, bar(42,43,key=abc)')
    assert parsed == [('foo', [], {}), ('bar', ['42', '43'], {'key': 'abc'})]


def test_get_transform_nonexistent():
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transform("nonexistent", project=None)


def test_get_transform_badspec(project):
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transform("pass(invalid_argument)", project)


def test_input_limiter():
    transf = annif.transformer.get_transform("limit(3)", project=None)
    assert transf.transform_text("running") == "run"


def test_chained_transforms_text():
    transf = annif.transformer.get_transform(
        "limit(5),pass,limit(3),", project=None)
    assert transf.transform_text("abcdefghij") == "abc"
