"""Unit tests for the input-transformers in Annif"""

import pytest
import annif.transformer
from annif.exception import ConfigurationException


def test_get_transformer_nonexistent():
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transformer("nonexistent", project=None)


def test_get_transformer_badspec(project):
    with pytest.raises(ConfigurationException):
        annif.transformer.get_transformer("pass(invalid_argument)", project)


def test_input_limiter():
    transf = annif.transformer.get_transformer("limit_input(3)", project=None)
    assert transf.transform_text("running") == "run"


def test_chained_transformers_text():
    transf = annif.transformer.get_transformer(
        "limit_input(5),   pass,limit_input(3),", project=None)
    assert transf.transform_text("abcdefghij") == "abc"
