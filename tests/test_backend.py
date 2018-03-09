"""Unit tests for backends in Annif"""

import pytest
import annif.backend


def test_get_backend_nonexistent():
    with pytest.raises(ValueError):
        annif.backend.get_backend_type("nonexistent")


def test_get_backend_type_dummy():
    dummy_type = annif.backend.get_backend_type("dummy")
    dummy = dummy_type(config={})
    result = dummy.analyze('this is some text')
    assert len(result) == 0


def test_get_backend_dummy():
    dummy = annif.backend.get_backend("dummy")
    assert dummy.config["key"] == "value"
    result = dummy.analyze('this is some text')
    assert len(result) == 0
