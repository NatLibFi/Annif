"""Unit tests for Annif exception classes"""

import pytest
from annif.exception import AnnifException
from click import ClickException


def test_annifexception_not_instantiable():
    with pytest.raises(TypeError):
        AnnifException("test message")


def test_annifexception_is_clickexception():

    # we need to define a custom class to make an instantiable exception
    class CustomException(AnnifException):
        @property
        def prefix(self):
            return "my prefix"

    exc = CustomException("test message")
    assert isinstance(exc, ClickException)
