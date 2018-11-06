"""Unit tests for backends in Annif"""

import os
import pytest
import annif
import annif.backend


def test_get_backend_nonexistent():
    with pytest.raises(ValueError):
        annif.backend.get_backend("nonexistent")


def test_get_backend_dummy(app, project):
    dummy_type = annif.backend.get_backend("dummy")
    dummy = dummy_type(backend_id='dummy', params={},
                       datadir=app.config['DATADIR'])
    result = dummy.analyze(text='this is some text', project=project)
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0
