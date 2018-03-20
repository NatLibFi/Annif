"""Unit tests for backends in Annif"""

import pytest
import annif
import annif.backend


def test_get_backend_nonexistent():
    with pytest.raises(ValueError):
        annif.backend.get_backend_type("nonexistent")


def test_get_backend_type_dummy():
    dummy_type = annif.backend.get_backend_type("dummy")
    dummy = dummy_type(backend_id='dummy', params={})
    result = dummy.analyze('this is some text')
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0


def test_get_backend_dummy():
    dummy = annif.backend.get_backend("dummy")
    assert dummy.params["key"] == "value"
    result = dummy.analyze('this is some text')
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0


def test_get_backend_tfidf_fi():
    tfidf_fi = annif.backend.get_backend("tfidf-fi")
    assert tfidf_fi.params["analyzer"] == "snowball(finnish)"


def test_get_backend_tfidf_en():
    tfidf_en = annif.backend.get_backend("tfidf-en")
    assert tfidf_en.params["analyzer"] == "snowball(english)"


def test_project_datadir(tmpdir):
    annif.cxapp.app.config['DATADIR'] = str(tmpdir)
    dummy = annif.backend.get_backend('dummy')
    datadir = dummy._get_datadir()
    assert datadir == tmpdir.join('backends/dummy')
    assert tmpdir.join('backends').exists()
    assert tmpdir.join('backends/dummy').exists()
