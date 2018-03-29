"""Unit tests for backends in Annif"""

import py.path
import pytest
import annif
import annif.backend


def test_get_backend_nonexistent():
    with pytest.raises(ValueError):
        annif.backend.get_backend_type("nonexistent")


def test_get_backend_type_dummy(app):
    dummy_type = annif.backend.get_backend_type("dummy")
    dummy = dummy_type(backend_id='dummy', params={},
                       datadir=app.config['DATADIR'])
    result = dummy.analyze('this is some text', project=None)
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0


def test_get_backend_dummy(app):
    with app.app_context():
        dummy = annif.backend.get_backend("dummy")
    assert dummy.params["key"] == "value"
    result = dummy.analyze('this is some text', project=None)
    assert len(result) == 1
    assert result[0].uri == 'http://example.org/dummy'
    assert result[0].label == 'dummy'
    assert result[0].score == 1.0


def test_get_backend_tfidf_fi(app):
    with app.app_context():
        tfidf_fi = annif.backend.get_backend("tfidf-fi")
    assert tfidf_fi.params["chunksize"] == "10"
    assert tfidf_fi.params["limit"] == "10"


def test_get_backend_tfidf_en(app):
    with app.app_context():
        tfidf_en = annif.backend.get_backend("tfidf-en")
    assert tfidf_en.params["chunksize"] == "10"
    assert tfidf_en.params["limit"] == "10"


def test_backend_datadir(app):
    with app.app_context():
        dummy = annif.backend.get_backend('dummy')
    datadir = py.path.local(app.config['DATADIR'])
    bedatadir = dummy._get_datadir()
    assert str(datadir.join('backends/dummy')) == str(py.path.local(bedatadir))
    assert datadir.join('backends').exists()
    assert datadir.join('backends/dummy').exists()
