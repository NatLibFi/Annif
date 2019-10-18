"""Unit tests for the TF-IDF backend in Annif"""

import annif
import annif.backend
import annif.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest
import unittest.mock


@pytest.fixture(scope='module')
def project(document_corpus, subject_index):
    proj = unittest.mock.Mock()
    proj.analyzer = annif.analyzer.get_analyzer('snowball(finnish)')
    proj.subjects = subject_index
    return proj


def test_tfidf_default_params(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        config_params={},
        datadir=str(datadir))

    expected_default_params = {
        'limit': 100  # From AnnifBackend class
    }
    actual_params = tfidf.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_tfidf_train(datadir, document_corpus, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        config_params={'limit': 10},
        datadir=str(datadir))

    tfidf.train(document_corpus, project)
    assert len(tfidf._index) > 0
    assert datadir.join('tfidf-index').exists()
    assert datadir.join('tfidf-index').size() > 0


def test_tfidf_suggest(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        config_params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert len(results) == 10
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]


def test_tfidf_suggest_unknown(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(
        backend_id='tfidf',
        config_params={'limit': 10},
        datadir=str(datadir))

    results = tfidf.suggest("abcdefghijk", project)  # unknown word

    assert len(results) == 0
