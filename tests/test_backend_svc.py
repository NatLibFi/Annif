"""Unit tests for the SVC backend in Annif"""

import pytest
import annif.backend
import annif.corpus
from annif.exception import NotInitializedException
from annif.exception import NotSupportedException


def test_svc_default_params(project):
    svc_type = annif.backend.get_backend("svc")
    svc = svc_type(
        backend_id='svc',
        config_params={},
        project=project)

    expected_default_params = {
        'limit': 100,
        'min_df': 1,
    }
    actual_params = svc.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_svc_suggest_no_vectorizer(project):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={},
        project=project)

    with pytest.raises(NotInitializedException):
        svc.suggest("example text")


def test_svc_train(datadir, document_corpus, project, caplog):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={},
        project=project)

    svc.train(document_corpus)
    assert svc._model is not None
    assert datadir.join('svc-model.gz').exists()
    assert 'training on a document with multiple subjects is not ' + \
           'supported by SVC; selecting one random subject.' in caplog.text


def test_svc_train_ngram(datadir, document_corpus, project):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={'ngram': 2},
        project=project)

    svc.train(document_corpus)
    assert svc._model is not None
    assert datadir.join('svc-model.gz').exists()


def test_svc_train_cached(datadir, project):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={},
        project=project)

    with pytest.raises(NotSupportedException):
        svc.train("cached")


def test_svc_train_nodocuments(datadir, project, empty_corpus):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={},
        project=project)

    with pytest.raises(NotSupportedException):
        svc.train(empty_corpus)


def test_svc_suggest(project):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={'limit': 20},
        project=project)

    results = svc.suggest("""Arkeologiaa sanotaan joskus myös...""")

    assert len(results) > 0
    assert len(results) <= 20
    hits = results.as_list(project.subjects)
    assert 'http://www.yso.fi/onto/yso/p10849' in [
        result.uri for result in hits]
    assert 'arkeologit' in [result.label for result in hits]


def test_svc_suggest_no_input(project):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={'limit': 8},
        project=project)

    results = svc.suggest("j")
    assert len(results) == 0


def test_svc_suggest_no_model(datadir, project):
    svc_type = annif.backend.get_backend('svc')
    svc = svc_type(
        backend_id='svc',
        config_params={},
        project=project)

    datadir.join('svc-model.gz').remove()
    with pytest.raises(NotInitializedException):
        svc.suggest("example text")
