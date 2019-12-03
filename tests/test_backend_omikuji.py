"""Unit tests for the Omikuji backend in Annif"""

import pytest
import annif.backend
import annif.corpus
from annif.exception import ConfigurationException
from annif.exception import NotInitializedException
from annif.exception import NotSupportedException

pytest.importorskip("annif.backend.omikuji")


def test_omikuji_default_params(project):
    omikuji_type = annif.backend.get_backend("omikuji")
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    expected_default_params = {
        'limit': 100,
    }
    actual_params = omikuji.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_omikuji_suggest_no_model(project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    with pytest.raises(NotInitializedException):
        results = omikuji.suggest("example text")


def test_omikuji_train(datadir, document_corpus, project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    omikuji.train(document_corpus)
    assert omikuji._model is not None
    assert datadir.join('omikuji-model').exists()
    assert datadir.join('omikuji-model').size() > 0


def test_omikuji_train_nodocuments(datadir, project, empty_corpus):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={},
        project=project)

    with pytest.raises(NotSupportedException):
        omikuji.train(empty_corpus)


def test_omikuji_suggest(project):
    omikuji_type = annif.backend.get_backend('omikuji')
    omikuji = omikuji_type(
        backend_id='omikuji',
        config_params={'chunksize': 4, 'probabilities': 1},
        project=project)

    results = omikuji.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(results) > 0
    assert 'http://www.yso.fi/onto/yso/p1265' in [
        result.uri for result in results]
    assert 'arkeologia' in [result.label for result in results]
