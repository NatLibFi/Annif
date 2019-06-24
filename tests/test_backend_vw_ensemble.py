"""Unit tests for the vw_ensemble backend in Annif"""

import pytest
import annif.backend
import annif.corpus

pytest.importorskip("annif.backend.vw_ensemble")


def test_vw_ensemble_train(app, datadir, tmpdir, fulltext_corpus, project):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'tfidf-fi'},
        datadir=str(datadir))

    with app.app_context():
        vw_ensemble.train(fulltext_corpus, project)
    assert datadir.join('vw-train.txt').exists()
    assert datadir.join('vw-train.txt').size() > 0
    assert datadir.join('vw-model').exists()
    assert datadir.join('vw-model').size() > 0


def test_vw_ensemble_initialize(app, datadir):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'tfidf-fi'},
        datadir=str(datadir))

    assert vw_ensemble._model is None
    with app.app_context():
        vw_ensemble.initialize()
    assert vw_ensemble._model is not None
    # initialize a second time - this shouldn't do anything
    with app.app_context():
        vw_ensemble.initialize()


def test_vw_ensemble_suggest(app, datadir, project):
    vw_ensemble_type = annif.backend.get_backend("vw_ensemble")
    vw_ensemble = vw_ensemble_type(
        backend_id='vw_ensemble',
        params={'sources': 'tfidf-fi'},
        datadir=str(datadir))

    results = vw_ensemble.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""", project)

    assert vw_ensemble._model is not None
    assert len(results) > 0
