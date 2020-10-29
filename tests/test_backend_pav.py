"""Unit tests for the PAV backend in Annif"""

import logging
import py.path
import pytest
from datetime import datetime, timedelta, timezone
import annif.backend
import annif.corpus
from annif.exception import NotSupportedException


def test_pav_default_params(document_corpus, app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={},
        project=app_project)

    expected_default_params = {
        'min-docs': 10,
        'limit': 100,
    }
    actual_params = pav.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_pav_is_not_trained(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)
    assert not pav.is_trained


def test_pav_train(tmpdir, app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))

    pav.train(document_corpus)
    datadir = py.path.local(app_project.datadir)
    assert datadir.join('pav-model-dummy-fi').exists()
    assert datadir.join('pav-model-dummy-fi').size() > 0


def test_pav_train_cached(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)

    with pytest.raises(NotSupportedException):
        pav.train("cached")


def test_pav_train_nodocuments(app_project, empty_corpus):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)

    with pytest.raises(NotSupportedException) as excinfo:
        pav.train(empty_corpus)
    assert 'training backend pav with no documents' in str(excinfo.value)


def test_pav_initialize(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)

    assert pav._models is None
    pav.initialize()
    assert pav._models is not None
    # initialize a second time - this shouldn't do anything
    pav.initialize()


def test_pav_suggest(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)

    results = pav.suggest("""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan.""")

    assert len(pav._models['dummy-fi']) == 1
    assert len(results) > 0


def test_pav_train_params(tmpdir, app_project, caplog):
    logger = annif.logger
    logger.propagate = True

    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)

    tmpfile = tmpdir.join('document.tsv')
    tmpfile.write("dummy\thttp://example.org/dummy\n" +
                  "another\thttp://example.org/dummy\n" +
                  "none\thttp://example.org/none")
    document_corpus = annif.corpus.DocumentFile(str(tmpfile))
    params = {'min-docs': 5}

    with caplog.at_level(logging.DEBUG):
        pav.train(document_corpus, params)
    parameters_spec = 'creating PAV model for source dummy-fi, min_docs=5'
    assert parameters_spec in caplog.text


def test_pav_is_trained(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)
    assert pav.is_trained


def test_pav_modification_time(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id='pav',
        config_params={'limit': 50, 'min-docs': 2, 'sources': 'dummy-fi'},
        project=app_project)
    assert datetime.now(timezone.utc) - pav.modification_time < timedelta(1)
