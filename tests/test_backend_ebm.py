"""Unit tests for the EBM backend in Annif"""

import numpy as np
import pytest

import annif
import annif.backend
from annif.corpus import Document
from annif.exception import NotInitializedException, NotSupportedException


class MockTransformer:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 1024))


ebm = pytest.importorskip("annif.backend.ebm")

_backend_conf = {
    "language": "fi",
    "limit": 10,
    "embedding_model_name": MockTransformer(),
    "embedding_dimensions": 1024,
}


def test_ebm_default_params(project):
    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(backend_id="ebm", config_params={}, project=project)

    expected_default_params = {"limit": 100}
    actual_params = ebm.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_ebm_not_initialized(project):
    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(backend_id="ebm", config_params={}, project=project)
    with pytest.raises(NotInitializedException):
        ebm.suggest([Document(text="example text")])[0]


def test_ebm_train_no_documents(project, empty_corpus):
    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(
        backend_id="ebm",
        config_params=_backend_conf,
        project=project,
    )

    with pytest.raises(NotSupportedException) as excinfo:
        ebm.train(empty_corpus)
    assert "training backend ebm with no documents" in str(excinfo.value)


def test_ebm_train(datadir, fulltext_corpus, project):
    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(
        backend_id="ebm",
        config_params=_backend_conf,
        project=project,
    )

    ebm.train(fulltext_corpus)

    assert ebm._model is not None
    assert datadir.join("ebm-duck.db").exists()
    assert datadir.join("ebm-duck.db").size() > 0
    assert datadir.join("ebm-train.gz").exists()
    assert datadir.join("ebm-train.gz").size() > 0
    assert datadir.join("ebm-model.gz").exists()
    assert datadir.join("ebm-model.gz").size() > 0


def test_ebm_suggest(project):
    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(backend_id="ebm", config_params=_backend_conf, project=project)

    results = ebm.suggest(
        [
            Document(
                text="""Arkeologia on tieteenala, jota sanotaan joskus
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
            )
        ]
    )[0]

    assert len(results) > 0
    assert len(results) <= 10


def test_ebm_train_cached(datadir, project):
    modelfile = datadir.join("ebm-model.gz")
    assert modelfile.exists()
    dbfile = datadir.join("ebm-duck.db")
    assert dbfile.exists()

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(
        backend_id="ebm",
        config_params=_backend_conf,
        project=project,
    )

    ebm.train("cached")
    assert ebm._model is not None
    assert modelfile.exists()
    assert modelfile.size() > 0
    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_ebm_train_cached_no_train_data(datadir, project):
    modelfile = datadir.join("ebm-model.gz")
    assert modelfile.exists()
    trainfile = datadir.join("ebm-train.gz")
    trainfile.remove()

    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(
        backend_id="ebm",
        config_params=_backend_conf,
        project=project,
    )

    with pytest.raises(NotInitializedException):
        ebm.train("cached")


def test_ebm_train_cached_no_db(datadir, project):
    modelfile = datadir.join("ebm-model.gz")
    assert modelfile.exists()
    dbfile = datadir.join("ebm-duck.db")
    dbfile.remove()

    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(
        backend_id="ebm",
        config_params=_backend_conf,
        project=project,
    )

    with pytest.raises(NotInitializedException):
        ebm.train("cached")


def test_ebm_suggest_no_model(datadir, project):
    ebm_type = annif.backend.get_backend("ebm")
    ebm = ebm_type(backend_id="ebm", config_params=_backend_conf, project=project)

    datadir.join("ebm-model.gz").remove()
    with pytest.raises(NotInitializedException):
        ebm.suggest([Document(text="example text")])
