"""Unit tests for the TF-IDF backend in Annif"""

import pytest

import annif
import annif.backend
from annif.corpus import Document
from annif.exception import NotInitializedException, OperationFailedException


def test_tfidf_default_params(project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={}, project=project)

    expected_default_params = {"limit": 100}  # From AnnifBackend class
    actual_params = tfidf.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_tfidf_train(datadir, document_corpus, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={"limit": 10}, project=project)

    tfidf.train(document_corpus)
    assert tfidf._tfidf_matrix.shape[0] > 0
    assert datadir.join("tfidf-matrix.npz").exists()
    assert datadir.join("tfidf-matrix.npz").size() > 0


def test_tfidf_suggest(project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={"limit": 10}, project=project)

    results = tfidf.suggest(
        [
            Document(
                text="""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
            )
        ]
    )[0]

    assert len(results) == 10
    archaeology = project.subjects.by_uri("http://www.yso.fi/onto/yso/p1265")
    assert archaeology in [result.subject_id for result in results]


def test_suggest_params(project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={"limit": 10}, project=project)
    params = {"limit": 3}

    results = tfidf.suggest(
        [
            Document(
                text="""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
            )
        ],
        params,
    )[0]
    assert len(results) == 3


def test_tfidf_suggest_unknown(project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={"limit": 10}, project=project)

    results = tfidf.suggest([Document(text="abcdefghijk")])[0]  # unknown word

    assert len(results) == 0


def test_tfidf_suggest_old_model_error(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={"limit": 10}, project=project)

    datadir.join("tfidf-matrix.npz").remove()
    datadir.join("tfidf-index").ensure()

    with pytest.raises(OperationFailedException) as excinfo:
        tfidf.suggest([Document(text="abcdefghijk")])

    assert (
        "TFIDF models trained on Annif versions older than 1.4 cannot be loaded"
        in str(excinfo.value)
    )


def test_tfidf_suggest_no_model_error(datadir, project):
    tfidf_type = annif.backend.get_backend("tfidf")
    tfidf = tfidf_type(backend_id="tfidf", config_params={"limit": 10}, project=project)

    datadir.join("tfidf-index").remove()

    with pytest.raises(NotInitializedException) as excinfo:
        tfidf.suggest([Document(text="abcdefghijk")])

    assert f"tf-idf matrix {datadir.join('tfidf-matrix.npz')} not found" in str(
        excinfo.value
    )
