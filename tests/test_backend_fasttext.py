"""Unit tests for the fastText backend in Annif"""

import logging

import pytest

import annif.backend
from annif.corpus import Document, DocumentFileTSV
from annif.exception import NotSupportedException

fasttext = pytest.importorskip("annif.backend.fasttext")


def test_fasttext_default_params(project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(backend_id="fasttext", config_params={}, project=project)

    expected_default_params = {
        "limit": 100,
        "chunksize": 1,
        "dim": 100,
        "lr": 0.25,
        "epoch": 5,
        "loss": "hs",
    }
    actual_params = fasttext.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_fasttext_train(document_corpus, project, datadir):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={"limit": 50, "dim": 100, "lr": 0.25, "epoch": 20, "loss": "hs"},
        project=project,
    )

    fasttext.train(document_corpus)
    assert fasttext._model is not None
    assert datadir.join("fasttext-model").exists()
    assert datadir.join("fasttext-model").size() > 0


def test_fasttext_train_cached_jobs(project, datadir):
    assert datadir.join("fasttext-train.txt").exists()
    datadir.join("fasttext-model").remove()
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={"limit": 50, "dim": 100, "lr": 0.25, "epoch": 20, "loss": "hs"},
        project=project,
    )

    fasttext.train("cached", jobs=2)
    assert fasttext._model is not None
    assert datadir.join("fasttext-model").exists()
    assert datadir.join("fasttext-model").size() > 0


def test_fasttext_train_unknown_subject(tmpdir, datadir, project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={"limit": 50, "dim": 100, "lr": 0.25, "epoch": 20, "loss": "hs"},
        project=project,
    )

    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "nonexistent\thttp://example.com/nonexistent\n"
        + "arkeologia\thttp://www.yso.fi/onto/yso/p1265"
    )
    document_corpus = DocumentFileTSV(str(tmpfile), project.subjects)

    fasttext.train(document_corpus)
    assert fasttext._model is not None
    assert datadir.join("fasttext-model").exists()
    assert datadir.join("fasttext-model").size() > 0


def test_fasttext_train_nodocuments(project, empty_corpus):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={"limit": 50, "dim": 100, "lr": 0.25, "epoch": 20, "loss": "hs"},
        project=project,
    )

    with pytest.raises(NotSupportedException) as excinfo:
        fasttext.train(empty_corpus)
    assert "training backend fasttext with no documents" in str(excinfo.value)


def test_train_fasttext_params(document_corpus, project, caplog):
    logger = annif.logger
    logger.propagate = True

    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={"limit": 51, "dim": 101, "lr": 0.21, "epoch": 21, "loss": "hs"},
        project=project,
    )
    params = {"dim": 1, "lr": 42.1, "epoch": 0}

    with caplog.at_level(logging.DEBUG, logger="annif"):
        fasttext.train(document_corpus, params)
    parameters_heading = "Backend fasttext: Model parameters:"
    assert parameters_heading in caplog.text
    for line in caplog.text.splitlines():
        if parameters_heading in line:
            assert "'dim': 1" in line
            assert "'lr': 42.1" in line
            assert "'epoch': 0" in line


def test_fasttext_train_pretrained(
    datadir, document_corpus, project, pretrained_vectors
):
    assert pretrained_vectors.exists()
    assert pretrained_vectors.size() > 0

    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={
            "limit": 50,
            "dim": 100,
            "lr": 0.25,
            "epoch": 20,
            "loss": "hs",
            "pretrainedVectors": str(pretrained_vectors),
        },
        project=project,
    )

    fasttext.train(document_corpus)
    assert fasttext._model is not None
    assert datadir.join("fasttext-model").exists()
    assert datadir.join("fasttext-model").size() > 0


def test_fasttext_train_pretrained_wrong_dim(
    datadir, document_corpus, project, pretrained_vectors
):
    assert pretrained_vectors.exists()
    assert pretrained_vectors.size() > 0

    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={
            "limit": 50,
            "dim": 50,
            "lr": 0.25,
            "epoch": 20,
            "loss": "hs",
            "pretrainedVectors": str(pretrained_vectors),
        },
        project=project,
    )

    with pytest.raises(ValueError):
        fasttext.train(document_corpus)
    assert fasttext._model is None


def test_fasttext_suggest(project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={
            "limit": 50,
            "chunksize": 1,
            "dim": 100,
            "lr": 0.25,
            "epoch": 20,
            "loss": "hs",
        },
        project=project,
    )

    results = fasttext.suggest(
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

    assert len(results) > 0
    archaeology = project.subjects.by_uri("http://www.yso.fi/onto/yso/p1265")
    assert archaeology in [result.subject_id for result in results]


def test_fasttext_suggest_empty_chunks(project):
    fasttext_type = annif.backend.get_backend("fasttext")
    fasttext = fasttext_type(
        backend_id="fasttext",
        config_params={
            "limit": 50,
            "chunksize": 1,
            "dim": 100,
            "lr": 0.25,
            "epoch": 20,
            "loss": "hs",
        },
        project=project,
    )

    results = fasttext.suggest([Document(text="")])[0]

    assert len(results) == 0
