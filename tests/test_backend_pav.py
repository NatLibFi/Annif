"""Unit tests for the PAV backend in Annif"""

import logging
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import py.path
import pytest

import annif.backend
import annif.util
from annif.corpus import Document, DocumentFileTSV
from annif.exception import NotInitializedException, NotSupportedException

from .test_backend_ensemble import (
    URI_A,
    URI_B,
    URI_X,
    make_batch,
    make_gold_corpus,
    make_misaligned_source_backend,
)


def test_pav_default_params(document_corpus, app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(backend_id="pav", config_params={}, project=app_project)

    expected_default_params = {
        "min-docs": 10,
        "limit": 100,
    }
    actual_params = pav.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_pav_is_not_trained(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )
    assert not pav.is_trained


def test_pav_train(tmpdir, app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "dummy\thttp://example.org/dummy\n"
        + "another\thttp://example.org/dummy\n"
        + "none\thttp://example.org/none"
    )
    document_corpus = DocumentFileTSV(str(tmpfile), app_project.subjects)

    pav.train(document_corpus)
    datadir = py.path.local(app_project.datadir)
    assert datadir.join("pav-model-dummy-fi").exists()
    assert datadir.join("pav-model-dummy-fi").size() > 0


def test_pav_train_cached(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    with pytest.raises(NotSupportedException):
        pav.train("cached")


def test_pav_train_nodocuments(app_project, empty_corpus):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    with pytest.raises(NotSupportedException) as excinfo:
        pav.train(empty_corpus)
    assert "training backend pav with no documents" in str(excinfo.value)


def test_pav_initialize(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    assert pav._models is None
    pav.initialize()
    assert pav._models is not None
    # initialize a second time - this shouldn't do anything
    pav.initialize()


def test_pav_suggest(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    results = pav.suggest(
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

    assert len(pav._models["dummy-fi"]) == 1
    assert len(results) > 0
    assert list(results)[0].score == pytest.approx(2 / 3)  # PAV recalculated score


def test_pav_train_params(tmpdir, app_project, caplog):
    logger = annif.logger
    logger.propagate = True

    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "dummy\thttp://example.org/dummy\n"
        + "another\thttp://example.org/dummy\n"
        + "none\thttp://example.org/none"
    )
    document_corpus = DocumentFileTSV(str(tmpfile), app_project.subjects)
    params = {"min-docs": 5}

    with caplog.at_level(logging.DEBUG):
        pav.train(document_corpus, params)
    parameters_spec = "creating PAV model for source dummy-fi, min_docs=5"
    assert parameters_spec in caplog.text


def test_pav_suggest_after_min_docs(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )

    results = pav.suggest(
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

    assert len(pav._models["dummy-fi"]) == 0
    assert len(results) > 0
    print(results)
    print(list(results)[0])
    assert list(results)[0].score == 1.0  # original score from dummy-fi project


def test_pav_is_trained(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )
    assert pav.is_trained


def test_pav_modification_time(app_project):
    pav_type = annif.backend.get_backend("pav")
    pav = pav_type(
        backend_id="pav",
        config_params={"limit": 50, "min-docs": 2, "sources": "dummy-fi"},
        project=app_project,
    )
    assert datetime.now(timezone.utc) - pav.modification_time < timedelta(1)


def test_suggest_train_corpus_aligns_source_vocabularies(tmpdir):
    # regression test: source projects with a different vocabulary than the
    # ensemble project must have their scores aligned by subject URI, and
    # the gold labels must be kept in the ensemble vocabulary space
    pav, source_project = make_misaligned_source_backend(
        "pav", [(URI_B, "B source"), (URI_X, "X source")]
    )
    corpus = make_gold_corpus(tmpdir, pav.project.subjects)

    subject_map = pav._source_subject_map("src")
    scores, true = pav._suggest_train_corpus(source_project, "src", subject_map, corpus)

    assert scores.shape == (2, 3)
    expected_scores = np.zeros((2, 3), dtype=np.float32)
    expected_scores[:, 1] = 0.7  # URI_B; URI_X cannot be mapped
    np.testing.assert_array_equal(scores.toarray(), expected_scores)

    expected_true = np.zeros((2, 3), dtype=bool)
    expected_true[0, 0] = True  # URI_A
    expected_true[1, 2] = True  # URI_C
    np.testing.assert_array_equal(true.toarray(), expected_true)


def test_merge_source_batches_aligns_source_vocabularies():
    pav, source_project = make_misaligned_source_backend(
        "pav", [(URI_B, "B source"), (URI_A, "A source")]
    )
    # regression models are keyed by ensemble subject IDs; only B has one
    reg = mock.Mock()
    reg.predict = lambda values: [min(1.0, value + 0.1) for value in values]
    pav._models = {"src": {1: reg}}

    # source scores: B=0.7, A=0.3
    batch = make_batch([0, 0], [0, 1], [0.7, 0.3], 1, len(source_project.subjects))
    merged = pav._merge_source_batches({"src": batch}, [("src", 1.0)], {"limit": 10})

    assert merged.array.shape == (1, 3)
    hits = list(merged[0])
    # B (regressed score) ranks before A (kept raw)
    assert [hit.subject_id for hit in hits] == [1, 0]
    assert hits[0].score == pytest.approx(0.8)
    assert hits[1].score == pytest.approx(0.3)


def test_create_pav_model_with_mismatched_source_vocabulary(tmpdir, monkeypatch):
    # models must be keyed by ensemble subject IDs even when the source
    # vocabulary differs from the ensemble vocabulary, and only concepts
    # that the source can suggest get models
    pav, _source_project = make_misaligned_source_backend(
        "pav", [(URI_X, "X source"), (URI_A, "A source")]
    )
    corpus = make_gold_corpus(tmpdir, pav.project.subjects)

    saved = {}
    monkeypatch.setattr(
        annif.util, "atomic_save", lambda obj, *args, **kwargs: saved.update(obj)
    )

    pav._create_pav_model("src", min_docs=1, corpus=corpus)

    # only A (ensemble subject 0) is both reachable from the source and has
    # gold standard examples; C has gold examples but the source cannot
    # suggest it, so no model is created for it
    assert list(saved.keys()) == [0]


def test_suggest_train_corpus_rejects_narrow_source_batch(tmpdir):
    # a source model trained on an older, smaller vocabulary emits batches
    # narrower than the source vocabulary; PAV training must reject them
    # loudly instead of silently misassigning scores
    pav, source_project = make_misaligned_source_backend(
        "pav", [(URI_B, "B source"), (URI_X, "X source")]
    )
    source_project.suggest = lambda docs, params=None: make_batch(
        [0], [0], [0.7], len(docs), 1
    )
    corpus = make_gold_corpus(tmpdir, pav.project.subjects)

    subject_map = pav._source_subject_map("src")
    with pytest.raises(NotInitializedException, match="Retrain"):
        pav._suggest_train_corpus(source_project, "src", subject_map, corpus)
