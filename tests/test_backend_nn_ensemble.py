"""Unit tests for the nn_ensemble backend in Annif"""

import time
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import py.path
import pytest

import annif.backend
from annif.corpus import Document, DocumentFileTSV, LimitingDocumentCorpus
from annif.exception import (
    NotInitializedException,
    NotSupportedException,
    OperationFailedException,
)

from .test_backend_ensemble import make_batch, make_ensemble_backend, make_subject_index
from .util import umask_context

pytest.importorskip("annif.backend.nn_ensemble")
lmdb = pytest.importorskip("lmdb")


def test_lmdb_idx_to_key_to_idx():
    assert annif.backend.nn_ensemble.idx_to_key(42) == b"00000042"
    assert annif.backend.nn_ensemble.key_to_idx(b"00000042") == 42


def test_nn_ensemble_initialize_parallel(project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble", config_params={"sources": "dummy-en"}, project=project
    )

    nn_ensemble.initialize(parallel=True)
    # model is still not loaded since we're preparing for parallel execution
    assert nn_ensemble._model is None


def test_nn_ensemble_suggest_no_model(project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble", config_params={"sources": "dummy-en"}, project=project
    )

    with pytest.raises(NotInitializedException):
        nn_ensemble.suggest([Document(text="example text")])[0]


def test_nn_ensemble_is_not_trained(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )
    assert not nn_ensemble.is_trained


def test_set_lmdb_map_size(registry, tmpdir):
    project = registry.get_project("dummy-en")
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en", "max-epochs": 1, "lmdb_map_size": 1},
        project=project,
    )
    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "dummy\thttp://example.org/dummy\n"
        + "another\thttp://example.org/dummy\n"
        + "none\thttp://example.org/none\n" * 40
    )
    document_corpus = DocumentFileTSV(str(tmpfile), project.subjects)

    with pytest.raises(lmdb.MapFullError):
        nn_ensemble.train(document_corpus)


def test_nn_ensemble_train_and_learn(registry, tmpdir):
    project = registry.get_project("dummy-en")
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en", "max-epochs": 1},
        project=project,
    )

    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "dummy\thttp://example.org/dummy\n"
        + "another\thttp://example.org/dummy\n"
        + "none\thttp://example.org/none\n" * 40
    )
    document_corpus = DocumentFileTSV(str(tmpfile), project.subjects)
    datadir = py.path.local(project.datadir)

    with umask_context(0o007):
        nn_ensemble.train(document_corpus)
        # verify LMDB database creation and expected permissions when using umask 0007
        assert datadir.join("nn-train.mdb").exists()
        assert datadir.join("nn-train.mdb").stat().mode & 0o777 == 0o770
        assert datadir.join("nn-train.mdb/data.mdb").exists()
        assert datadir.join("nn-train.mdb/data.mdb").stat().mode & 0o777 == 0o660

    assert datadir.join("nn-model.pt").exists()
    assert datadir.join("nn-model.pt").size() > 0

    # test online learning
    modelfile = datadir.join("nn-model.pt")

    old_size = modelfile.size()
    old_mtime = modelfile.mtime()

    time.sleep(0.1)  # make sure the timestamp has a chance to increase

    # Learning is typically performed on one document at a time
    document_corpus_single_doc = LimitingDocumentCorpus(document_corpus, 1)
    nn_ensemble.learn(document_corpus_single_doc)

    assert modelfile.size() != old_size or modelfile.mtime() != old_mtime


def test_nn_ensemble_train_cached(registry):
    # make sure we have the cached training data from the previous test
    project = registry.get_project("dummy-en")
    datadir = py.path.local(project.datadir)
    assert datadir.join("nn-train.mdb").exists()

    datadir.join("nn-model.pt").remove()

    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=project,
    )

    nn_ensemble.train("cached")

    assert datadir.join("nn-model.pt").exists()
    assert datadir.join("nn-model.pt").size() > 0


def test_nn_ensemble_train_and_learn_params(registry, tmpdir, capfd):
    project = registry.get_project("dummy-en")
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en", "max-epochs": 3},
        project=project,
    )

    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "dummy\thttp://example.org/dummy\n"
        + "another\thttp://example.org/dummy\n"
        + "none\thttp://example.org/none"
    )
    document_corpus = DocumentFileTSV(str(tmpfile), project.subjects)

    train_params = {"max-epochs": 3}
    nn_ensemble.train(document_corpus, train_params)
    _, err = capfd.readouterr()
    assert "Epoch 3/3" in err

    learn_params = {"learn-epochs": 2}
    nn_ensemble.learn(document_corpus, learn_params)
    _, err = capfd.readouterr()
    assert "Epoch 2/2" in err


def test_nn_ensemble_is_trained(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )
    assert nn_ensemble.is_trained


def test_nn_ensemble_modification_time(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )
    assert datetime.now(timezone.utc) - nn_ensemble.modification_time < timedelta(1)


@mock.patch("annif.backend.nn_ensemble.NNEnsembleModel.load", side_effect=Exception)
def test_nn_ensemble_initialize_error(load, app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )
    assert nn_ensemble._model is None
    with pytest.raises(
        OperationFailedException,
        match=r"loading model from .*; original error message: .*",
    ):
        nn_ensemble.initialize()
    assert load.called


def test_nn_ensemble_initialize(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )

    assert nn_ensemble._model is None
    nn_ensemble.initialize()
    assert nn_ensemble._model is not None
    # initialize a second time - this shouldn't do anything
    nn_ensemble.initialize()


def test_nn_ensemble_default_params(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )

    expected_default_params = {
        "lr": 0.003,
        "limit": 100,
    }
    actual_params = nn_ensemble.params
    for param, val in expected_default_params.items():
        assert param in actual_params and actual_params[param] == val


def test_nn_ensemble_train_nodocuments(project, empty_corpus):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble", config_params={"sources": "dummy-en"}, project=project
    )
    with pytest.raises(NotSupportedException):
        nn_ensemble.train(empty_corpus)


def test_nn_ensemble_suggest(app_project):
    nn_ensemble_type = annif.backend.get_backend("nn_ensemble")
    nn_ensemble = nn_ensemble_type(
        backend_id="nn_ensemble",
        config_params={"sources": "dummy-en"},
        project=app_project,
    )

    results = nn_ensemble.suggest(
        [
            Document(
                text="""Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen
        tiede tai oikeammin joukko tieteitä, jotka tutkivat ihmisen
        menneisyyttä. Tutkimusta tehdään analysoimalla muinaisjäännöksiä
        eli niitä jälkiä, joita ihmisten toiminta on jättänyt maaperään
        tai vesistöjen pohjaan."""
            )
        ]
    )[0]

    assert nn_ensemble._model is not None
    assert len(results) > 0


URI_A = "http://example.org/a"
URI_B = "http://example.org/b"
URI_C = "http://example.org/c"
URI_X = "http://example.org/x"


def test_corpus_to_vectors_aligns_source_vocabularies(tmpdir):
    # regression test: source projects with a different vocabulary than the
    # ensemble project must have their score vectors aligned by subject URI
    # before they are stacked into the training data
    target_subjects = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source_subjects = make_subject_index([(URI_B, "B source"), (URI_X, "X source")])

    source_project = mock.Mock()
    source_project.subjects = source_subjects
    source_project.suggest = lambda docs, params=None: make_batch(
        [0, 0], [0, 1], [0.7, 0.3], len(docs), len(source_subjects)
    )

    nn_ensemble = make_ensemble_backend(
        target_subjects, {"src": source_project}, backend_name="nn_ensemble"
    )

    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write("dummy\thttp://example.org/a\nanother\thttp://example.org/c\n")
    corpus = DocumentFileTSV(str(tmpfile), target_subjects)

    env = lmdb.open(
        str(tmpdir.mkdir("mdb")), map_size=1024 * 1024, writemap=True, mode=0o775
    )
    with env.begin(write=True, buffers=True) as txn:
        seq = annif.backend.nn_ensemble.LMDBDataset(txn)
        nn_ensemble._corpus_to_vectors(corpus, seq, 1)

        assert len(seq) == 2
        inputs, targets = seq[0]
        assert inputs.shape == (1, 3)
        # source subject 0 (URI_B) maps to ensemble subject 1; URI_X cannot
        # be mapped and its score is dropped
        expected = np.zeros((1, 3), dtype=np.float32)
        expected[0, 1] = np.log1p(0.7)
        np.testing.assert_allclose(inputs.numpy(), expected, rtol=1e-6)
        # gold standard subject for the first document is URI_A
        expected_target = np.zeros(3, dtype=np.float32)
        expected_target[0] = np.log1p(1.0)
        np.testing.assert_allclose(targets.numpy(), expected_target, rtol=1e-6)
    env.close()


def test_merge_source_batches_aligns_source_vocabularies():
    target_subjects = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source_subjects = make_subject_index([(URI_B, "B source"), (URI_A, "A source")])

    source_project = mock.Mock()
    source_project.subjects = source_subjects

    nn_ensemble = make_ensemble_backend(
        target_subjects, {"src": source_project}, backend_name="nn_ensemble"
    )
    nn_ensemble._create_model([("src", 1.0)])

    # source scores: B=0.7, A=0.3
    batch = make_batch([0, 0], [0, 1], [0.7, 0.3], 1, len(source_subjects))
    merged = nn_ensemble._merge_source_batches(
        {"src": batch}, [("src", 1.0)], {"limit": 10}
    )

    assert merged.array.shape == (1, 3)
    # suggestions are ranked by aligned scores: B before A before unmapped C
    assert [hit.subject_id for hit in merged[0]] == [1, 0, 2]


def test_merge_source_batches_rows_follow_sources_order():
    # the model weights are indexed by the sources order, so the rows of
    # the stacked score vectors must follow it even when the batch dict
    # is ordered differently
    target_subjects = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source_project = mock.Mock()
    source_project.subjects = target_subjects
    nn_ensemble = make_ensemble_backend(
        target_subjects,
        {"one": source_project, "two": source_project},
        backend_name="nn_ensemble",
    )
    nn_ensemble._create_model([("one", 1.0), ("two", 0.0)])

    # source "one" scores A, source "two" scores B; dict order is reversed
    batch_one = make_batch([0], [0], [0.9], 1, len(target_subjects))
    batch_two = make_batch([0], [1], [0.9], 1, len(target_subjects))
    merged = nn_ensemble._merge_source_batches(
        {"two": batch_two, "one": batch_one},
        [("one", 1.0), ("two", 0.0)],
        {"limit": 10},
    )

    # weight 1.0 is on source "one", so A must be the top suggestion
    assert list(merged[0])[0].subject_id == 0
