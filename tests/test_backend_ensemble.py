"""Unit tests for the ensemble backend in Annif"""

import unittest.mock

import numpy as np
import pytest
from scipy.sparse import csr_array

import annif.backend
import annif.registry
from annif.backend.ensemble import EnsembleHPObjective, EnsembleOptimizer
from annif.corpus import Document
from annif.exception import NotSupportedException
from annif.suggestion import SuggestionBatch
from annif.vocab.subject_index import SubjectIndexFile
from annif.vocab.types import Subject

URI_A = "http://example.org/a"
URI_B = "http://example.org/b"
URI_C = "http://example.org/c"
URI_D = "http://example.org/d"
URI_X = "http://example.org/x"


def make_subject_index(entries):
    """Create a SubjectIndexFile from (uri, label) tuples; a None label
    marks the subject as deprecated."""

    index = SubjectIndexFile()
    for uri, label in entries:
        labels = None if label is None else {"en": label}
        index.append(Subject(uri=uri, labels=labels, notation=None))
    return index


def make_batch(rows, cols, data, n_docs, width):
    """Create a SuggestionBatch from parallel row/column/data arrays."""

    return SuggestionBatch(
        csr_array(
            (
                np.array(data, dtype=np.float32),
                (
                    np.array(rows, dtype=np.int64),
                    np.array(cols, dtype=np.int64),
                ),
            ),
            shape=(n_docs, width),
            dtype=np.float32,
        )
    )


def make_ensemble_backend(target_subjects, sources):
    """Create an ensemble backend whose project uses target_subjects and
    whose source projects (a mapping of project_id -> subject index) use
    the given source vocabularies."""

    registry = unittest.mock.Mock()
    projects = {}
    for source_id, subjects in sources.items():
        source_project = unittest.mock.Mock()
        source_project.subjects = subjects
        projects[source_id] = source_project
    ensemble_project = unittest.mock.Mock()
    ensemble_project.subjects = target_subjects
    ensemble_project.registry = registry
    projects["ensemble"] = ensemble_project
    registry.get_project.side_effect = projects.get

    ensemble_type = annif.backend.get_backend("ensemble")
    return ensemble_type(
        backend_id="ensemble",
        config_params={"sources": ",".join(sources)},
        project=ensemble_project,
    )


def test_ensemble_train(project, document_corpus):
    ensemble_type = annif.backend.get_backend("ensemble")
    ensemble = ensemble_type(
        backend_id="ensemble", config_params={"sources": "dummy"}, project=project
    )

    with pytest.raises(NotSupportedException):
        ensemble.train(document_corpus)


def test_align_batch_same_vocabulary_is_noop():
    # source and ensemble share the same subject index object, so a
    # full-width batch is already aligned and must be returned unchanged
    subjects = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    backend = make_ensemble_backend(subjects, {"src": subjects})
    batch = make_batch([0, 0, 1], [0, 1, 1], [0.4, 0.8, 1.0], 2, len(subjects))

    assert backend._align_batch(batch, "src") is batch


def test_align_batch_same_vocabulary_compact_batch():
    # even with the same subject index object, a compact batch (columns
    # numbered by active subjects only) must be expanded and aligned
    subjects = make_subject_index([(URI_A, "A"), (URI_B, None), (URI_C, "C")])
    assert len(subjects.active) == 2
    backend = make_ensemble_backend(subjects, {"src": subjects})
    # compact columns: 0 = A, 1 = C; B is deprecated and not addressable
    batch = make_batch([0, 0], [0, 1], [0.5, 1.0], 1, len(subjects.active))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 3)
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 0] = 0.5  # A
    expected[0, 2] = 1.0  # C
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_maps_compact_columns_by_uri():
    # source vocabulary is compact and in a different order than the
    # ensemble vocabulary; columns must be remapped via URI
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    batch = make_batch([0, 0, 1], [0, 1, 0], [0.4, 0.8, 1.0], 2, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (2, 3)
    assert aligned.array.dtype == np.float32
    expected = np.zeros((2, 3), dtype=np.float32)
    expected[0, 1] = 0.4  # source col 0 (B) -> target id 1
    expected[0, 0] = 0.8  # source col 1 (A) -> target id 0
    expected[1, 1] = 1.0  # source col 0 (B) -> target id 1
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_drops_unmappable_uris():
    # subjects that cannot be mapped to the ensemble vocabulary - because
    # their URI is unknown, or maps to a deprecated subject - must be
    # dropped from the aligned batch instead of crashing
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, None)])
    source = make_subject_index([(URI_D, "D"), (URI_C, "C"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    # source cols: 0 = D (unknown URI), 1 = C (deprecated in target), 2 = A
    batch = make_batch([0, 0, 0], [0, 1, 2], [0.7, 0.9, 0.2], 1, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 3)
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 0] = 0.2  # only A survives
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_raises_on_width_mismatch():
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    backend = make_ensemble_backend(target, {"src": source})
    batch = make_batch([0], [0], [0.5], 1, 5)  # matches neither vocabulary

    with pytest.raises(ValueError, match="5 columns"):
        backend._align_batch(batch, "src")


def test_align_batch_empty_batch():
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    batch = make_batch([], [], [], 2, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (2, 3)
    assert aligned.array.nnz == 0


def test_align_batch_target_width_batch_with_different_vocab():
    # the batch is as wide as the ensemble vocabulary, but the source
    # vocabulary is different; it must not be passed through unaligned
    target = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    source = make_subject_index([(URI_X, "X"), (URI_A, "A"), (URI_B, None)])
    assert len(source) != len(target)
    backend = make_ensemble_backend(target, {"src": source})
    # active source subjects: X (col 0), A (col 1)
    batch = make_batch([0, 0], [0, 1], [0.7, 0.3], 1, len(target))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 2)
    expected = np.zeros((1, 2), dtype=np.float32)
    expected[0, 0] = 0.3  # A survives; X is not in the target vocabulary
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_source_width_batch_indexed_by_subject_id():
    # full-width batch from a different vocabulary with deprecated
    # subjects: columns are source subject IDs, not active positions
    target = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    source = make_subject_index([(URI_A, "A"), (URI_B, None), (URI_C, "C")])
    assert len(source.active) != len(source)
    backend = make_ensemble_backend(target, {"src": source})
    # source subject IDs: 0 = A, 1 = B (deprecated), 2 = C
    batch = make_batch([0, 0], [0, 2], [0.4, 0.6], 1, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 2)
    expected = np.zeros((1, 2), dtype=np.float32)
    expected[0, 0] = 0.4  # A survives; C is not in the target vocabulary
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_merge_source_batches_aligns_and_averages():
    # one source is compact and needs alignment, the other shares the
    # ensemble subject index; the merged batch must combine both
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    compact_source = make_subject_index([(URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": compact_source, "full": target})
    aligned_batch = make_batch([0], [0], [1.0], 1, len(target))
    compact_batch = make_batch([0], [0], [0.5], 1, len(compact_source))

    sources = [("src", 1.0), ("full", 1.0)]
    batch_by_source = {"src": compact_batch, "full": aligned_batch}
    merged = backend._merge_source_batches(batch_by_source, sources, {"limit": 10})

    # compact source scores B=0.5, aligned source scores A=1.0
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 0] = 0.5  # A: averaged 1.0 and 0.0
    expected[0, 1] = 0.25  # B: averaged 0.5 and 0.0
    np.testing.assert_allclose(merged.array.toarray(), expected)


def test_ensemble_suggest(app):
    # integration test with the real registry: ensemble of dummy-en and
    # dummy-private, both sharing the ensemble vocabulary
    with app.app_context():
        project = annif.registry.get_project("ensemble")
        batch = project.suggest([Document(text="the cat sat on the mat")])

    assert len(batch) == 1
    hits = list(batch[0])
    assert len(hits) == 1
    assert hits[0].subject_id == 0
    assert hits[0].score == pytest.approx(1.0)


def test_ensemble_objective():
    # the objective must align compact source batches before averaging
    target = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    source = make_subject_index([(URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    # compact source batch: col 1 is A with score 1.0
    batch = make_batch([0], [1], [1.0], 1, len(source))
    args = {
        "backend": backend,
        "gold_batches": [[[0]]],  # gold subject: A
        "source_batches": [{"src": batch}],
        "sources": ["src"],
        "subject_index": target,
        "limit": "10",
        "metric": "F1 score (doc avg)",
    }
    trial = unittest.mock.Mock()
    trial.suggest_float.return_value = 0.8

    score = EnsembleHPObjective.objective(trial, args)

    trial.suggest_float.assert_called_with("src", 0.0, 1.0)
    assert score == pytest.approx(1.0)


def test_ensemble_optimizer_prepare(app):
    with app.app_context():
        project = annif.registry.get_project("ensemble")
        corpus = unittest.mock.Mock()
        corpus.doc_batches = [
            [Document(text="doc one")],
            [Document(text="doc two")],
        ]

        optimizer = EnsembleOptimizer(project.backend, corpus, "NDCG")
        args = optimizer._prepare(n_jobs=1)

    assert args["backend"] is project.backend
    assert args["sources"] == ["dummy-en", "dummy-private"]
    assert len(args["source_batches"]) == 2
