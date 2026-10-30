"""Unit tests for the ensemble backend in Annif"""

import unittest.mock

import numpy as np
import pytest
from scipy.sparse import csr_array

import annif.backend
import annif.registry
from annif.backend.ensemble import EnsembleOptimizer
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


def make_ensemble_backend(target_subjects, sources, backend_name="ensemble"):
    """Create an ensemble backend of the given type whose project uses
    target_subjects and whose source projects (a mapping of project_id to
    either a subject index or a ready-made mock project) use the given
    sources."""

    registry = unittest.mock.Mock()
    projects = {}
    for source_id, source in sources.items():
        if isinstance(source, unittest.mock.Mock):
            source_project = source
        else:
            source_project = unittest.mock.Mock()
            source_project.subjects = source
        projects[source_id] = source_project
    ensemble_project = unittest.mock.Mock()
    ensemble_project.subjects = target_subjects
    ensemble_project.registry = registry
    projects["ensemble"] = ensemble_project
    registry.get_project.side_effect = projects.get

    backend_type = annif.backend.get_backend(backend_name)
    return backend_type(
        backend_id=backend_name,
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


def test_align_batch_ensemble_width_batch_is_noop():
    # a batch that is already as wide as the ensemble vocabulary is
    # returned unchanged when the vocabularies are identical: this
    # covers sources sharing the ensemble vocabulary and subclasses like
    # PAV that rebuild batches at ensemble width
    subjects = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    backend = make_ensemble_backend(subjects, {"src": subjects})
    batch = make_batch([0, 0], [0, 1], [0.7, 0.3], 1, len(subjects))

    assert backend._align_batch(batch, "src") is batch


def test_align_batch_same_size_different_vocab_is_mapped():
    # a source vocabulary of the same size but with different URIs must
    # not be passed through unaligned, but mapped via URI
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_D, "D"), (URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    # source cols: 1 = B, 2 = A; D is not in the target vocabulary
    batch = make_batch([0, 0], [1, 2], [0.7, 0.3], 1, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 3)
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 1] = 0.7  # B
    expected[0, 0] = 0.3  # A
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_same_vocabulary_checks_by_uri():
    # vocabularies are the same when they have the same active subjects
    # with the same subject IDs, determined by URI
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    backend = make_ensemble_backend(
        target,
        {
            "shared": target,
            "copy": make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")]),
            "subset": make_subject_index([(URI_B, "B"), (URI_A, "A")]),
            "samesize": make_subject_index([(URI_D, "D"), (URI_B, "B"), (URI_A, "A")]),
        },
    )

    assert backend._same_vocabulary("shared")
    assert backend._same_vocabulary("copy")
    assert not backend._same_vocabulary("subset")
    assert not backend._same_vocabulary("samesize")


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


def test_align_batch_compact_batch_with_shared_vocabulary():
    # a compact batch from a source sharing the ensemble vocabulary must
    # be expanded; deprecated subjects are not addressable by compact
    # columns and cannot carry scores
    subjects = make_subject_index([(URI_A, "A"), (URI_B, None), (URI_C, "C")])
    assert len(subjects.active) == 2
    backend = make_ensemble_backend(subjects, {"src": subjects})
    # compact columns: 0 = A, 1 = C
    batch = make_batch([0, 0], [0, 1], [0.5, 1.0], 1, len(subjects.active))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 3)
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 0] = 0.5  # A
    expected[0, 2] = 1.0  # C
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_drops_unmappable_uris():
    # subjects that cannot be mapped to the ensemble vocabulary - because
    # their URI is unknown, or maps to a deprecated subject - must be
    # dropped from the aligned batch instead of crashing
    target = make_subject_index(
        [(URI_A, "A"), (URI_B, "B"), (URI_C, None), (URI_X, "X")]
    )
    source = make_subject_index([(URI_D, "D"), (URI_C, "C"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    # source cols: 0 = D (unknown URI), 1 = C (deprecated in target), 2 = A
    batch = make_batch([0, 0, 0], [0, 1, 2], [0.7, 0.9, 0.2], 1, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 4)
    expected = np.zeros((1, 4), dtype=np.float32)
    expected[0, 0] = 0.2  # only A survives
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_raises_on_width_mismatch():
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    backend = make_ensemble_backend(target, {"src": source})
    batch = make_batch([0], [0], [0.5], 1, 5)  # matches neither vocabulary

    with pytest.raises(ValueError, match="5 columns"):
        backend._align_batch(batch, "src")


def test_align_batch_full_width_batch_with_deprecated_source_subjects():
    # a full-width batch (columns are subject IDs) from a source vocabulary
    # containing deprecated subjects is mapped via URI instead of being
    # rejected; deprecated source subjects cannot carry scores
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_B, None), (URI_A, "A"), (URI_C, "C")])
    backend = make_ensemble_backend(target, {"src": source})
    # full-width: col 0 = B (deprecated), col 1 = A, col 2 = C
    batch = make_batch([0, 0], [1, 2], [0.7, 0.3], 1, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (1, 3)
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 0] = 0.7  # A
    expected[0, 2] = 0.3  # C
    np.testing.assert_array_equal(aligned.array.toarray(), expected)


def test_align_batch_empty_batch():
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    batch = make_batch([], [], [], 2, len(source))

    aligned = backend._align_batch(batch, "src")

    assert aligned.array.shape == (2, 3)
    assert aligned.array.nnz == 0


def test_merge_source_batches_aligns_and_averages():
    # both sources need alignment; the merged batch must combine the
    # aligned scores
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    backend = make_ensemble_backend(
        target,
        {
            "one": make_subject_index([(URI_B, "B"), (URI_A, "A")]),
            "two": make_subject_index([(URI_A, "A"), (URI_C, "C")]),
        },
    )
    # source "one" scores B=0.5; source "two" scores C=1.0
    batch_one = make_batch([0], [0], [0.5], 1, 2)
    batch_two = make_batch([0], [1], [1.0], 1, 2)

    sources = [("one", 1.0), ("two", 1.0)]
    batch_by_source = {"one": batch_one, "two": batch_two}
    merged = backend._merge_source_batches(batch_by_source, sources, {"limit": 10})

    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 1] = 0.25  # B: averaged 0.5 and 0.0
    expected[0, 2] = 0.5  # C: averaged 0.0 and 1.0
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


def test_ensemble_optimizer_prepare_aligns_source_batches():
    # the optimizer must align the collected source batches once, so
    # that every objective evaluation can reuse them
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_B, "B"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    source_project = backend.project.registry.get_project("src")
    # compact source batch: col 0 is B with score 0.5, col 1 is A 1.0
    source_project.suggest.return_value = make_batch(
        [0, 0], [0, 1], [0.5, 1.0], 1, len(source)
    )
    corpus = unittest.mock.Mock()
    corpus.doc_batches = [[Document(text="some text")]]

    optimizer = EnsembleOptimizer(backend, corpus, "NDCG")
    args = optimizer._prepare(n_jobs=1)

    assert args["sources"] == ["src"]
    batch = args["source_batches"][0]["src"]
    assert batch.array.shape == (1, 3)
    expected = np.zeros((1, 3), dtype=np.float32)
    expected[0, 0] = 1.0  # A
    expected[0, 1] = 0.5  # B
    np.testing.assert_array_equal(batch.array.toarray(), expected)


def test_source_subject_map_same_vocabulary_is_none():
    # when the source vocabulary matches the ensemble vocabulary by URI
    # and size, no mapping is needed
    subjects = make_subject_index([(URI_A, "A"), (URI_B, "B")])
    backend = make_ensemble_backend(subjects, {"src": subjects})

    assert backend._source_subject_map("src") is None


def test_source_subject_map_maps_by_uri_and_drops_unmappable():
    target = make_subject_index(
        [(URI_A, "A"), (URI_B, "B"), (URI_C, None), (URI_X, "X")]
    )
    source = make_subject_index([(URI_D, "D"), (URI_C, "C"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})

    # source col 0 (D) is unknown, col 1 (C) maps to a deprecated subject,
    # col 2 (A) maps to target subject 0
    np.testing.assert_array_equal(
        backend._source_subject_map("src"), np.array([-1, -1, 0])
    )


def test_align_source_vector_maps_scores():
    target = make_subject_index([(URI_A, "A"), (URI_B, "B"), (URI_C, "C")])
    source = make_subject_index([(URI_C, "C"), (URI_A, "A")])
    backend = make_ensemble_backend(target, {"src": source})
    subject_map = backend._source_subject_map("src")
    vector = np.array([0.4, 0.8], dtype=np.float32)

    aligned = backend._align_source_vector(vector, subject_map)

    expected = np.array([0.8, 0.0, 0.4], dtype=np.float32)
    np.testing.assert_array_equal(aligned, expected)
    # a None map passes the vector through unchanged
    assert backend._align_source_vector(vector, None) is vector
