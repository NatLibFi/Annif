"""Unit tests for suggestion processing in Annif"""

import numpy as np
from scipy.sparse import csr_array

from annif.suggestion import (
    ListSuggestionResult,
    SubjectSuggestion,
    VectorSuggestionResult,
    filter_suggestion,
)


def generate_suggestions(n, subject_index):
    suggestions = []
    for i in range(n):
        suggestions.append(SubjectSuggestion(subject_id=i, score=1.0 / (i + 1)))
    return ListSuggestionResult(suggestions)


def test_filter_suggestion_limit():
    pred = csr_array([[0, 1, 3, 2], [1, 4, 3, 0]])
    filtered = filter_suggestion(pred, limit=2)
    assert filtered.toarray().tolist() == [[0, 0, 3, 2], [0, 4, 3, 0]]


def test_filter_suggestion_threshold():
    pred = csr_array([[0, 1, 3, 2], [1, 4, 3, 0]])
    filtered = filter_suggestion(pred, threshold=2)
    assert filtered.toarray().tolist() == [[0, 0, 3, 2], [0, 4, 3, 0]]


def test_filter_suggestion_limit_and_threshold():
    pred = csr_array([[0, 1, 3, 2], [1, 4, 3, 0]])
    filtered = filter_suggestion(pred, limit=2, threshold=3)
    assert filtered.toarray().tolist() == [[0, 0, 3, 0], [0, 4, 3, 0]]


def test_list_suggestion_result_vector(subject_index):
    suggestions = ListSuggestionResult(
        [
            # subject: seals (labels)
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p7141"),
                score=1.0,
            ),
            # subject: Vikings
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p6479"),
                score=0.5,
            ),
        ]
    )
    vector = suggestions.as_vector(len(subject_index))
    assert isinstance(vector, np.ndarray)
    assert len(vector) == len(subject_index)
    assert vector.sum() == 1.5
    for subject_id, score in enumerate(vector):
        if subject_index[subject_id].labels is None:  # deprecated
            assert score == 0.0
        elif subject_index[subject_id].labels["fi"] == "sinetit":
            assert score == 1.0
        elif subject_index[subject_id].labels["fi"] == "viikingit":
            assert score == 0.5
        else:
            assert score == 0.0


def test_list_suggestions_vector_enforce_score_range(subject_index):
    suggestions = ListSuggestionResult(
        [
            # subject: seals (labels)
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p7141"),
                score=1.5,
            ),
            # subject: Vikings
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p6479"),
                score=1.0,
            ),
            # subject: excavations
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p14173"),
                score=0.5,
            ),
            # subject: runestones
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p14588"),
                score=0.0,
            ),
            # subject: Viking Age
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p12738"),
                score=-0.5,
            ),
        ]
    )
    vector = suggestions.as_vector(len(subject_index))
    assert vector.sum() == 2.5
    found = 0
    for subject_id, score in enumerate(vector):
        if subject_index[subject_id].labels is None:
            continue  # skip deprecated subjects
        if subject_index[subject_id].labels["fi"] == "sinetit":
            assert score == 1.0
            found += 1
        elif subject_index[subject_id].labels["fi"] == "viikinkiaika":
            assert score == 0.0
            found += 1
        else:
            assert score in (1.0, 0.5, 0.0)
    assert found == 2


def test_list_suggestion_result_vector_destination(subject_index):
    suggestions = ListSuggestionResult(
        [
            # subject: seals (labels)
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p7141"),
                score=1.0,
            ),
            # subject: Vikings
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://www.yso.fi/onto/yso/p6479"),
                score=0.5,
            ),
        ]
    )
    destination = np.zeros(len(subject_index), dtype=np.float32)
    vector = suggestions.as_vector(len(subject_index), destination=destination)
    assert vector is destination


def test_list_suggestion_result_vector_notfound(subject_index):
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                subject_id=subject_index.by_uri("http://example.com/notfound"),
                score=1.0,
            )
        ]
    )
    assert suggestions.as_vector(len(subject_index)).sum() == 0


def test_vector_suggestion_result_as_vector(subject_index):
    orig_vector = np.ones(len(subject_index), dtype=np.float32)
    suggestions = VectorSuggestionResult(orig_vector)
    vector = suggestions.as_vector(len(subject_index))
    assert (vector == orig_vector).all()


def test_vector_suggestions_enforce_score_range(subject_index):
    orig_vector = np.array([-0.1, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    suggestions = VectorSuggestionResult(orig_vector)
    vector = suggestions.as_vector(len(subject_index))
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    assert (vector == expected).all()


def test_vector_suggestion_result_as_vector_destination(subject_index):
    orig_vector = np.ones(len(subject_index), dtype=np.float32)
    suggestions = VectorSuggestionResult(orig_vector)
    destination = np.zeros(len(subject_index), dtype=np.float32)
    assert not (destination == orig_vector).all()  # destination is all zeros

    vector = suggestions.as_vector(len(subject_index), destination=destination)
    assert vector is destination
    assert (destination == orig_vector).all()  # destination now all ones
