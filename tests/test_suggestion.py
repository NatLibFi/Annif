"""Unit tests for suggestion processing in Annif"""

import pytest
from scipy.sparse import csr_array

from annif.corpus import Subject
from annif.suggestion import SubjectSuggestion, SuggestionBatch, filter_suggestion


def generate_suggestions(n, subject_index):
    return [SubjectSuggestion(subject_id=i, score=1.0 / (i + 1)) for i in range(n)]


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


def test_suggestionbatch_from_sequence(dummy_subject_index):
    orig_suggestions = [
        [
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/dummy"),
                score=0.8,
            ),
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/none"),
                score=0.2,
            ),
        ]
    ]

    sbatch = SuggestionBatch.from_sequence(orig_suggestions, dummy_subject_index)
    assert len(sbatch) == 1
    suggestions = list(sbatch[0])
    assert len(suggestions) == 2
    assert suggestions[0].subject_id == dummy_subject_index.by_uri(
        "http://example.org/dummy"
    )
    assert suggestions[0].score == pytest.approx(0.8)
    assert suggestions[1].subject_id == dummy_subject_index.by_uri(
        "http://example.org/none"
    )
    assert suggestions[1].score == pytest.approx(0.2)


def test_suggestionbatch_from_sequence_enforce_score_range(dummy_subject_index):
    orig_suggestions = [
        [
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/dummy"),
                score=1.2,
            ),
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/none"),
                score=-0.2,
            ),
        ]
    ]

    sbatch = SuggestionBatch.from_sequence(orig_suggestions, dummy_subject_index)
    assert len(sbatch) == 1
    suggestions = list(sbatch[0])
    assert len(suggestions) == 1
    assert suggestions[0].subject_id == dummy_subject_index.by_uri(
        "http://example.org/dummy"
    )
    assert suggestions[0].score == pytest.approx(1.0)


def test_suggestionbatch_from_sequence_with_deprecated(dummy_subject_index):
    dummy_subject_index.append(
        Subject(uri="http://example.org/deprecated", labels=None, notation=None)
    )

    orig_suggestions = [
        [
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/dummy"),
                score=0.8,
            ),
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/deprecated"),
                score=0.5,
            ),
            SubjectSuggestion(
                subject_id=dummy_subject_index.by_uri("http://example.org/none"),
                score=0.2,
            ),
        ]
    ]

    sbatch = SuggestionBatch.from_sequence(orig_suggestions, dummy_subject_index)
    assert len(sbatch) == 1
    suggestions = list(sbatch[0])
    assert len(suggestions) == 2
    assert suggestions[0].subject_id == dummy_subject_index.by_uri(
        "http://example.org/dummy"
    )
    assert suggestions[0].score == pytest.approx(0.8)
    assert suggestions[1].subject_id == dummy_subject_index.by_uri(
        "http://example.org/none"
    )
    assert suggestions[1].score == pytest.approx(0.2)
