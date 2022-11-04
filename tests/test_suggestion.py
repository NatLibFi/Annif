"""Unit tests for suggestion processing in Annif"""

from annif.suggestion import (
    SubjectSuggestion,
    SuggestionResult,
    LazySuggestionResult,
    ListSuggestionResult,
    VectorSuggestionResult,
    SuggestionFilter,
)
from annif.corpus import Subject
import numpy as np


def generate_suggestions(n, subject_index):
    suggestions = []
    for i in range(n):
        suggestions.append(SubjectSuggestion(subject_id=i, score=1.0 / (i + 1)))
    return ListSuggestionResult(suggestions)


def test_hitfilter_limit(subject_index):
    origsuggestions = generate_suggestions(10, subject_index)
    suggestions = SuggestionFilter(subject_index, limit=5)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 5


def test_hitfilter_threshold(subject_index):
    origsuggestions = generate_suggestions(10, subject_index)
    suggestions = SuggestionFilter(subject_index, threshold=0.5)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 2


def test_hitfilter_zero_score(subject_index):
    origsuggestions = ListSuggestionResult([SubjectSuggestion(subject_id=0, score=0.0)])
    suggestions = SuggestionFilter(subject_index)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 0


def test_hitfilter_list_suggestion_results_with_deprecated_subjects(subject_index):
    subject_index.append(
        Subject(uri="http://example.org/deprecated", labels=None, notation=None)
    )
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
            # a deprecated subject
            SubjectSuggestion(subject_id=None, score=0.5),
        ]
    )
    filtered_suggestions = SuggestionFilter(subject_index)(suggestions)
    assert isinstance(filtered_suggestions, SuggestionResult)
    assert len(filtered_suggestions) == 2
    assert filtered_suggestions.as_list()[0] == suggestions.as_list()[0]
    assert filtered_suggestions.as_list()[1] == suggestions.as_list()[1]


def test_hitfilter_vector_suggestion_results_with_deprecated_subjects(subject_index):
    subject_index.append(
        Subject(uri="http://example.org/deprecated", labels=None, notation=None)
    )
    vector = np.ones(len(subject_index))
    suggestions = VectorSuggestionResult(vector)
    filtered_suggestions = SuggestionFilter(subject_index)(suggestions)

    assert len(suggestions) == len(filtered_suggestions) + len(
        subject_index.deprecated_ids()
    )

    deprecated_id = subject_index.by_uri("http://example.org/deprecated")
    deprecated = SubjectSuggestion(subject_id=deprecated_id, score=1.0)

    assert deprecated in suggestions.as_list()
    assert deprecated not in filtered_suggestions.as_list()


def test_lazy_suggestion_result(subject_index):
    lsr = LazySuggestionResult(lambda: generate_suggestions(10, subject_index))
    assert lsr._object is None
    assert len(lsr) == 10
    assert len(lsr.as_list()) == 10
    assert lsr.as_vector(len(subject_index)) is not None
    assert lsr.as_list()[0] is not None
    filtered = lsr.filter(subject_index, limit=5, threshold=0.0)
    assert len(filtered) == 5
    assert lsr._object is not None


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
