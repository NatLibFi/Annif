"""Unit tests for suggestion processing in Annif"""

from annif.suggestion import SubjectSuggestion, SuggestionResult, \
    LazySuggestionResult, ListSuggestionResult, VectorSuggestionResult, \
    SuggestionFilter
import numpy as np


def generate_suggestions(n, subject_index):
    suggestions = []
    for i in range(n):
        uri = 'http://example.org/{}'.format(i)
        suggestions.append(SubjectSuggestion(uri=uri,
                                             label='hit {}'.format(i),
                                             notation=None,
                                             score=1.0 / (i + 1)))
    return ListSuggestionResult(suggestions)


def test_hitfilter_limit(subject_index):
    origsuggestions = generate_suggestions(10, subject_index)
    suggestions = SuggestionFilter(subject_index, limit=5)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 5


def test_hitfilter_threshold(subject_index):
    origsuggestions = generate_suggestions(10, subject_index)
    suggestions = SuggestionFilter(subject_index,
                                   threshold=0.5)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 2


def test_hitfilter_zero_score(subject_index):
    origsuggestions = ListSuggestionResult(
        [SubjectSuggestion(uri='uri', label='label', notation=None,
                           score=0.0)])
    suggestions = SuggestionFilter(subject_index)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 0


def test_hitfilter_list_suggestion_results_with_deprecated_subjects(
        subject_index):
    subject_index.append('http://example.org/deprecated', None, None)
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p7141',
                label='sinetit',
                notation=None,
                score=1.0),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p6479',
                label='viikingit',
                notation=None,
                score=0.5),
            SubjectSuggestion(
                uri='http://example.org/deprecated',
                label=None,
                notation=None,
                score=0.5)])
    filtered_suggestions = SuggestionFilter(subject_index)(suggestions)
    assert isinstance(filtered_suggestions, SuggestionResult)
    assert len(filtered_suggestions) == 2
    assert filtered_suggestions.as_list(
        subject_index)[0] == suggestions.as_list(subject_index)[0]
    assert filtered_suggestions.as_list(
        subject_index)[1] == suggestions.as_list(subject_index)[1]


def test_hitfilter_vector_suggestion_results_with_deprecated_subjects(
        subject_index):
    subject_index.append('http://example.org/deprecated', None, None)
    vector = np.ones(len(subject_index))
    suggestions = VectorSuggestionResult(vector)
    filtered_suggestions = SuggestionFilter(subject_index)(suggestions)

    assert len(suggestions) == len(filtered_suggestions) \
        + len(subject_index.deprecated_ids())

    deprecated = SubjectSuggestion(
        uri='http://example.org/deprecated',
        label=None,
        notation=None,
        score=1.0)
    assert deprecated in suggestions.as_list(subject_index)
    assert deprecated not in filtered_suggestions.as_list(subject_index)


def test_lazy_suggestion_result(subject_index):
    lsr = LazySuggestionResult(lambda: generate_suggestions(10, subject_index))
    assert lsr._object is None
    assert len(lsr) == 10
    assert len(lsr.as_list(subject_index)) == 10
    assert lsr.as_vector(subject_index) is not None
    assert lsr.as_list(subject_index)[0] is not None
    filtered = lsr.filter(subject_index, limit=5, threshold=0.0)
    assert len(filtered) == 5
    assert lsr._object is not None


def test_list_suggestion_result_vector(subject_index):
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p7141',
                label='sinetit',
                notation=None,
                score=1.0),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p6479',
                label='viikingit',
                notation=None,
                score=0.5)])
    vector = suggestions.as_vector(subject_index)
    assert isinstance(vector, np.ndarray)
    assert len(vector) == len(subject_index)
    assert vector.sum() == 1.5
    for subject_id, score in enumerate(vector):
        if subject_index[subject_id][1] == 'sinetit':
            assert score == 1.0
        elif subject_index[subject_id][1] == 'viikingit':
            assert score == 0.5
        else:
            assert score == 0.0


def test_list_suggestions_vector_enforce_score_range(subject_index):
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p7141',
                label='sinetit',
                notation=None,
                score=1.5),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p6479',
                label='viikingit',
                notation=None,
                score=1.0),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p14173',
                label='kaivaukset',
                notation=None,
                score=0.5),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p14588',
                label='riimukivet',
                notation=None,
                score=0.0),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p12738',
                label='viikinkiaika',
                notation=None,
                score=-0.5)])
    vector = suggestions.as_vector(subject_index)
    assert vector.sum() == 2.5
    for subject_id, score in enumerate(vector):
        if subject_index[subject_id][1] == 'sinetit':
            assert score == 1.0
        elif subject_index[subject_id][1] == 'viikinkiaika':
            assert score == 0.0
        else:
            assert score in (1.0, 0.5, 0.0)


def test_list_suggestion_result_vector_destination(subject_index):
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p7141',
                label='sinetit',
                notation=None,
                score=1.0),
            SubjectSuggestion(
                uri='http://www.yso.fi/onto/yso/p6479',
                label='viikingit',
                notation=None,
                score=0.5)])
    destination = np.zeros(len(subject_index), dtype=np.float32)
    vector = suggestions.as_vector(subject_index, destination=destination)
    assert vector is destination


def test_list_suggestion_result_vector_notfound(subject_index):
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                uri='http://example.com/notfound',
                label='not found',
                notation=None,
                score=1.0)])
    assert suggestions.as_vector(subject_index).sum() == 0


def test_vector_suggestion_result_as_vector(subject_index):
    orig_vector = np.ones(len(subject_index), dtype=np.float32)
    suggestions = VectorSuggestionResult(orig_vector)
    vector = suggestions.as_vector(subject_index)
    assert (vector == orig_vector).all()


def test_vector_suggestions_enforce_score_range(subject_index):
    orig_vector = np.array([-0.1, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    suggestions = VectorSuggestionResult(orig_vector)
    vector = suggestions.as_vector(subject_index)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    assert (vector == expected).all()


def test_vector_suggestion_result_as_vector_destination(subject_index):
    orig_vector = np.ones(len(subject_index), dtype=np.float32)
    suggestions = VectorSuggestionResult(orig_vector)
    destination = np.zeros(len(subject_index), dtype=np.float32)
    assert not (destination == orig_vector).all()  # destination is all zeros

    vector = suggestions.as_vector(subject_index, destination=destination)
    assert vector is destination
    assert (destination == orig_vector).all()      # destination now all ones
