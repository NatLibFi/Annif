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
    return ListSuggestionResult(suggestions, subject_index)


def test_hitfilter_limit(subject_index):
    origsuggestions = generate_suggestions(10, subject_index)
    suggestions = SuggestionFilter(limit=5)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 5


def test_hitfilter_threshold(subject_index):
    origsuggestions = generate_suggestions(10, subject_index)
    suggestions = SuggestionFilter(threshold=0.5)(origsuggestions)
    assert isinstance(suggestions, SuggestionResult)
    assert len(suggestions) == 2


def test_hitfilter_zero_score(subject_index):
    origsuggestions = ListSuggestionResult(
        [SubjectSuggestion(uri='uri', label='label', notation=None,
                           score=0.0)],
        subject_index)
    suggestions = SuggestionFilter()(origsuggestions)
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
                score=0.5)],
        subject_index)
    filtered_suggestions = SuggestionFilter()(suggestions)
    assert isinstance(filtered_suggestions, SuggestionResult)
    assert len(filtered_suggestions) == 2
    assert filtered_suggestions[0] == suggestions[0]
    assert filtered_suggestions[1] == suggestions[1]


def test_hitfilter_vector_suggestion_results_with_deprecated_subjects(
        subject_index):
    subject_index.append('http://example.org/deprecated', None, None)
    vector = np.ones(len(subject_index))
    suggestions = VectorSuggestionResult(vector, subject_index)
    filtered_suggestions = SuggestionFilter()(suggestions)

    assert len(suggestions) == len(filtered_suggestions) \
        + len(subject_index.deprecated_ids())

    deprecated = SubjectSuggestion(
        uri='http://example.org/deprecated',
        label=None,
        notation=None,
        score=1.0)
    assert deprecated in list(suggestions.hits)
    assert deprecated not in list(filtered_suggestions.hits)


def test_lazy_suggestion_result(subject_index):
    lar = LazySuggestionResult(lambda: generate_suggestions(10, subject_index))
    assert lar._object is None
    assert len(lar) == 10
    assert len(lar.hits) == 10
    assert lar.vector is not None
    assert lar[0] is not None
    filtered = lar.filter(limit=5, threshold=0.0)
    assert len(filtered) == 5
    assert lar._object is not None


def test_list_suggestions_vector(document_corpus, subject_index):
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
                score=0.5)],
        subject_index)
    assert isinstance(suggestions.vector, np.ndarray)
    assert len(suggestions.vector) == len(subject_index)
    assert suggestions.vector.sum() == 1.5
    for subject_id, score in enumerate(suggestions.vector):
        if subject_index[subject_id][1] == 'sinetit':
            assert score == 1.0
        elif subject_index[subject_id][1] == 'viikingit':
            assert score == 0.5
        else:
            assert score == 0.0


def test_list_suggestions_vector_notfound(document_corpus, subject_index):
    suggestions = ListSuggestionResult(
        [
            SubjectSuggestion(
                uri='http://example.com/notfound',
                label='not found',
                notation=None,
                score=1.0)],
        subject_index)
    assert suggestions.vector.sum() == 0
