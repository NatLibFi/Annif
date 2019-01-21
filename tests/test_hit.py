"""Unit tests for hit processing in Annif"""

from annif.hit import AnalysisHit, AnalysisResult, ListAnalysisResult, \
    HitFilter
from annif.corpus import SubjectIndex
import numpy as np


def generate_hits(n, subject_index):
    hits = []
    for i in range(n):
        hits.append(AnalysisHit(uri='http://example.org/{}'.format(i),
                                label='hit {}'.format(i),
                                score=1.0 / (i + 1)))
    return ListAnalysisResult(hits, subject_index)


def test_hitfilter_limit(subject_index):
    orighits = generate_hits(10, subject_index)
    hits = HitFilter(limit=5)(orighits)
    assert isinstance(hits, AnalysisResult)
    assert len(hits) == 5


def test_hitfilter_threshold(subject_index):
    orighits = generate_hits(10, subject_index)
    hits = HitFilter(threshold=0.5)(orighits)
    assert isinstance(hits, AnalysisResult)
    assert len(hits) == 2


def test_hitfilter_zero_score(subject_index):
    orighits = ListAnalysisResult(
        [AnalysisHit(uri='uri', label='label', score=0.0)],
        subject_index)
    hits = HitFilter()(orighits)
    assert isinstance(hits, AnalysisResult)
    assert len(hits) == 0


def test_analysishits_vector(document_corpus):
    subjects = SubjectIndex(document_corpus)
    hits = ListAnalysisResult(
        [
            AnalysisHit(
                uri='http://www.yso.fi/onto/yso/p7141',
                label='sinetit',
                score=1.0),
            AnalysisHit(
                uri='http://www.yso.fi/onto/yso/p6479',
                label='viikingit',
                score=0.5)],
        subjects)
    assert isinstance(hits.vector, np.ndarray)
    assert len(hits.vector) == len(subjects)
    assert hits.vector.sum() == 1.5
    for subject_id, score in enumerate(hits.vector):
        if subjects[subject_id][1] == 'sinetit':
            assert score == 1.0
        elif subjects[subject_id][1] == 'viikingit':
            assert score == 0.5
        else:
            assert score == 0.0


def test_analysishits_vector_notfound(document_corpus):
    subjects = SubjectIndex(document_corpus)
    hits = ListAnalysisResult(
        [
            AnalysisHit(
                uri='http://example.com/notfound',
                label='not found',
                score=1.0)],
        subjects)
    assert hits.vector.sum() == 0
