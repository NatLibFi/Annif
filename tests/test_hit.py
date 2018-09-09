"""Unit tests for hit processing in Annif"""

from annif.hit import AnalysisHit, AnalysisResult, HitFilter
from annif.corpus import SubjectIndex
import numpy as np


def generate_hits(n):
    hits = []
    for i in range(n):
        hits.append(AnalysisHit(uri='http://example.org/{}'.format(i),
                                label='hit {}'.format(i),
                                score=1.0 / (i + 1)))
    return AnalysisResult(hits)


def test_hitfilter_limit():
    orighits = generate_hits(10)
    hits = HitFilter(limit=5)(orighits)
    assert isinstance(hits, AnalysisResult)
    assert len(hits) == 5


def test_hitfilter_threshold():
    orighits = generate_hits(10)
    hits = HitFilter(threshold=0.5)(orighits)
    assert isinstance(hits, AnalysisResult)
    assert len(hits) == 2


def test_hitfilter_zero_score():
    orighits = AnalysisResult(
        [AnalysisHit(uri='uri', label='label', score=0.0)])
    hits = HitFilter()(orighits)
    assert isinstance(hits, AnalysisResult)
    assert len(hits) == 0


def test_analysishits_as_vector(subject_corpus):
    subjects = SubjectIndex(subject_corpus)
    hits = AnalysisResult([AnalysisHit(uri='http://www.yso.fi/onto/yso/p7141',
                                       label='sinetit', score=1.0),
                           AnalysisHit(uri='http://www.yso.fi/onto/yso/p6479',
                                       label='viikingit', score=0.5)
                           ])
    vector = hits.as_vector(subjects)
    assert isinstance(vector, np.ndarray)
    assert len(vector) == len(subjects)
    assert vector.sum() == 1.5
    for subject_id, score in enumerate(vector):
        if subjects[subject_id][1] == 'sinetit':
            assert score == 1.0
        elif subjects[subject_id][1] == 'viikingit':
            assert score == 0.5
        else:
            assert score == 0.0
