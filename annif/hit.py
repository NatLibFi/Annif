"""Representing hits from analysis."""

import collections
import numpy as np


AnalysisHit = collections.namedtuple('AnalysisHit', 'uri label score')
WeightedHits = collections.namedtuple('WeightedHits', 'hits weight')


class HitFilter:
    """A reusable filter for filtering AnalysisHit objects."""

    def __init__(self, limit=None, threshold=0.0):
        self._limit = limit
        self._threshold = threshold

    def __call__(self, orighits):
        hits = sorted(orighits, key=lambda hit: hit.score, reverse=True)
        if self._limit is not None:
            hits = hits[:self._limit]
        return AnalysisResult([hit for hit in hits
                               if hit.score >= self._threshold and
                               hit.score > 0.0])


class AnalysisResult:
    """A sequence of hits returned by an analysis operation."""

    def __init__(self, hits=None, vector=None, limit=None, subject_index=None):
        if hits is not None:
            self._hits = [hit for hit in hits if hit.score > 0.0]
        else:
            self._hits = None
        self._vector = vector
        self._limit = limit
        self._subject_index = subject_index

    @classmethod
    def from_vector(cls, vector, limit, subject_index):
        """Create an AnalysisResult from a one-dimensional score vector
        where the indexes match the given subject index. Keep only the
        number of results specified by the limit parameter."""

        return AnalysisResult(
            vector=vector,
            limit=limit,
            subject_index=subject_index)

    def _vector_to_hits(self):
        top_scores = sorted(enumerate(self._vector),
                            key=lambda id_score: id_score[1],
                            reverse=True)
        hits = []
        for subject_id, score in top_scores[:self._limit]:
            if score <= 0.0:
                continue
            subject = self._subject_index[subject_id]
            hits.append(
                AnalysisHit(
                    uri=subject[0],
                    label=subject[1],
                    score=score))
        return AnalysisResult(hits)

    @property
    def hits(self):
        if self._hits is None:
            self._hits = self._vector_to_hits()
        return self._hits

    def __len__(self):
        return len(self.hits)

    def __getitem__(self, idx):
        return self.hits[idx]

    def as_vector(self, subject_index):
        """Return the hits as a one-dimensional NumPy array of scores, using a
           subject index as the source of subjects."""

        if self._vector is None:
            self._vector = np.zeros(len(subject_index))
            for hit in self._hits:
                subject_id = subject_index.by_uri(hit.uri)
                self._vector[subject_id] = hit.score
        return self._vector
