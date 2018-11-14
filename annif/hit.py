"""Representing hits from analysis."""

import abc
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
        return orighits.filter(self._limit, self._threshold)


class AnalysisResult(metaclass=abc.ABCMeta):
    """Abstract base class for a set of hits returned by an analysis
    operation."""

    @property
    @abc.abstractmethod
    def hits(self):
        """Return the hits as an ordered sequence of AnalysisHit objects,
        highest scores first."""
        pass

    @property
    @abc.abstractmethod
    def vector(self):
        """Return the hits as a one-dimensional score vector
        where the indexes match the given subject index."""
        pass

    @abc.abstractmethod
    def filter(self, limit=None, threshold=0.0):
        """Return a subset of the hits, filtered by the given limit and
        score threshold, as another AnalysisResult object."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """Return the number of hits with non-zero scores."""
        pass

    def __getitem__(self, idx):
        return self.hits[idx]


class VectorAnalysisResult(AnalysisResult):
    """AnalysisResult implementation based primarily on NumPy vectors."""

    def __init__(self, vector, subject_index):
        self._vector = vector
        self._subject_index = subject_index
        self._subject_order = None
        self._hits = None

    def _vector_to_hits(self):
        hits = []
        for subject_id in self.subject_order:
            score = self._vector[subject_id]
            if score <= 0.0:
                continue  # we can skip the remaining ones
            subject = self._subject_index[subject_id]
            hits.append(
                AnalysisHit(
                    uri=subject[0],
                    label=subject[1],
                    score=score))
        return ListAnalysisResult(hits, self._subject_index)

    @property
    def subject_order(self):
        if self._subject_order is None:
            self._subject_order = np.argsort(self._vector)[::-1]
        return self._subject_order

    @property
    def hits(self):
        if self._hits is None:
            self._hits = self._vector_to_hits()
        return self._hits

    @property
    def vector(self):
        return self._vector

    def filter(self, limit=None, threshold=0.0):
        mask = (self._vector > threshold)
        if limit is not None:
            limit_mask = np.zeros(len(self._vector), dtype=np.bool)
            top_k_subjects = self.subject_order[:limit]
            limit_mask[top_k_subjects] = True
            mask = mask & limit_mask
        return VectorAnalysisResult(self._vector * mask, self._subject_index)

    def __len__(self):
        return (self._vector > 0.0).sum()


class ListAnalysisResult(AnalysisResult):
    """AnalysisResult implementation based primarily on lists of hits."""

    def __init__(self, hits, subject_index):
        self._hits = [hit for hit in hits if hit.score > 0.0]
        self._subject_index = subject_index
        self._vector = None

    def _hits_to_vector(self):
        vector = np.zeros(len(self._subject_index))
        for hit in self._hits:
            subject_id = self._subject_index.by_uri(hit.uri)
            if subject_id is not None:
                vector[subject_id] = hit.score
        return vector

    @property
    def hits(self):
        return self._hits

    @property
    def vector(self):
        if self._vector is None:
            self._vector = self._hits_to_vector()
        return self._vector

    def filter(self, limit=None, threshold=0.0):
        hits = sorted(self.hits, key=lambda hit: hit.score, reverse=True)
        if limit is not None:
            hits = hits[:limit]
        return ListAnalysisResult([hit for hit in hits
                                   if hit.score >= threshold and
                                   hit.score > 0.0],
                                  self._subject_index)

    def __len__(self):
        return len(self._hits)
