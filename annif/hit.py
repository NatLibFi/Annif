"""Representing hits from analysis."""

import collections


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
        return [hit for hit in hits
                if hit.score >= self._threshold and hit.score > 0.0]


class AnalysisHits:
    """A sequence of hits returned by an analysis operation."""

    def __init__(self, hits):
        self._hits = hits

    def __len__(self):
        return len(self._hits)

    def __getitem__(self, idx):
        return self._hits[idx]
