"""Class representing a single hit from analysis."""

import collections


AnalysisHit = collections.namedtuple('AnalysisHit', 'uri label score')


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
