"""Class representing a single hit from analysis."""

import collections


AnalysisHit = collections.namedtuple('AnalysisHit', 'uri label score')


class HitFilter:
    """A class that represents a filtered collection of hits.
    The original hits are given in the constructor along with parameters
    used for filtering. The collection can then be iterated."""

    def __init__(self, hits, limit=None, threshold=0.0):
        self._hits = hits
        self._limit = limit
        self._threshold = threshold

    def __iter__(self):
        hits = sorted(self._hits, key=lambda hit: hit.score, reverse=True)
        if self._limit is not None:
            hits = hits[:self._limit]
        for hit in hits:
            if hit.score >= self._threshold and hit.score > 0.0:
                yield hit
