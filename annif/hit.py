"""Class representing a single hit from analysis."""

import collections


AnalysisHit = collections.namedtuple('AnalysisHit', 'uri label score')


class HitFilter:
    """A class that represents a filtered collection of hits.
    The original hits are given in the constructor along with parameters
    used for filtering. The collection can then be iterated."""

    def __init__(self, hits, limit=1000, threshold=0.0):
        self._hits = hits
        self._limit = limit
        self._threshold = threshold

    def __iter__(self):
        hits = sorted(self._hits, key=lambda hit: hit.score, reverse=True)
        for hit in hits[:self._limit]:
            if hit.score >= self._threshold and hit.score > 0.0:
                yield hit
