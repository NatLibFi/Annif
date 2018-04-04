"""Class representing a single hit from analysis."""

import collections


AnalysisHit = collections.namedtuple('AnalysisHit', 'uri label score')
