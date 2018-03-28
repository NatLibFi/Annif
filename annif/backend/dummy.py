"""Dummy backend for testing basic interaction of projects and backends"""


from annif.hit import AnalysisHit
from . import backend


class DummyBackend(backend.AnnifBackend):
    name = "dummy"

    def analyze(self, text, params={}):
        score = float(params.get('score', 1.0))
        return [AnalysisHit('http://example.org/dummy', 'dummy', score)]
