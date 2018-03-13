"""Dummy backend for testing basic interaction of projects and backends"""


from annif.hit import AnalysisHit
from . import backend


class DummyBackend(backend.AnnifBackend):
    name = "dummy"

    def analyze(self, text):
        return [AnalysisHit('http://example.org/dummy', 'dummy', 1.0)]
