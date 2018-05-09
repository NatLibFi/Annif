"""Dummy backend for testing basic interaction of projects and backends"""


from annif.hit import AnalysisHit
from . import backend


class DummyBackend(backend.AnnifBackend):
    name = "dummy"
    initialized = False

    def initialize(self):
        self.initialized = True

    def _analyze(self, text, project, params):
        score = float(params.get('score', 1.0))
        return [AnalysisHit(uri='http://example.org/dummy',
                            label='dummy', score=score)]
