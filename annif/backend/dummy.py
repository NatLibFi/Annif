"""Dummy backend for testing basic interaction of projects and backends"""


from . import backend


class DummyBackend(backend.AnnifBackend):
    name = "dummy"
    
    def analyze(self, text):
        return []
