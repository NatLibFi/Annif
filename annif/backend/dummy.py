"""Dummy backend for testing basic interaction of projects and backends"""


from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from . import backend


class DummyBackend(backend.AnnifLearningBackend):
    name = "dummy"
    initialized = False
    uri = 'http://example.org/dummy'
    label = 'dummy'

    def initialize(self):
        self.initialized = True

    def _suggest(self, text, project, params):
        score = float(params.get('score', 1.0))
        return ListSuggestionResult([SubjectSuggestion(uri=self.uri,
                                                       label=self.label,
                                                       score=score)],
                                    project.subjects)

    def learn(self, corpus, project):
        # in this dummy backend we "learn" by picking up the URI and label
        # of the first subject of the first document in the learning set
        # and using that in subsequent analysis results
        for doc in corpus.documents:
            if doc.uris and doc.labels:
                self.uri = list(doc.uris)[0]
                self.label = list(doc.labels)[0]
            break
