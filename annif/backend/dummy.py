"""Dummy backend for testing basic interaction of projects and backends"""


from annif.suggestion import ListSuggestionResult, SubjectSuggestion

from . import backend


class DummyBackend(backend.AnnifLearningBackend):
    name = "dummy"
    initialized = False
    subject_id = 0
    is_trained = True
    modification_time = None

    def default_params(self):
        return backend.AnnifBackend.DEFAULT_PARAMETERS

    def initialize(self, parallel=False):
        self.initialized = True

    def _suggest(self, text, params):
        score = float(params.get("score", 1.0))

        # allow overriding returned subject via uri parameter
        if "uri" in params:
            subject_id = self.project.subjects.by_uri(params["uri"])
        else:
            subject_id = self.subject_id

        return ListSuggestionResult(
            [SubjectSuggestion(subject_id=subject_id, score=score)]
        )

    def _learn(self, corpus, params):
        # in this dummy backend we "learn" by picking up the subject ID
        # of the first subject of the first document in the learning set
        # and using that in subsequent analysis results
        for doc in corpus.documents:
            if doc.subject_set:
                self.subject_id = doc.subject_set[0]
            break
