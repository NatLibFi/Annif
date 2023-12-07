"""Dummy backend for testing basic interaction of projects and backends"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from annif.suggestion import SubjectSuggestion

from . import backend

if TYPE_CHECKING:
    from annif.corpus.document import DocumentCorpus


class DummyBackend(backend.AnnifLearningBackend):
    name = "dummy"
    initialized = False
    subject_id = 0
    is_trained = True
    modification_time = None

    def initialize(self, parallel: bool = False) -> None:
        self.initialized = True

    def _suggest(self, text: str, params: dict[str, Any]) -> list[SubjectSuggestion]:
        score = float(params.get("score", 1.0))

        # Ensure tests fail if "text" with wrong type ends up here
        assert isinstance(text, str)

        # Give no hits for no text
        if len(text) == 0:
            return []

        # allow overriding returned subject via uri parameter
        if "uri" in params:
            subject_id = self.project.subjects.by_uri(params["uri"])
        else:
            subject_id = self.subject_id

        return [SubjectSuggestion(subject_id=subject_id, score=score)]

    def _learn(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
    ) -> None:
        # in this dummy backend we "learn" by picking up the subject ID
        # of the first subject of the first document in the learning set
        # and using that in subsequent analysis results
        for doc in corpus.documents:
            if doc.subject_set:
                self.subject_id = doc.subject_set[0]
            break
