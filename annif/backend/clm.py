"""CLM backend that validates candidate subjects against a document using
a Contrastive Language Model (CLM) service. For each candidate subject from
the source projects, the backend asks the CLM service how likely the
proposition "This document is about {label}." is to be true, and keeps only
the candidates whose probability is at least the configured threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests

from annif.exception import NotSupportedException, OperationFailedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import ensemble

if TYPE_CHECKING:
    from annif.corpus.document import Document, DocumentCorpus


class CLMBackend(ensemble.BaseEnsembleBackend):
    """Ensemble-style backend that filters source candidates with a CLM
    service using noul-type true/false questions."""

    name = "clm"

    DEFAULT_PARAMETERS = {
        "endpoint": "http://127.0.0.1:8700",
        "model": "clm-latest",
        "threshold": 0.6,
    }

    @property
    def is_trained(self) -> bool:
        """The backend itself is untrained, but it requires all source
        projects to be trained."""
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    def _train(
        self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0
    ) -> None:
        raise NotSupportedException("Training the clm backend is not possible.")

    def _label_for_subject(self, subject_id: int) -> str | None:
        """Return the label of the given subject in the project language,
        falling back to any available label, then the notation, then None."""
        try:
            subject = self.project.subjects[subject_id]
        except IndexError:  # deprecated subject
            return None
        if subject.labels:
            if self.project.language in subject.labels:
                return subject.labels[self.project.language]
            return next(iter(subject.labels.values()))
        if subject.notation:
            return subject.notation
        return None

    def _validate_document(
        self, doc: Document, suggestions: list[SubjectSuggestion], params
    ) -> list[SubjectSuggestion]:
        """Ask the CLM service which of the candidate subjects are valid for
        the given document, and return the subset that pass the threshold."""
        questions = {}
        for suggestion in suggestions:
            label = self._label_for_subject(suggestion.subject_id)
            if label is None:
                self.debug(
                    f"no label found for subject {suggestion.subject_id}, "
                    "excluding it"
                )
                continue
            # questions must have unique IDs; use the subject ID as the key
            questions[str(suggestion.subject_id)] = {
                "type": "noul",
                "instructions": f"This document is about {label}.",
            }
        if not questions:
            return []

        payload = {
            "state": doc.text,
            "model": params["model"],
            "questions": questions,
        }
        endpoint = params["endpoint"].rstrip("/") + "/v1/systemone"
        try:
            req = requests.post(endpoint, json=payload)
            req.raise_for_status()
        except requests.exceptions.RequestException as err:
            msg = f"CLM request to {endpoint} failed: {err}"
            raise OperationFailedException(msg) from err
        try:
            response = req.json()
        except ValueError as err:
            msg = f"CLM response JSON decode failed: {err}"
            raise OperationFailedException(msg) from err

        threshold = float(params["threshold"])
        valid_ids = set()
        answers = response.get("answers", {})
        scores = []
        for subject_id_str, answer in answers.items():
            score = answer.get("noul")
            if score is None:
                continue
            scores.append(score)
            if score >= threshold:
                valid_ids.add(int(subject_id_str))
        if scores:
            self.info(
                "CLM noul scores: min {:.3f}, mean {:.3f}, max {:.3f} "
                "({} of {} candidates >= threshold {:.2f})".format(
                    min(scores),
                    sum(scores) / len(scores),
                    max(scores),
                    len(valid_ids),
                    len(scores),
                    threshold,
                )
            )

        return [
            suggestion
            for suggestion in suggestions
            if suggestion.subject_id in valid_ids
        ]

    def _suggest_batch(
        self, documents: list[Document], params: dict[str, Any]
    ) -> SuggestionBatch:
        # merge the source suggestions with the regular weighted average
        merged = super()._suggest_batch(documents, params)
        limit = int(params["limit"])

        validated = []
        for idx, doc in enumerate(documents):
            suggestions = list(merged[idx])
            kept = self._validate_document(doc, suggestions, params)
            self.debug(
                f"CLM validated {len(kept)} of {len(suggestions)} " "candidate subjects"
            )
            validated.append(kept)

        return SuggestionBatch.from_sequence(
            validated, self.project.subjects, limit=limit
        )
