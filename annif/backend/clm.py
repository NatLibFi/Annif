"""CLM backend that uses a Contrastive Language Model (CLM) service to
score candidate subjects against a document. For each candidate subject
from the source projects, the backend asks the CLM service how likely the
proposition "This document is about {label}." is to be true (a noul-type
question). Depending on the mode, the candidates are then either filtered
(drop candidates whose score is below the threshold) or reranked (keep all
candidates, re-score them by multiplying the source score with the noul
score)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import requests

from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import ensemble

if TYPE_CHECKING:
    from annif.corpus.document import Document, DocumentCorpus


class CLMBackend(ensemble.BaseEnsembleBackend):
    """Ensemble-style backend that filters or reranks source candidates
    with a CLM service using noul-type true/false questions."""

    name = "clm"

    DEFAULT_PARAMETERS = {
        "endpoint": "http://127.0.0.1:8700",
        "model": "clm-latest",
        "threshold": 0.6,
        "mode": "filter",
        "retries": 2,
        "instruction": "This document is about {label}.",
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

    def _query_clm(
        self, doc: Document, suggestions: list[SubjectSuggestion], params
    ) -> dict[int, float]:
        """Ask the CLM service for a noul score for every candidate subject
        that has a label, and return a mapping subject_id -> noul score.
        Candidates without a label are not included in the mapping."""
        instruction = params["instruction"]
        if "{label}" not in instruction:
            raise ConfigurationException(
                "instruction parameter must contain a {label} placeholder"
            )
        questions = {}
        for suggestion in suggestions:
            label = self._label_for_subject(suggestion.subject_id)
            if label is None:
                self.debug(
                    f"no label found for subject {suggestion.subject_id}, "
                    "skipping the question"
                )
                continue
            # questions must have unique IDs; use the subject ID as the key
            questions[str(suggestion.subject_id)] = {
                "type": "noul",
                "instructions": instruction.format(label=label),
            }
        if not questions:
            return {}

        payload = {
            "state": doc.text,
            "model": params["model"],
            "questions": questions,
        }
        endpoint = params["endpoint"].rstrip("/") + "/v1/systemone"
        retries = int(params["retries"])
        attempt = 0
        while True:
            try:
                req = requests.post(endpoint, json=payload)
                req.raise_for_status()
                break
            except requests.exceptions.RequestException as err:
                attempt += 1
                if attempt > retries:
                    msg = "CLM request to {} failed after {} attempts: {}".format(
                        endpoint, retries + 1, err
                    )
                    raise OperationFailedException(msg) from err
                delay = 2**attempt
                self.warning(
                    "CLM request failed ({err}); retrying in {delay}s "
                    "({attempt}/{retries})".format(
                        err=err, delay=delay, attempt=attempt, retries=retries
                    )
                )
                time.sleep(delay)
        try:
            response = req.json()
        except ValueError as err:
            msg = f"CLM response JSON decode failed: {err}"
            raise OperationFailedException(msg) from err

        noul_scores = {}
        for subject_id_str, answer in response.get("answers", {}).items():
            score = answer.get("noul")
            if score is not None:
                noul_scores[int(subject_id_str)] = score

        if noul_scores:
            values = list(noul_scores.values())
            self.info(
                "CLM noul scores: min {:.3f}, mean {:.3f}, max {:.3f} "
                "for {} candidates".format(
                    min(values),
                    sum(values) / len(values),
                    max(values),
                    len(values),
                )
            )

        return noul_scores

    def _process_document(
        self, doc: Document, suggestions: list[SubjectSuggestion], params
    ) -> list[SubjectSuggestion]:
        """Process the candidate subjects of one document according to the
        configured mode and return the resulting suggestions."""
        noul_scores = self._query_clm(doc, suggestions, params)

        if params["mode"] == "filter":
            threshold = float(params["threshold"])
            kept = [
                suggestion
                for suggestion in suggestions
                if noul_scores.get(suggestion.subject_id, -1.0) >= threshold
            ]
            self.debug(
                f"CLM filtered {len(kept)} of {len(suggestions)} "
                f"candidate subjects (threshold {threshold})"
            )
            return kept

        # rerank mode: keep all candidates, re-score by source score * noul
        reranked = []
        for suggestion in suggestions:
            noul = noul_scores.get(suggestion.subject_id)
            if noul is None:  # no label, question not asked
                reranked.append(suggestion)
                continue
            reranked.append(
                SubjectSuggestion(
                    subject_id=suggestion.subject_id,
                    score=suggestion.score * noul,
                )
            )
        # the new order differs from the source order, so sort explicitly
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked

    def _suggest_batch(
        self, documents: list[Document], params: dict[str, Any]
    ) -> SuggestionBatch:
        # merge the source suggestions with the regular weighted average
        merged = super()._suggest_batch(documents, params)
        limit = int(params["limit"])

        processed = [
            self._process_document(doc, list(merged[idx]), params)
            for idx, doc in enumerate(documents)
        ]

        return SuggestionBatch.from_sequence(
            processed, self.project.subjects, limit=limit
        )
