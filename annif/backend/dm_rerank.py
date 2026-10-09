"""dm_rerank backend that uses a decision model reranking service to score
candidate subjects against a document. For each candidate subject from the
source projects, the backend asks the service a noul-type (true/false)
question about the subject and the document. The default proposition is a
centrality ("sharp") question, "Is '{label}' a central subject of this
document ... not merely a passing or incidental mention?", which can be
replaced with any custom template via the 'instruction' parameter. The
candidates are then re-scored as a linear combination of the per-document
min-max normalized source score and noul score, weighted by the
blend-alpha parameter."""

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
    from configparser import SectionProxy

    from annif.corpus.document import Document, DocumentCorpus
    from annif.project import AnnifProject


class DMRerankBackend(ensemble.BaseEnsembleBackend):
    """Ensemble-style backend that blends source candidates with noul
    scores from a decision model reranking service (noul-type true/false
    questions)."""

    name = "dm_rerank"

    DEFAULT_PARAMETERS = {
        "endpoint": "http://127.0.0.1:8700",
        "model": "",
        "blend-alpha": 0.85,
        "retries": 2,
        # the "sharp" centrality predicate: makes the model judge whether
        # the subject is a central one (a primary heading), not merely
        # mentioned in passing; tested to outperform a plain "is about"
        # proposition on the prototype test sets
        "instruction": (
            "Is '{label}' a central subject of this document - one a "
            "librarian would assign as a primary heading - not merely a "
            "passing or incidental mention?"
        ),
    }

    def __init__(
        self,
        backend_id: str,
        config_params: dict[str, Any] | SectionProxy,
        project: AnnifProject,
    ) -> None:
        super().__init__(backend_id, config_params, project)

    @property
    def is_trained(self) -> bool:
        """The backend itself is untrained, but it requires all source
        projects to be trained."""
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    def _train(
        self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0
    ) -> None:
        raise NotSupportedException("Training the dm_rerank backend is not possible.")

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

    def _query_dm_rerank(
        self, doc: Document, suggestions: list[SubjectSuggestion], params
    ) -> dict[int, float]:
        """Ask the reranking service for a noul score for every candidate
        subject that has a label, and return a mapping subject_id -> noul
        score. Candidates without a label are not included in the mapping."""
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
            "questions": questions,
        }
        model = params["model"]
        if model:
            payload["model"] = model
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
                    msg = (
                        "dm_rerank request to {} failed after {} attempts: {}"
                    ).format(endpoint, retries + 1, err)
                    raise OperationFailedException(msg) from err
                delay = 2**attempt
                self.warning(
                    "dm_rerank request failed ({err}); retrying in {delay}s "
                    "({attempt}/{retries})".format(
                        err=err, delay=delay, attempt=attempt, retries=retries
                    )
                )
                time.sleep(delay)
        try:
            response = req.json()
        except ValueError as err:
            msg = f"dm_rerank response JSON decode failed: {err}"
            raise OperationFailedException(msg) from err

        noul_scores = {}
        for subject_id_str, answer in response.get("answers", {}).items():
            score = answer.get("noul")
            if score is not None:
                noul_scores[int(subject_id_str)] = score

        if noul_scores:
            values = list(noul_scores.values())
            self.info(
                "dm_rerank noul scores: min {:.3f}, mean {:.3f}, max {:.3f} "
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
        """Re-score the candidate subjects of one document as a blend of
        the source score and the noul score from the reranking service."""
        noul_scores = self._query_dm_rerank(doc, suggestions, params)
        alpha = float(params["blend-alpha"])
        return self._blend(suggestions, noul_scores, alpha)

    @staticmethod
    def _blend(
        suggestions: list[SubjectSuggestion],
        noul_scores: dict[int, float],
        alpha: float,
    ) -> list[SubjectSuggestion]:
        """Blend the source and noul scores, keeping all candidates.

        Both score sets are min-max normalized per document (a degenerate
        set maps to 0.5), then the new score is
        alpha * source_norm + (1 - alpha) * noul_norm. A candidate without
        a noul score (no label, question not asked) is treated as 0.0."""
        if not suggestions:
            return []

        src_min = min(s.score for s in suggestions)
        src_max = max(s.score for s in suggestions)
        # candidates without a noul score count as 0.0
        noul_values = [
            noul_scores.get(suggestion.subject_id, 0.0) for suggestion in suggestions
        ]
        noul_min = min(noul_values)
        noul_max = max(noul_values)

        def norm(value: float, lo: float, hi: float) -> float:
            if hi == lo:
                return 0.5
            return (value - lo) / (hi - lo)

        blended = [
            SubjectSuggestion(
                subject_id=suggestion.subject_id,
                score=alpha * norm(suggestion.score, src_min, src_max)
                + (1 - alpha) * norm(noul, noul_min, noul_max),
            )
            for suggestion, noul in zip(suggestions, noul_values)
        ]
        # the new order differs from the source order, so sort explicitly
        blended.sort(key=lambda s: s.score, reverse=True)
        return blended

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
