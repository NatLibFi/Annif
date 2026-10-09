"""dm_rerank backend that uses a decision model reranking service to score
candidate subjects against a document."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import requests

from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.util import boolean

from . import ensemble

if TYPE_CHECKING:
    from configparser import SectionProxy

    from annif.corpus.document import Document, DocumentCorpus
    from annif.project import AnnifProject


# Built-in indexing-policy rules prepended to the document state when
# 'state-rules' is enabled: the format/genre rule.
STATE_RULES = (
    "You are a librarian performing topical subject indexing. "
    "A document can have several central subjects. "
    "Do NOT assign the document's own format or genre as a subject "
    "(a book of recipes is not about 'cookbooks', a pattern book is not "
    "about 'handicraft patterns', a travel book is not about "
    "'travelogues').\n\nDocument to be indexed:\n"
)


class DMRerankBackend(ensemble.BaseEnsembleBackend):
    """Ensemble-style backend that blends source candidates with noul
    scores from a decision model reranking service (noul-type true/false
    questions)."""

    name = "dm_rerank"

    DEFAULT_PARAMETERS = {
        "endpoint": "http://127.0.0.1:8700",
        "model": "",
        "model-strength": 0.5,
        "gate-threshold": -0.25,
        "retries": 2,
        "timeout": 60,
        "state-rules": False,
        "state-prefix": "",
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

    def _state_prefix(self, params: dict[str, Any]) -> str:
        """Return the text to prepend to the document state: the custom
        'state-prefix' when set, the built-in indexing-policy rules when
        'state-rules' is enabled, otherwise an empty string."""
        custom = str(params["state-prefix"])
        if custom:
            return custom
        if boolean(params["state-rules"]):
            return STATE_RULES
        return ""

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

        state_prefix = self._state_prefix(params)
        payload = {
            "state": state_prefix + doc.text,
            "questions": questions,
        }
        model = params["model"]
        if model:
            payload["model"] = model
        endpoint = params["endpoint"].rstrip("/") + "/v1/systemone"
        retries = int(params["retries"])
        timeout = float(params["timeout"])
        attempt = 0
        while True:
            try:
                req = requests.post(endpoint, json=payload, timeout=timeout)
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
        """Re-score the candidate subjects of one document using the noul
        scores from the reranking service and the model-strength /
        gate-threshold parameters."""
        noul_scores = self._query_dm_rerank(doc, suggestions, params)
        b = float(params["model-strength"])
        gate_threshold = float(params["gate-threshold"])
        return self._blend(suggestions, noul_scores, b, gate_threshold)

    @staticmethod
    def _blend(
        suggestions: list[SubjectSuggestion],
        noul_scores: dict[int, float],
        b: float,
        gate_threshold: float,
    ) -> list[SubjectSuggestion]:
        """Blend the source and noul scores, keeping all candidates.

        Each candidate with a noul score is re-scored as

            score = source_score * noul**b * sigmoid((noul - tau) / 0.1)

        where both source_score and noul are used as returned (the noul
        yes-probability is already on a [0, 1] scale) and tau is the gate
        threshold. b=0 with a negative tau (the default) reproduces the
        pure source order; b=0 with a positive tau approximates a hard
        gate that keeps only confident "yes" answers in source order. A
        candidate without a noul score (no label, question not asked)
        keeps its source score, so a missing or failed decision model
        degrades to the source ranking."""
        if not suggestions:
            return []

        blended = []
        for suggestion in suggestions:
            noul = noul_scores.get(suggestion.subject_id)
            if noul is None:
                score = suggestion.score
            else:
                gate = 1.0 / (1.0 + math.exp(-(noul - gate_threshold) / 0.1))
                score = suggestion.score * noul**b * gate
            blended.append(
                SubjectSuggestion(subject_id=suggestion.subject_id, score=score)
            )
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
