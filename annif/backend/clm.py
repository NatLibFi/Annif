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
from rdflib import URIRef
from rdflib.namespace import RDFS, SKOS

from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.lexical.util import get_subject_labels
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import ensemble

if TYPE_CHECKING:
    from configparser import SectionProxy

    from rdflib.graph import Graph

    from annif.corpus.document import Document, DocumentCorpus
    from annif.project import AnnifProject


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
        "info": "",
        "max-info-len": 300,
    }

    def __init__(
        self,
        backend_id: str,
        config_params: dict[str, Any] | SectionProxy,
        project: AnnifProject,
    ) -> None:
        super().__init__(backend_id, config_params, project)
        self._graph = None
        self._info_cache: dict[tuple, str] = {}

    def _vocab_graph(self) -> Graph:
        """Load (and cache) the vocabulary as an rdflib graph"""
        if self._graph is None:
            self.info("loading vocabulary graph for subject info")
            self._graph = self.project.vocab.as_graph()
        return self._graph

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

    @staticmethod
    def _parse_info_types(info: str) -> list[str]:
        """Split the info parameter on top-level commas, keeping commas that
        are inside parentheses (e.g. 'prefLabel(en,sv)') together."""
        types = []
        depth = 0
        current = []
        for ch in info:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                types.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        types.append("".join(current).strip())
        return [t for t in types if t]

    def _subject_info(self, subject_id: int, params: dict[str, Any]) -> str:
        """Return additional information about the subject, collected from
        the SKOS vocabulary graph according to the 'info' parameter. The
        parameter is a comma-separated list of: definition, scopeNote, note,
        prefLabel, altLabel, hiddenLabel, broader, collection. prefLabel may
        carry a language list in parentheses (e.g. 'prefLabel(en)' or
        'prefLabel(en,sv)'); a bare 'prefLabel' selects every language other
        than the project language. An empty parameter or missing data
        results in an empty string. The result is truncated to
        'max-info-len' characters and cached per subject."""
        info_types = self._parse_info_types(params["info"])
        if not info_types:
            return ""

        max_len = int(params["max-info-len"])
        cache_key = (subject_id, tuple(info_types), max_len)
        if cache_key in self._info_cache:
            return self._info_cache[cache_key]

        graph = self._vocab_graph()
        try:
            uri = URIRef(self.project.subjects[subject_id].uri)
        except IndexError:  # deprecated subject
            return ""
        lang = self.project.language

        parts = []
        for info_type in info_types:
            if info_type in ("definition", "scopeNote", "note"):
                objects = list(graph.objects(uri, getattr(SKOS, info_type)))
                # prefer values in the project language
                objects.sort(key=lambda o: o.language != lang)
                values = [str(o) for o in objects]
                parts.extend(values)
            elif info_type in ("altLabel", "hiddenLabel"):
                values = get_subject_labels(
                    graph, str(uri), [getattr(SKOS, info_type)], lang
                )
                if values:
                    parts.append("also known as: {}".format(", ".join(values)))
            elif info_type == "broader":
                labels = []
                for obj in graph.objects(uri, SKOS.broader):
                    obj_id = self.project.subjects.by_uri(str(obj), warnings=False)
                    if obj_id is None:
                        continue
                    try:
                        subject = self.project.subjects[obj_id]
                    except IndexError:  # deprecated subject
                        continue
                    if subject.labels:
                        label = subject.labels.get(lang)
                        if label is None:
                            label = next(iter(subject.labels.values()))
                        labels.append(label)
                if labels:
                    parts.append("a subfield of: {}".format(", ".join(labels)))
            elif info_type == "collection":
                # the collections that this concept is a member of
                # (inverse skos:member); collections may be labelled with
                # skos:prefLabel or a plain rdfs:label
                labels = []
                for coll in graph.subjects(SKOS.member, uri):
                    by_lang = {}
                    for prop in (SKOS.prefLabel, RDFS.label):
                        for label in graph.objects(coll, prop):
                            by_lang.setdefault(label.language, []).append(str(label))
                    values = by_lang.get(lang)
                    if values is None:  # no label in the project language
                        for vals in by_lang.values():
                            values = vals
                            break
                    if values:
                        labels.append(values[0])
                if labels:
                    parts.append("member of: {}".format(", ".join(labels[:5])))
            elif info_type == "prefLabel" or (
                info_type.startswith("prefLabel(") and info_type.endswith(")")
            ):
                # prefLabel, prefLabel(en), prefLabel(en,sv): labels in the
                # given languages, or every language but the project language
                # when no list is given
                if info_type == "prefLabel":
                    languages = None  # all languages except the project language
                else:
                    inner = info_type[len("prefLabel(") : -1].strip()
                    languages = [
                        lang.strip() for lang in inner.split(",") if lang.strip()
                    ] or None
                by_lang = {}
                for label in graph.objects(uri, SKOS.prefLabel):
                    if languages is None and label.language == lang:
                        continue
                    if languages is not None and label.language not in languages:
                        continue
                    by_lang.setdefault(label.language, []).append(str(label))
                for other_lang, values in sorted(
                    by_lang.items(), key=lambda kv: kv[0] or ""
                ):
                    if other_lang is None:
                        parts.extend(values)
                    else:
                        parts.extend(
                            "{}: {}".format(other_lang, value) for value in values
                        )
            else:
                raise ConfigurationException(
                    "unknown info type '{}' (allowed: definition, scopeNote, "
                    "note, prefLabel[langs], altLabel, hiddenLabel, broader, "
                    "collection)".format(info_type)
                )

        info = " ".join(parts)[:max_len].strip()
        self._info_cache[cache_key] = info
        return info

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
        use_info = "{info}" in instruction
        questions = {}
        for suggestion in suggestions:
            label = self._label_for_subject(suggestion.subject_id)
            if label is None:
                self.debug(
                    f"no label found for subject {suggestion.subject_id}, "
                    "skipping the question"
                )
                continue
            info = self._subject_info(suggestion.subject_id, params) if use_info else ""
            # questions must have unique IDs; use the subject ID as the key
            questions[str(suggestion.subject_id)] = {
                "type": "noul",
                "instructions": instruction.format(label=label, info=info),
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
