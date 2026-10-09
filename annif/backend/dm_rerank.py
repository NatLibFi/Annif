"""dm_rerank backend that uses a decision model reranking service to score
candidate subjects against a document."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import requests

import annif.parallel
import annif.transform
from annif.exception import (
    ConfigurationException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.util import boolean, parse_sources

from . import ensemble, hyperopt

if TYPE_CHECKING:
    from configparser import SectionProxy

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.backend.hyperopt import HPRecommendation
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


class DMRerankBackend(ensemble.BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble-style backend that blends source candidates with noul
    scores from a decision model reranking service (noul-type true/false
    questions)."""

    name = "dm_rerank"

    DEFAULT_PARAMETERS = {
        "endpoint": "http://localhost:8080/v1/systemone",
        "model": "",
        "model-strength": 0.5,
        "gate-threshold": -1.0,
        "retries": 2,
        "timeout": 60,
        "state-rules": False,
        "state-prefix": "",
        "state-transform": "pass",
        "max-candidates": 0,
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
        self._state_transform = None

    @property
    def is_trained(self) -> bool:
        """The backend itself is untrained, but it requires all source
        projects to be trained."""
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    @property
    def state_transform(self):
        """The transform applied to the document text before it is sent
        to the reranking service as the question state, on top of the
        project transform. Defaults to pass (no extra transformation).
        Useful for keeping the state within the model's context length
        while the source projects run on the full text."""
        if self._state_transform is None:
            spec = str(self.params["state-transform"])
            self._state_transform = annif.transform.get_transform(spec, project=None)
        return self._state_transform

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
        state = self.state_transform.transform_doc(doc).text
        payload = {
            "state": state_prefix + state,
            "questions": questions,
        }
        model = params["model"]
        if model:
            payload["model"] = model
        endpoint = params["endpoint"].rstrip("/")
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
            self.debug(
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
        threshold. b=0 with a sufficiently negative tau (e.g. -1.0, where
        the sigmoid is >= 0.99995 for all noul) reproduces the pure
        source order, so the pair (0, -1.0) is the exact no-op point; a
        positive tau approximates a hard gate that keeps only confident
        "yes" answers in source order. A candidate without a noul score
        (no label, question not asked) keeps its source score, so a
        missing or failed decision model degrades to the source
        ranking."""
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
        # optionally restrict how many of the top candidates the decision
        # model scores; candidates beyond max-candidates are dropped
        max_candidates = int(params["max-candidates"])

        def candidates(idx: int) -> list[SubjectSuggestion]:
            suggestions = list(merged[idx])
            if max_candidates > 0:
                return suggestions[:max_candidates]
            return suggestions

        processed = [
            self._process_document(doc, candidates(idx), params)
            for idx, doc in enumerate(documents)
        ]

        return SuggestionBatch.from_sequence(
            processed, self.project.subjects, limit=limit
        )

    def get_hp_optimizer(
        self, corpus: DocumentCorpus, metric: str
    ) -> DMRerankOptimizer:
        return DMRerankOptimizer(self, corpus, metric, DMRerankHPObjective)


class DMRerankHPObjective(hyperopt.HPObjective):
    """Objective function of the dm_rerank hyperparameter optimizer. Sweeps
    the model-strength and gate-threshold blend parameters over the cached
    source and noul score batches; the expensive decision-model calls have
    already been made during _prepare, so each trial is a pure re-blend of
    the same per-document scores."""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        import annif.eval

        b = trial.suggest_float("model-strength", 0.0, 1.0)
        # the low end -1.0 makes (b=0, tau=-1.0) the exact pure-source
        # reference point (the sigmoid is >= 0.99995 there for all
        # noul in [0, 1]), so the optimizer can represent "don't use
        # the model" and a genuine source-dominant nudge (small b with
        # a near-identity gate)
        tau = trial.suggest_float("gate-threshold", -1.0, 0.5)
        eval_batch = annif.eval.EvaluationBatch(args["subject_index"])
        blended_lists = [
            DMRerankBackend._blend(source, noul, b, tau)
            for source, noul in zip(args["source_suggestions"], args["noul_scores"])
        ]
        eval_batch.evaluate_many(blended_lists, args["gold_batches"])
        results = eval_batch.results(metrics=[args["metric"]])
        return results[args["metric"]]


class DMRerankOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the dm_rerank backend. The decision
    model is queried once per document (in _prepare) and the
    model-strength / gate-threshold blend parameters are then searched
    over the cached scores with TPE. The search space is model-strength
    in [0, 1] and gate-threshold in [-1.0, 0.5]; the point
    (model-strength=0, gate-threshold=-1.0) reproduces the pure source
    order exactly, so the search can fall back to the source ranking
    if the noul scores carry no useful signal, and the region just
    above it is the source-dominant 'nudge' where the model adds a
    small correction on top of the source scores."""

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        sources = parse_sources(self._backend.params["sources"])
        source_ids = [project_id for project_id, _ in sources]
        weights = [weight for _, weight in sources]
        limit = int(self._backend.params["limit"])
        max_candidates = int(self._backend.params["max-candidates"])

        psmap = annif.parallel.ProjectSuggestMap(
            self._backend.project.registry,
            source_ids,
            backend_params=None,
            limit=None,
            threshold=0.0,
        )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        # materialize the corpus once: doc_batches is a one-shot generator
        # and the documents are needed again for the model queries below;
        # apply the project transform (e.g. transform=limit(5000)) so the
        # source suggestions and the decision model queries match the
        # suggest path
        transform = self._backend.project.transform
        doc_batches = [
            [transform.transform_doc(doc) for doc in batch]
            for batch in self._corpus.doc_batches
        ]
        documents = [doc for batch in doc_batches for doc in batch]

        self._backend.info(
            "generating source suggestions for {} documents".format(len(documents))
        )
        source_suggestions = []
        gold_batches = []
        with pool_class(jobs) as pool:
            results = pool.map(psmap.suggest_batch, doc_batches)

        for batch, (suggestions_by_source, subject_sets) in zip(doc_batches, results):
            # merge the per-source batches the same way as the suggest
            # path
            merged = SuggestionBatch.from_averaged(
                [suggestions_by_source[project_id] for project_id in source_ids],
                weights,
            ).filter(limit=limit)
            for idx in range(len(batch)):
                candidates = list(merged[idx])
                if max_candidates > 0:
                    candidates = candidates[:max_candidates]
                source_suggestions.append(candidates)
                gold_batches.append(subject_sets[idx])

        # query the decision model once per document: this is the
        # expensive part, but it is only done once, since the blend
        # parameters are searched over these cached scores afterwards
        self._backend.info(
            "querying the decision model reranking service for {} "
            "documents".format(len(documents))
        )
        noul_scores = [
            self._backend._query_dm_rerank(
                doc, source_suggestions[idx], self._backend.params
            )
            for idx, doc in enumerate(documents)
        ]

        return {
            "source_suggestions": source_suggestions,
            "noul_scores": noul_scores,
            "gold_batches": gold_batches,
            "subject_index": self._backend.project.subjects,
            "metric": self._metric,
        }

    def _postprocess(self, study: Study) -> HPRecommendation:
        best = study.best_params
        lines = [
            f"model-strength={best['model-strength']:.4f}",
            f"gate-threshold={best['gate-threshold']:.4f}",
        ]
        return hyperopt.HPRecommendation(lines=lines, score=study.best_value)
