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
    from datetime import datetime

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus.document import Document, DocumentCorpus
    from annif.project import AnnifProject


# Damping for the odds-ratio term of the blend
ODDS_EPS = 0.01

# Built-in indexing-policy rules prepended to the document input when
# 'input-rules' is enabled: the format/genre rule.
INPUT_RULES = (
    "You are a librarian performing topical subject indexing. "
    "A document can have several central subjects. "
    "Do NOT assign the document's own format or genre as a subject "
    "(a book of recipes is not about 'cookbooks', a pattern book is not "
    "about 'handicraft patterns', a travel book is not about "
    "'travelogues').\n\nDocument to be indexed:\n"
)

# When the reranking service rejects a request (HTTP 400), retry it with
# the input reduced to these fractions of the original, in order.
INPUT_REDUCE_FRACTIONS = (1.0, 0.75, 0.5)


class _RequestRejected(Exception):
    """Internal signal for a request rejected by the service (HTTP 400)."""


def _squash(raw: float) -> float:
    """Map an unbounded raw blend score into [0, 1) with a sigmoid-like
    monotonic squash: f(x) = x / (1 + x) for x >= 0."""
    if raw <= 0.0:
        return 0.0
    return raw / (1.0 + raw)


class DMRerankBackend(ensemble.BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble-style backend that blends source candidates with scores
    from a decision model reranking service, which answers a true/false
    (predicate) question about each candidate."""

    name = "dm_rerank"

    DEFAULT_PARAMETERS = {
        "endpoint": "http://localhost:9931/v1/systemone",
        "model": "",
        "model-strength": 0.3,
        "gate-threshold": -1.0,
        "retries": 2,
        "timeout": 60,
        "input-rules": False,
        "input-prefix": "",
        "input-transform": "pass",
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
        self._input_transforms: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        """The backend itself is untrained, but it requires all source
        projects to be trained."""
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    @property
    def modification_time(self) -> datetime | None:
        """The backend has no model files of its own, so the most recent
        modification time of the source projects is reported."""
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def _get_input_transform(self, params: dict[str, Any]):
        """The transform applied to the document text before it is sent
        to the reranking service as the question input, on top of the
        project transform. Defaults to pass (no extra transformation).
        Useful for keeping the input within the model's context length
        while the source projects run on the full text. Resolved from the
        effective parameters, so per-call overrides are honored; the
        built transform is cached per specification."""
        spec = str(params["input-transform"])
        if spec not in self._input_transforms:
            self._input_transforms[spec] = annif.transform.get_transform(
                spec, project=self.project
            )
        return self._input_transforms[spec]

    def _train(
        self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0
    ) -> None:
        raise NotSupportedException("Training the dm_rerank backend is not possible.")

    def _label_for_subject(self, subject_id: int) -> str | None:
        """Return a usable label for the subject: the project-language label
        when it is nonempty, otherwise the first nonempty label in any
        language, else None (the candidate is skipped)."""
        try:
            subject = self.project.subjects[subject_id]
        except IndexError:  # deprecated subject
            return None
        if not subject.labels:
            return None
        preferred = subject.labels.get(self.project.language)
        if preferred:
            return preferred
        return next((label for label in subject.labels.values() if label), None)

    def _input_prefix(self, params: dict[str, Any]) -> str:
        """Return the text to prepend to the document input: the custom
        'input-prefix' when set, the built-in indexing-policy rules when
        'input-rules' is enabled, otherwise an empty string."""
        custom = str(params["input-prefix"])
        if custom:
            return custom
        if boolean(params["input-rules"]):
            return INPUT_RULES
        return ""

    def _post_with_transient_retries(
        self, endpoint: str, payload: dict[str, Any], timeout: float, retries: int
    ) -> requests.Response:
        """POST the payload, retrying transient failures (5xx, 408, 429,
        timeouts, connection errors) with exponential backoff. A 400
        response raises _RequestRejected; other 4xx responses raise
        OperationFailedException immediately."""
        attempt = 0
        while True:
            try:
                req = requests.post(endpoint, json=payload, timeout=timeout)
                req.raise_for_status()
                return req
            except requests.exceptions.RequestException as err:
                if isinstance(err, requests.exceptions.HTTPError):
                    response = err.response
                    status = response.status_code if response is not None else None
                    if status == 400:
                        raise _RequestRejected() from err
                    if status is None or (status < 500 and status not in (408, 429)):
                        # a deterministic 4xx (invalid payload, unauthorized
                        # endpoint, ...) will not succeed on retry, so fail fast
                        msg = f"dm_rerank request to {endpoint} failed: {err}"
                        raise OperationFailedException(msg) from err
                attempt += 1
                if attempt > retries:
                    msg = (
                        "dm_rerank request to {} failed after {} attempts: {}"
                    ).format(endpoint, retries + 1, err)
                    raise OperationFailedException(msg) from err
                # exponential backoff starting from 1 second: 1s, 2s, 4s, ...
                delay = 2 ** (attempt - 1)
                self.warning(
                    "dm_rerank request failed ({err}); retrying in {delay}s "
                    "({attempt}/{retries})".format(
                        err=err, delay=delay, attempt=attempt, retries=retries
                    )
                )
                time.sleep(delay)

    def _validate_instruction(self, instruction: str) -> None:
        """The instruction template must contain a {label} placeholder and
        be a valid format string."""
        if "{label}" not in instruction:
            raise ConfigurationException(
                "instruction parameter must contain a {label} placeholder"
            )
        try:
            instruction.format(label="test")
        except (KeyError, IndexError, ValueError) as err:
            msg = f"invalid instruction template: {err}"
            raise ConfigurationException(msg) from err

    def _build_systemone_questions(
        self, suggestions: list[SubjectSuggestion], instruction: str
    ) -> dict[str, dict[str, str]]:
        """Build the System One true/false question for each candidate
        subject that has a label, keyed by the subject ID. (The System One
        dialect names the binary question type 'noul' and carries the
        questions as a map keyed by question ID.)"""
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
        return questions

    @staticmethod
    def _reduced_input(input_text: str, fraction: float) -> str:
        """Truncate the input to the given fraction of its length."""
        if fraction >= 1.0:
            return input_text
        return input_text[: max(0, round(len(input_text) * fraction))]

    def _send_with_input_reduction(
        self,
        endpoint: str,
        input_prefix: str,
        input_text: str,
        questions: dict[str, dict[str, str]],
        model: str,
        retries: int,
        timeout: float,
    ) -> requests.Response | None:
        """Send the System One request, retrying on a 400 response with
        the input reduced to the fractions in INPUT_REDUCE_FRACTIONS of
        the original. Returns None if the request is still rejected after
        the last fraction."""
        for fraction_index, fraction in enumerate(INPUT_REDUCE_FRACTIONS):
            # the System One dialect names the shared input field 'state'
            payload = {
                "state": input_prefix + self._reduced_input(input_text, fraction),
                "questions": questions,
            }
            if model:
                payload["model"] = model
            try:
                return self._post_with_transient_retries(
                    endpoint, payload, timeout, retries
                )
            except _RequestRejected:
                if fraction_index + 1 >= len(INPUT_REDUCE_FRACTIONS):
                    self.warning(
                        "dm_rerank request to {endpoint} still rejected "
                        "(400) with the input reduced to {fraction:.0%}; "
                        "scoring this document with its source scores "
                        "only".format(endpoint=endpoint, fraction=fraction)
                    )
                    return None
                next_fraction = INPUT_REDUCE_FRACTIONS[fraction_index + 1]
                self.warning(
                    "dm_rerank request to {endpoint} rejected (400); "
                    "retrying with the input reduced to {fraction:.0%} of "
                    "the original".format(endpoint=endpoint, fraction=next_fraction)
                )
        return None

    def _parse_systemone_response(self, req: requests.Response) -> dict[int, float]:
        """Parse the System One response into a mapping subject_id ->
        decision model score (the 'noul' value: the probability that the
        predicate is true)."""
        try:
            response = req.json()
        except ValueError as err:
            msg = f"dm_rerank response JSON decode failed: {err}"
            raise OperationFailedException(msg) from err

        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            msg = "dm_rerank response does not contain a valid 'answers' object"
            raise OperationFailedException(msg)
        dm_scores = {}
        for subject_id_str, answer in answers.items():
            if not isinstance(answer, dict):
                msg = (
                    f"dm_rerank answer for subject {subject_id_str!r} "
                    "is not an object"
                )
                raise OperationFailedException(msg)
            score = answer.get("noul")
            if score is None:
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                msg = (
                    f"dm_rerank decision model score for subject "
                    f"{subject_id_str!r} is not a number: {score!r}"
                )
                raise OperationFailedException(msg)
            if not 0.0 <= score <= 1.0:
                msg = (
                    f"dm_rerank decision model score for subject "
                    f"{subject_id_str!r} is outside [0, 1]: {score!r}"
                )
                raise OperationFailedException(msg)
            try:
                subject_id = int(subject_id_str)
            except (TypeError, ValueError) as err:
                msg = f"dm_rerank answer key {subject_id_str!r} is not a subject ID"
                raise OperationFailedException(msg) from err
            dm_scores[subject_id] = float(score)

        if dm_scores:
            values = list(dm_scores.values())
            self.debug(
                "dm_rerank decision model scores: min {:.3f}, mean {:.3f}, "
                "max {:.3f} for {} candidates".format(
                    min(values),
                    sum(values) / len(values),
                    max(values),
                    len(values),
                )
            )

        return dm_scores

    def _query_dm_rerank(
        self, doc: Document, suggestions: list[SubjectSuggestion], params
    ) -> dict[int, float]:
        """Ask the reranking service for a decision model score for every
        candidate subject that has a label, and return a mapping
        subject_id -> decision model score. Candidates without a label are
        not included in the mapping.

        A 400 response is retried with the input reduced to the fractions
        in INPUT_REDUCE_FRACTIONS of the original; if every fraction is
        rejected, an empty mapping is returned so that the document keeps
        its plain source scores and the run is not aborted."""
        instruction = str(params["instruction"])
        self._validate_instruction(instruction)
        questions = self._build_systemone_questions(suggestions, instruction)
        if not questions:
            return {}

        input_prefix = self._input_prefix(params)
        input_text = self._get_input_transform(params).transform_doc(doc).text
        req = self._send_with_input_reduction(
            params["endpoint"].rstrip("/"),
            input_prefix,
            input_text,
            questions,
            params["model"],
            int(params["retries"]),
            float(params["timeout"]),
        )
        if req is None:
            return {}
        return self._parse_systemone_response(req)

    def _validate_params(self, params: dict[str, Any]) -> None:
        """Validate the effective parameters to avoid numerical issues."""
        raw = params["model-strength"]
        try:
            model_strength = float(raw)
        except (TypeError, ValueError) as err:
            msg = f"invalid model-strength parameter: {raw!r}"
            raise ConfigurationException(msg, backend_id=self.backend_id) from err
        if not math.isfinite(model_strength) or not 0.0 <= model_strength <= 2.0:
            msg = (
                f"model-strength {model_strength!r} is outside the valid "
                "range [0.0, 2.0]"
            )
            raise ConfigurationException(msg, backend_id=self.backend_id)

        raw = params["gate-threshold"]
        try:
            gate_threshold = float(raw)
        except (TypeError, ValueError) as err:
            msg = f"invalid gate-threshold parameter: {raw!r}"
            raise ConfigurationException(msg, backend_id=self.backend_id) from err
        if not math.isfinite(gate_threshold) or not -1.0 <= gate_threshold <= 1.0:
            msg = (
                f"gate-threshold {gate_threshold!r} is outside the valid "
                "range [-1.0, 1.0]"
            )
            raise ConfigurationException(msg, backend_id=self.backend_id)

    def _process_document(
        self, doc: Document, suggestions: list[SubjectSuggestion], params
    ) -> list[SubjectSuggestion]:
        """Re-score the candidate subjects of one document using the
        decision model scores from the reranking service and the
        model-strength / gate-threshold parameters."""
        dm_scores = self._query_dm_rerank(doc, suggestions, params)
        b = float(params["model-strength"])
        gate_threshold = float(params["gate-threshold"])
        return self._blend(suggestions, dm_scores, b, gate_threshold)

    @staticmethod
    def _blend(
        suggestions: list[SubjectSuggestion],
        dm_scores: dict[int, float],
        b: float,
        gate_threshold: float,
    ) -> list[SubjectSuggestion]:
        """Blend the source and decision model scores, keeping all
        candidates.

        Each candidate with a decision model score p (the probability that
        the predicate is true) is re-scored as

            raw = source_score
                  * ((p + eps) / (1 - p + eps))**b
                  * sigmoid((p - tau) / 0.1)

        where eps = 0.01, b is the model-strength, and tau is the
        gate-threshold; a candidate without a decision model score passes
        its plain source score through unmodified.

        A sufficiently negative gate-threshold (<= -1.0) makes the gate
        exactly 1.0 for all decision model scores in [0, 1], turning it
        fully off.

        The model-strength and gate-threshold are validated in
        _validate_params before blending: the model-strength is within
        [0.0, 2.0], bounding the exponent of the odds term so the
        exponentiation cannot overflow, and the gate-threshold is within
        [-1.0, 1.0], so the sigmoid gate is bounded and cannot overflow.

        All raw scores are mapped into [0, 1) with the monotonic squash."""
        if not suggestions:
            return []

        blended = []
        for suggestion in suggestions:
            dm_score = dm_scores.get(suggestion.subject_id)
            if dm_score is None:
                raw = suggestion.score
            else:
                if gate_threshold <= -1.0:
                    gate = 1.0
                else:
                    gate = 1.0 / (1.0 + math.exp(-(dm_score - gate_threshold) / 0.1))
                odds = (dm_score + ODDS_EPS) / (1.0 - dm_score + ODDS_EPS)
                raw = suggestion.score * odds**b * gate
            score = _squash(raw)
            blended.append(
                SubjectSuggestion(subject_id=suggestion.subject_id, score=score)
            )
        # the new order differs from the source order, so sort explicitly
        blended.sort(key=lambda s: s.score, reverse=True)
        return blended

    def _suggest_batch(
        self, documents: list[Document], params: dict[str, Any]
    ) -> SuggestionBatch:
        self._validate_params(params)
        # merge the source suggestions with the regular weighted average,
        # without applying the output limit yet: the reranking may promote
        # candidates beyond the limit
        sources = parse_sources(params["sources"])
        batch_by_source = self._suggest_with_sources(documents, sources)
        merged = SuggestionBatch.from_averaged(
            [batch_by_source[project_id] for project_id, _ in sources],
            [weight for _, weight in sources],
        )
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
    source and decision model score batches; the expensive decision-model
    calls have already been made during _prepare, so each trial is a pure
    re-blend of the same per-document scores."""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        import annif.eval

        # the odds-ratio blend can both boost and demote: b > 1 lets the
        # decision model lift confident "yes" candidates above their
        # source score
        b = trial.suggest_float("model-strength", 0.0, 2.0)
        # the low end -1.0 makes (b=0, tau=-1.0) the exact pure-source
        # reference point, so the search space can represent a genuine
        # source-dominant nudge (small b with a near-identity gate); the
        # pure-source configuration itself is evaluated as an enqueued
        # baseline trial, so the recommendation is never worse than the
        # source ranking
        tau = trial.suggest_float("gate-threshold", -1.0, 0.5)
        eval_batch = annif.eval.EvaluationBatch(args["subject_index"])
        limit = int(args["limit"])
        blended_lists = [
            DMRerankBackend._blend(source, dm_scores, b, tau)[:limit]
            for source, dm_scores in zip(args["source_suggestions"], args["dm_scores"])
        ]
        eval_batch.evaluate_many(blended_lists, args["gold_batches"])
        results = eval_batch.results(metrics=[args["metric"]])
        return results[args["metric"]]


class DMRerankOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the dm_rerank backend. The decision
    model is queried once per document (in _prepare) and the
    model-strength / gate-threshold blend parameters are then searched
    over the cached scores with TPE. The pure-source configuration
    (model-strength=0.0, gate-threshold=-1.0) is always evaluated as an
    enqueued baseline trial, so the recommendation can never be worse
    than the source ranking."""

    def _enqueue_trials(self, study: Study) -> None:
        # explicitly evaluate the pure-source fallback point: with
        # continuous ranges the optimizer would never sample it, and a
        # positive model strength or active gate breaks ties in the
        # source scores, so every sampled configuration can perform
        # worse than the original source ranking
        study.enqueue_trial(
            {"model-strength": 0.0, "gate-threshold": -1.0},
        )

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        sources = parse_sources(self._backend.params["sources"])
        source_ids = [project_id for project_id, _ in sources]
        weights = [weight for _, weight in sources]
        limit = int(self._backend.params["limit"])
        max_candidates = int(self._backend.params["max-candidates"])

        for project_id in source_ids:
            project = self._backend.project.registry.get_project(project_id)
            project.initialize()

        psmap = annif.parallel.ProjectSuggestMap(
            self._backend.project.registry,
            source_ids,
            backend_params=None,
            limit=None,
            threshold=0.0,
        )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        # process the corpus incrementally, batch by batch: apply the
        # project transform (e.g. transform=limit(5000)), generate the
        # source suggestions, and query the decision model once per
        # document, retaining only the per-document source scores, decision
        # model scores, and gold labels that the trials need; each batch's
        # documents can be garbage-collected as soon as it is done, so
        # the full text of the whole corpus never has to fit in memory
        transform = self._backend.project.transform
        source_suggestions = []
        dm_scores = []
        gold_batches = []
        n_documents = 0
        with pool_class(jobs) as pool:
            for batch in self._corpus.doc_batches:
                docs = [transform.transform_doc(doc) for doc in batch]

                # generate the source suggestions for this batch
                ((suggestions_by_source, subject_sets),) = pool.map(
                    psmap.suggest_batch, [docs]
                )

                # merge the per-source batch the same way as the suggest
                # path and query the decision model once per document:
                # this is the expensive part, but it is only done once,
                # since the blend parameters are searched over these
                # cached scores afterwards
                merged = SuggestionBatch.from_averaged(
                    [suggestions_by_source[project_id] for project_id in source_ids],
                    weights,
                )
                for idx in range(len(docs)):
                    candidates = list(merged[idx])
                    if max_candidates > 0:
                        candidates = candidates[:max_candidates]
                    source_suggestions.append(candidates)
                    gold_batches.append(subject_sets[idx])
                    dm_scores.append(
                        self._backend._query_dm_rerank(
                            docs[idx], candidates, self._backend.params
                        )
                    )
                n_documents += len(docs)

        if n_documents:
            self._backend.info(
                "queried the decision model reranking service for {} "
                "documents".format(n_documents)
            )

        return {
            "source_suggestions": source_suggestions,
            "dm_scores": dm_scores,
            "gold_batches": gold_batches,
            "limit": limit,
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
