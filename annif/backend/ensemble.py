"""Ensemble backend that combines results from multiple projects"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_array

import annif.eval
import annif.parallel
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SuggestionBatch

from . import backend, hyperopt

if TYPE_CHECKING:
    from datetime import datetime

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus.document import Document, DocumentCorpus


class BaseEnsembleBackend(backend.AnnifBackend):
    """Base class for ensemble backends"""

    # cache of subject ID mappings per source project
    _subject_map_cache = None

    def _get_sources_attribute(self, attr: str) -> list[bool | None]:
        params = self._get_backend_params(None)
        sources = annif.util.parse_sources(params["sources"])
        return [
            getattr(self.project.registry.get_project(project_id), attr)
            for project_id, _ in sources
        ]

    def initialize(self, parallel: bool = False) -> None:
        # initialize all the source projects
        params = self._get_backend_params(None)
        for project_id, _ in annif.util.parse_sources(params["sources"]):
            project = self.project.registry.get_project(project_id)
            project.initialize(parallel)

    def _suggest_with_sources(
        self, documents: list[Document], sources: list[tuple[str, float]]
    ) -> dict[str, SuggestionBatch]:
        return {
            project_id: self.project.registry.get_project(project_id).suggest(documents)
            for project_id, _ in sources
        }

    def _same_vocabulary(self, source_project_id: str) -> bool:
        """Check whether the vocabulary of the given source project is
        the same as the ensemble vocabulary, based on subject URIs."""

        source_subjects = self.project.registry.get_project(source_project_id).subjects
        target_subjects = self.project.subjects
        if source_subjects is target_subjects:
            return True

        source_active = source_subjects.active
        return len(source_active) == len(target_subjects.active) and all(
            target_subjects.by_uri(subject.uri, warnings=False) == subject_id
            for subject_id, subject in source_active
        )

    def _source_subject_map(self, source_project_id: str) -> np.ndarray | None:
        """Return an array that maps subject IDs of the given source project
        to subject IDs of the ensemble project, or None when the source
        vocabulary already matches the ensemble vocabulary."""

        source_subjects = self.project.registry.get_project(source_project_id).subjects
        target_subjects = self.project.subjects

        if self._subject_map_cache is None:
            self._subject_map_cache = {}
        cached = self._subject_map_cache.get(source_project_id)
        if (
            cached is not None
            and cached[0] is source_subjects
            and cached[1] is target_subjects
        ):
            return cached[2]

        if len(source_subjects) == len(target_subjects) and self._same_vocabulary(
            source_project_id
        ):
            subject_map = None
        else:
            subject_map = np.full(len(source_subjects), -1, dtype=np.int64)
            for subject_id, subject in source_subjects.active:
                target_id = target_subjects.by_uri(subject.uri, warnings=False)
                if target_id is not None:
                    subject_map[subject_id] = target_id

        self._subject_map_cache[source_project_id] = (
            source_subjects,
            target_subjects,
            subject_map,
        )
        return subject_map

    def _check_source_width(
        self, source_project_id: str, width: int, expected: int, kind: str = "batch"
    ) -> None:
        """Raise NotInitializedException if the width of a source project's
        output does not match its vocabulary, as happens with an unretrained
        model after a vocabulary change."""

        if width != expected:
            raise NotInitializedException(
                f"Source '{source_project_id}' returned a {kind} with "
                f"{width} columns, but its vocabulary has {expected} "
                "subjects. Retrain the source project after vocabulary changes.",
                backend_id=self.backend_id,
            )

    def _align_source_vector(
        self, vector: np.ndarray, subject_map: np.ndarray | None, source_project_id: str
    ) -> np.ndarray:
        """Map a dense score vector of a single source project from its own
        vocabulary space into the ensemble vocabulary space."""

        expected = (
            len(self.project.subjects) if subject_map is None else len(subject_map)
        )
        self._check_source_width(
            source_project_id, len(vector), expected, "score vector"
        )

        if subject_map is None:
            return vector
        aligned = np.zeros(len(self.project.subjects), dtype=np.float32)
        keep = subject_map >= 0
        aligned[subject_map[keep]] = vector[keep]
        return aligned

    def _align_batch(
        self,
        batch: SuggestionBatch,
        source_project_id: str,
    ) -> SuggestionBatch:
        """Align a source batch with the ensemble vocabulary."""

        target_subjects = self.project.subjects
        subject_map = self._source_subject_map(source_project_id)
        expected = len(target_subjects) if subject_map is None else len(subject_map)
        self._check_source_width(source_project_id, batch.array.shape[1], expected)

        if subject_map is None:
            return batch
        array = batch.array.tocoo()
        target_cols = subject_map[array.col]
        keep = target_cols >= 0
        return SuggestionBatch(
            csr_array(
                (array.data[keep], (array.row[keep], target_cols[keep])),
                shape=(array.shape[0], len(target_subjects)),
                dtype=np.float32,
            )
        )

    def _merge_source_batches(
        self,
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        """Merge the given SuggestionBatches from each source into a single
        SuggestionBatch. The default implementation computes a weighted
        average based on the weights given in the sources tuple. Intended
        to be overridden in subclasses."""

        batches = [
            self._align_batch(batch_by_source[project_id], project_id)
            for project_id, _ in sources
        ]
        weights = [weight for _, weight in sources]
        return SuggestionBatch.from_averaged(batches, weights).filter(
            limit=int(params["limit"])
        )

    def _suggest_batch(
        self, documents: list[Document], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        batch_by_source = self._suggest_with_sources(documents, sources)
        return self._merge_source_batches(batch_by_source, sources, params)


class EnsembleHPObjective(hyperopt.HPObjective):
    """Objective function of the ensemble hyperparameter optimizer"""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        eval_batch = annif.eval.EvaluationBatch(args["subject_index"])
        proj_weights = {
            project_id: trial.suggest_float(project_id, 0.0, 1.0)
            for project_id in args["sources"]
        }
        for gold_batch, src_batches in zip(
            args["gold_batches"], args["source_batches"]
        ):
            batches = [src_batches[project_id] for project_id in args["sources"]]
            weights = [proj_weights[project_id] for project_id in args["sources"]]
            avg_batch = SuggestionBatch.from_averaged(batches, weights).filter(
                limit=int(args["limit"])
            )
            eval_batch.evaluate_many(avg_batch, gold_batch)
        results = eval_batch.results(metrics=[args["metric"]])
        return results[args["metric"]]


class EnsembleOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the ensemble backend"""

    def __init__(
        self, backend: EnsembleBackend, corpus: DocumentCorpus, metric: str
    ) -> None:
        super().__init__(backend, corpus, metric, EnsembleHPObjective)
        self._sources = [
            project_id
            for project_id, _ in annif.util.parse_sources(
                backend.config_params["sources"]
            )
        ]

    def _prepare(self, n_jobs: int = 1) -> dict[str, Any]:
        gold_batches = []
        source_batches = []

        for project_id in self._sources:
            project = self._backend.project.registry.get_project(project_id)
            project.initialize()

        psmap = annif.parallel.ProjectSuggestMap(
            self._backend.project.registry,
            self._sources,
            backend_params=None,
            # collect full source batches; the limit is applied after
            # alignment and averaging in the objective, as at inference time
            limit=None,
            threshold=0.0,
        )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        with pool_class(jobs) as pool:
            for suggestions, gold_batch in pool.imap_unordered(
                psmap.suggest_batch, self._corpus.doc_batches
            ):
                # align each batch with the ensemble vocabulary as it is received
                source_batches.append(
                    {
                        project_id: self._backend._align_batch(batch, project_id)
                        for project_id, batch in suggestions.items()
                    }
                )
                gold_batches.append(gold_batch)

        return {
            "gold_batches": gold_batches,
            "source_batches": source_batches,
            "subject_index": self._backend.project.subjects,
            "sources": self._sources,
            "limit": self._backend.params["limit"],
            "metric": self._metric,
        }

    def _normalize(self, hps: dict[str, float]) -> dict[str, float]:
        total = sum(hps.values())
        return {source: hps[source] / total for source in hps}

    def _format_cfg_line(self, hps: dict[str, float]) -> str:
        return "sources=" + ",".join(
            [f"{src}:{weight:.4f}" for src, weight in hps.items()]
        )

    def _postprocess(self, study: Study) -> HPRecommendation:
        line = self._format_cfg_line(self._normalize(study.best_params))
        return hyperopt.HPRecommendation(lines=[line], score=study.best_value)


class EnsembleBackend(BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that combines results from multiple projects"""

    name = "ensemble"

    @property
    def is_trained(self) -> bool:
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    @property
    def modification_time(self) -> datetime | None:
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def get_hp_optimizer(
        self, corpus: DocumentCorpus, metric: str
    ) -> EnsembleOptimizer:
        return EnsembleOptimizer(self, corpus, metric)

    def _train(self, corpus: DocumentCorpus, params: dict[str, Any], jobs: int = 0):
        raise NotSupportedException("Training ensemble backend is not possible.")
