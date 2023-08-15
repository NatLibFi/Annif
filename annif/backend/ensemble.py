"""Ensemble backend that combines results from multiple projects"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import annif.eval
import annif.parallel
import annif.util
from annif.exception import NotSupportedException
from annif.suggestion import SuggestionBatch

from . import backend, hyperopt

if TYPE_CHECKING:
    from datetime import datetime

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus.document import DocumentCorpus


class BaseEnsembleBackend(backend.AnnifBackend):
    """Base class for ensemble backends"""

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
        self, texts: list[str], sources: list[tuple[str, float]]
    ) -> dict[str, SuggestionBatch]:
        return {
            project_id: self.project.registry.get_project(project_id).suggest(texts)
            for project_id, _ in sources
        }

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

        batches = [batch_by_source[project_id] for project_id, _ in sources]
        weights = [weight for _, weight in sources]
        return SuggestionBatch.from_averaged(batches, weights).filter(
            limit=int(params["limit"])
        )

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        sources = annif.util.parse_sources(params["sources"])
        batch_by_source = self._suggest_with_sources(texts, sources)
        return self._merge_source_batches(batch_by_source, sources, params)


class EnsembleOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the ensemble backend"""

    def __init__(
        self, backend: EnsembleBackend, corpus: DocumentCorpus, metric: str
    ) -> None:
        super().__init__(backend, corpus, metric)
        self._sources = [
            project_id
            for project_id, _ in annif.util.parse_sources(
                backend.config_params["sources"]
            )
        ]

    def _prepare(self, n_jobs: int = 1) -> None:
        self._gold_batches = []
        self._source_batches = []

        for project_id in self._sources:
            project = self._backend.project.registry.get_project(project_id)
            project.initialize()

        psmap = annif.parallel.ProjectSuggestMap(
            self._backend.project.registry,
            self._sources,
            backend_params=None,
            limit=int(self._backend.params["limit"]),
            threshold=0.0,
        )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        with pool_class(jobs) as pool:
            for suggestions, gold_batch in pool.imap_unordered(
                psmap.suggest_batch, self._corpus.doc_batches
            ):
                self._source_batches.append(suggestions)
                self._gold_batches.append(gold_batch)

    def _normalize(self, hps: dict[str, float]) -> dict[str, float]:
        total = sum(hps.values())
        return {source: hps[source] / total for source in hps}

    def _format_cfg_line(self, hps: dict[str, float]) -> str:
        return "sources=" + ",".join(
            [f"{src}:{weight:.4f}" for src, weight in hps.items()]
        )

    def _objective(self, trial: Trial) -> float:
        eval_batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        proj_weights = {
            project_id: trial.suggest_float(project_id, 0.0, 1.0)
            for project_id in self._sources
        }
        for gold_batch, src_batches in zip(self._gold_batches, self._source_batches):
            batches = [src_batches[project_id] for project_id in self._sources]
            weights = [proj_weights[project_id] for project_id in self._sources]
            avg_batch = SuggestionBatch.from_averaged(batches, weights).filter(
                limit=int(self._backend.params["limit"])
            )
            eval_batch.evaluate_many(avg_batch, gold_batch)
        results = eval_batch.results(metrics=[self._metric])
        return results[self._metric]

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
