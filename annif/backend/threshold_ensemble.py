"""Threshold-based ensemble that activates sources based on a score threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_array

import annif.eval
import annif.parallel
import annif.util
from annif.suggestion import SuggestionBatch

from . import hyperopt
from .ensemble import EnsembleBackend, EnsembleOptimizer

if TYPE_CHECKING:
    from optuna.study import Study

    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus.document import DocumentCorpus
    from annif.project import AnnifProject


class ThresholdEnsembleHPObjective(hyperopt.HPObjective):
    """Objective function of the threshold ensemble hyperparameter optimizer."""

    @classmethod
    def objective(cls, trial, args) -> float:
        threshold = trial.suggest_float(
            "threshold",
            0.001,
            0.5,
            log=True,
        )

        eval_batch = annif.eval.EvaluationBatch(args["subject_index"])

        for gold_batch, source_batches in zip(
            args["gold_batches"],
            args["source_batches"],
        ):
            merged_batch = args["backend"]._merge_source_batches(
                source_batches,
                args["sources"],
                {
                    "limit": args["limit"],
                    "threshold": threshold,
                },
            )
            eval_batch.evaluate_many(merged_batch, gold_batch)

        results = eval_batch.results(metrics=[args["metric"]])
        return results[args["metric"]]


class ThresholdEnsembleOptimizer(EnsembleOptimizer):
    """Hyperparameter optimizer for the threshold ensemble."""

    def __init__(
        self,
        backend: ThresholdEnsembleBackend,
        corpus: DocumentCorpus,
        metric: str,
    ) -> None:
        hyperopt.HyperparameterOptimizer.__init__(
            self,
            backend,
            corpus,
            metric,
            ThresholdEnsembleHPObjective,
        )

        self._sources = [
            project_id
            for project_id, _ in annif.util.parse_sources(
                backend.config_params["sources"]
            )
        ]

    def _prepare(self, n_jobs):
        args = super()._prepare(n_jobs)
        args["backend"] = self._backend
        return args

    def _postprocess(self, study: Study) -> HPRecommendation:
        line = f"threshold={study.best_params['threshold']:.4f}"

        return hyperopt.HPRecommendation(
            lines=[line],
            score=study.best_value,
        )


class ThresholdEnsembleBackend(EnsembleBackend):
    """Ensemble backend that activates sources based on a score threshold."""

    name = "threshold_ensemble"

    def __init__(
        self,
        backend_id: str,
        config_params: dict[str, Any],
        project: "AnnifProject",
    ):
        self.threshold = float(config_params.get("threshold", 0.1))
        super().__init__(backend_id, config_params, project)

    def get_hp_optimizer(
        self,
        corpus: DocumentCorpus,
        metric: str,
    ) -> ThresholdEnsembleOptimizer:
        return ThresholdEnsembleOptimizer(self, corpus, metric)

    def _merge_source_batches(
        self,
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        """Merge the given SuggestionBatches from each source into a single
        SuggestionBatch. In this implementation, a source is activated for
        a document only if it has at least one suggestion at or above the
        configured threshold. Among activated sources, their configured weights
        determine the weighted average. When there is only one source, scores
        below the threshold are removed entirely rather than merely being
        used for source activation."""

        threshold = float(params.get("threshold", self.threshold))
        limit = int(params.get("limit", 10))
        first_batch = next(iter(batch_by_source.values()))

        # With a single source, apply a hard threshold directly.
        if len(sources) == 1:
            return first_batch.filter(
                threshold=threshold,
                limit=limit,
            )

        # Accumulate the weighted predictions for every document.
        n_docs, n_subjects = first_batch.array.shape
        weighted_sum = csr_array(
            (n_docs, n_subjects),
            dtype="float32",
        )

        # For each document, accumulate the weights of the sources that
        # are actually active for that document.
        weight_sum = np.zeros(n_docs, dtype="float64")

        for project_id, source_weight in sources:
            batch = batch_by_source[project_id]

            if not batch:
                continue

            # The filtered batch is used ONLY for the activation decision.
            # The original batch.array is still used for the weighted sum.
            filtered_batch = batch.filter(threshold=threshold)

            # This gives us one boolean value per document without iterating
            # over SuggestionResult objects in Python.
            active = np.diff(filtered_batch.array.indptr) > 0

            if not np.any(active):
                continue

            # Add this source's configured weight to the denominator for
            # every document for which this source is active.
            weight_sum[active] += source_weight

            # Add this source's original prediction matrix to the numerator,
            # but only for active documents.
            weighted_sum += batch.array.multiply(
                # `active[:, None]` gives one multiplier (0/1) per document:
                active[:, None]
                * source_weight
            ).tocsr()

        # If no source was activated for any document, return an empty
        # SuggestionBatch with the same shape as the source batches.
        if not np.any(weight_sum):
            return SuggestionBatch(csr_array((n_docs, n_subjects), dtype="float32"))

        # Avoid division by zero for documents where no source was active.
        inverse_weight_sum = np.zeros_like(weight_sum)

        # Dividing each row by its corresponding weight_sum gives the same
        # weighted-average calculation as SuggestionBatch.from_averaged(),
        # but with a different set of active sources for each document.
        active_documents = weight_sum > 0
        inverse_weight_sum[active_documents] = 1.0 / weight_sum[active_documents]

        averaged_array = weighted_sum.multiply(inverse_weight_sum[:, None]).tocsr()
        averaged_batch = SuggestionBatch(averaged_array)

        return averaged_batch.filter(limit=limit)
