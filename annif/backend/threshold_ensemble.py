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
        super().__init__(backend, corpus, metric)
        self._objective = ThresholdEnsembleHPObjective

    def _prepare(self, n_jobs: int = 1):
        args = super()._prepare(n_jobs)
        args["backend"] = self._backend
        args["sources"] = annif.util.parse_sources(
            self._backend.config_params["sources"]
        )
        return args

    def _postprocess(self, study: Study) -> HPRecommendation:
        line = f"threshold={study.best_params['threshold']:.4f}"

        isoelastic_x, isoelastic_y = self.calculate_isoelastic_point(study)

        print(f"Found isoelastic point with score {isoelastic_y:.4f} with:")
        print("---")
        print(f"threshold={isoelastic_x:.4f}")
        print("---")

        return hyperopt.HPRecommendation(
            lines=[line],
            score=study.best_value,
        )

    def calculate_isoelastic_point(self, study):
        """Calculate the isoelastic point assuming the optimization curve
        follows a logarithmic relationship: y = a + b * ln(x)
        The isoelastic point is where dy/dx = 1, which occurs at x = b.
        """
        trials = [
            trial
            for trial in study.get_trials()
            if trial.value is not None
            and "threshold" in trial.params
            and trial.params["threshold"] > 0
        ]

        x = np.array([t.params["threshold"] for t in trials])
        y = np.array([t.value for t in trials])

        b, a = np.polyfit(np.log(x), y, 1)

        isoelastic_x = b
        isoelastic_y = a + b * np.log(isoelastic_x)

        return isoelastic_x, isoelastic_y


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
        self.filter = bool(config_params.get("filter", False))
        super().__init__(backend_id, config_params, project)

    def get_hp_optimizer(
        self,
        corpus: DocumentCorpus,
        metric: str,
    ) -> ThresholdEnsembleOptimizer:
        return ThresholdEnsembleOptimizer(self, corpus, metric)

    def _align_batch(
        self,
        batch: SuggestionBatch,
        source_project_id: str,
    ) -> SuggestionBatch:
        """Align a compact source batch with the ensemble vocabulary."""
        source_subjects = self.project.registry.get_project(source_project_id).subjects
        target_subjects = self.project.subjects

        if batch.array.shape[1] == len(target_subjects):
            return batch

        source_active = source_subjects.active
        if batch.array.shape[1] != len(source_active):
            # TODO: use Annif cli warning machinery
            raise ValueError(
                f"Source '{source_project_id}' has {batch.array.shape[1]} "
                f"columns, but its vocabulary has {len(source_active)} "
                "active subjects."
            )

        array = batch.array.tocoo()
        target_cols = np.empty_like(array.col)

        for source_col in np.unique(array.col):
            target_cols[array.col == source_col] = target_subjects.by_uri(
                source_active[source_col][1].uri,
                warnings=False,
            )

        return SuggestionBatch(
            csr_array(
                (array.data, (array.row, target_cols)),
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
        SuggestionBatch. In this implementation, a source is activated for
        a document only if it has at least one suggestion at or above the
        configured threshold. Among activated sources, their configured weights
        determine the weighted average. When there is only one source, scores
        below the threshold are removed entirely rather than merely being
        used for source activation."""

        threshold = float(params.get("threshold", self.threshold))
        filter_b = bool(params.get("filter", self.filter))
        limit = int(params.get("limit", 10))
        first_batch = next(iter(batch_by_source.values()))

        n_docs = first_batch.array.shape[0]
        n_subjects = len(self.project.subjects)

        weight_sum = np.zeros(n_docs, dtype="float32")
        weighted_sum = csr_array(
            (n_docs, n_subjects),
            dtype="float32",
        )

        for project_id, source_weight in sources:
            batch = batch_by_source[project_id]

            if not batch:
                continue

            batch = self._align_batch(batch, project_id)

            filtered_batch = batch.filter(threshold=threshold)
            active = np.diff(filtered_batch.array.indptr) > 0

            if not np.any(active):
                continue

            weight_sum[active] += source_weight

            weighted_sum += (
                (filtered_batch.array if filter_b else batch.array)
                .multiply(active[:, None] * source_weight)
                .tocsr()
            )

        if not np.any(weight_sum):
            return SuggestionBatch(
                csr_array(
                    (n_docs, n_subjects),
                    dtype="float32",
                )
            )

        inverse_weight_sum = np.zeros_like(weight_sum)
        active_documents = weight_sum > 0
        inverse_weight_sum[active_documents] = 1.0 / weight_sum[active_documents]

        averaged_array = weighted_sum.multiply(inverse_weight_sum[:, None]).tocsr()
        averaged_batch = SuggestionBatch(averaged_array)

        return averaged_batch.filter(limit=limit)
