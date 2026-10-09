"""Parallel processing functionality for Annif"""

from __future__ import annotations

import multiprocessing
import multiprocessing.dummy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections import defaultdict
    from collections.abc import Iterator
    from typing import Callable

    from annif.corpus import Document, SubjectSet
    from annif.registry import AnnifRegistry
    from annif.suggestion import SuggestionBatch, SuggestionResult, SuggestionResults


# Start method for processes created by the multiprocessing module.
# A value of None means using the platform-specific default.
# Intended to be overridden in unit tests.
MP_START_METHOD = None


class BaseWorker:
    """Base class for workers that implement tasks executed via
    multiprocessing. The init method can be used to store data objects that
    are necessary for the operation. They will be stored in a class
    attribute that is accessible to the static worker method. The storage
    solution is inspired by this blog post:
    https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution # noqa
    """

    args = None

    @classmethod
    def init(cls, args) -> None:
        cls.args = args  # pragma: no cover


class ProjectSuggestMap:
    """A utility class that can be used to wrap one or more projects and
    provide a mapping method that converts Document objects to suggestions.
    Intended to be used with the multiprocessing module."""

    def __init__(
        self,
        registry: AnnifRegistry,
        project_ids: list[str],
        backend_params: defaultdict[str, Any] | None,
        limit: int | None,
        threshold: float,
    ) -> None:
        self.registry = registry
        self.project_ids = project_ids
        self.backend_params = backend_params
        self.limit = limit
        self.threshold = threshold

    def suggest(self, doc: Document) -> tuple[dict[str, SuggestionResult], SubjectSet]:
        filtered_hits = {}
        for project_id in self.project_ids:
            project = self.registry.get_project(project_id)
            batch = project.suggest([doc], self.backend_params)
            filtered_hits[project_id] = batch.filter(self.limit, self.threshold)[0]
        return (filtered_hits, doc.subject_set)

    def suggest_batch(
        self, batch
    ) -> tuple[dict[str, SuggestionBatch], Iterator[SubjectSet]]:
        filtered_hit_sets = {}
        subject_sets = [doc.subject_set for doc in batch]

        for project_id in self.project_ids:
            project = self.registry.get_project(project_id)
            suggestion_batch = project.suggest(batch, self.backend_params)
            filtered_hit_sets[project_id] = suggestion_batch.filter(
                self.limit, self.threshold
            )
        return (filtered_hit_sets, subject_sets)


def get_pool(n_jobs: int) -> tuple[int | None, Callable]:
    """return a suitable constructor for multiprocessing pool class, and the correct
    jobs argument for it, for the given amount of parallel jobs"""

    ctx = multiprocessing.get_context(MP_START_METHOD)

    if n_jobs < 1:
        n_jobs = None
        pool_constructor: Callable = ctx.Pool
    elif n_jobs == 1:
        # use the dummy wrapper around threading to avoid subprocess overhead
        pool_constructor = multiprocessing.dummy.Pool
    else:
        pool_constructor = ctx.Pool

    return n_jobs, pool_constructor


class OptimizeEvaluator(BaseWorker):
    """A utility class that can be used to evaluate different limit and
    threshold parameter combinations of stored suggestion results against
    gold standard subjects. The evaluator instance is passed to the worker
    processes once, as pool initializer arguments (see BaseWorker), instead
    of being marshalled with every combination. Intended to be used with the
    multiprocessing module."""

    def __init__(
        self,
        registry: AnnifRegistry,
        project_id: str,
        suggestion_results: SuggestionResults,
        subject_set_batches: list[list[SubjectSet]],
        metrics_to_optimize: list[str],
    ) -> None:
        self.registry = registry
        self.project_id = project_id
        self.suggestion_results = suggestion_results
        self.subject_set_batches = subject_set_batches
        self.metrics_to_optimize = metrics_to_optimize

    @classmethod
    def evaluate_combination(
        cls, limit_threshold: tuple[int, float]
    ) -> tuple[int, float, dict[str, float]]:
        import annif.eval

        limit, threshold = limit_threshold
        self = cls.args

        project = self.registry.get_project(self.project_id)
        eval_batch = annif.eval.EvaluationBatch(project.subjects)

        filtered_results = self.suggestion_results.filter(limit, threshold)
        for batch, subject_sets in zip(
            filtered_results.batches, self.subject_set_batches
        ):
            eval_batch.evaluate_many(batch, subject_sets)

        results = eval_batch.results(metrics=self.metrics_to_optimize)

        return (limit, threshold, results)
