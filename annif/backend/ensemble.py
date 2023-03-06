"""Ensemble backend that combines results from multiple projects"""


import annif.eval
import annif.parallel
import annif.suggestion
import annif.util
from annif.exception import NotSupportedException

from . import backend, hyperopt


class BaseEnsembleBackend(backend.AnnifBackend):
    """Base class for ensemble backends"""

    def _get_sources_attribute(self, attr):
        params = self._get_backend_params(None)
        sources = annif.util.parse_sources(params["sources"])
        return [
            getattr(self.project.registry.get_project(project_id), attr)
            for project_id, _ in sources
        ]

    def initialize(self, parallel=False):
        # initialize all the source projects
        params = self._get_backend_params(None)
        for project_id, _ in annif.util.parse_sources(params["sources"]):
            project = self.project.registry.get_project(project_id)
            project.initialize(parallel)

    def _normalize_hits(self, hits, source_project):
        """Hook for processing hits from backends. Intended to be overridden
        by subclasses."""
        return hits

    def _suggest_with_sources(self, texts, sources):
        hit_sets_from_sources = []
        for project_id, weight in sources:
            source_project = self.project.registry.get_project(project_id)
            hit_sets = source_project.suggest(texts)
            norm_hit_sets = [
                self._normalize_hits(hits, source_project) for hits in hit_sets
            ]
            hit_sets_from_sources.append(
                [
                    annif.suggestion.WeightedSuggestion(
                        hits=norm_hits, weight=weight, subjects=source_project.subjects
                    )
                    for norm_hits in norm_hit_sets
                ]
            )
        return hit_sets_from_sources

    def _merge_hit_sets_from_sources(self, hit_sets_from_sources, params):
        """Hook for merging hits from sources. Can be overridden by
        subclasses."""
        return [
            annif.util.merge_hits(hits, len(self.project.subjects))
            for hits in hit_sets_from_sources
        ]

    def _suggest_batch(self, texts, params):
        sources = annif.util.parse_sources(params["sources"])
        hit_sets_from_sources = self._suggest_with_sources(texts, sources)
        return self._merge_hit_sets_from_sources(hit_sets_from_sources, params)


class EnsembleOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the ensemble backend"""

    def __init__(self, backend, corpus, metric):
        super().__init__(backend, corpus, metric)
        self._sources = [
            project_id
            for project_id, _ in annif.util.parse_sources(
                backend.config_params["sources"]
            )
        ]

    def _prepare(self, n_jobs=1):
        self._gold_subjects = []
        self._source_hits = []

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
            for hit_sets, subject_sets in pool.imap_unordered(
                psmap.suggest_batch, self._corpus.doc_batches
            ):
                self._gold_subjects.extend(subject_sets)
                self._source_hits.extend(self._hit_sets_to_list(hit_sets))

    def _hit_sets_to_list(self, hit_sets):
        """Convert a dict of lists of hits to a list of dicts of hits, e.g.
        {"proj-1": [p-1-doc-1-hits, p-1-doc-2-hits]
         "proj-2": [p-2-doc-1-hits, p-2-doc-2-hits]}
        to
        [{"proj-1": p-1-doc-1-hits, "proj-2": p-2-doc-1-hits},
         {"proj-1": p-1-doc-2-hits, "proj-2": p-2-doc-2-hits}]
        """
        return [
            dict(zip(hit_sets.keys(), doc_hits)) for doc_hits in zip(*hit_sets.values())
        ]

    def _normalize(self, hps):
        total = sum(hps.values())
        return {source: hps[source] / total for source in hps}

    def _format_cfg_line(self, hps):
        return "sources=" + ",".join(
            [f"{src}:{weight:.4f}" for src, weight in hps.items()]
        )

    def _objective(self, trial):
        batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        weights = {
            project_id: trial.suggest_uniform(project_id, 0.0, 1.0)
            for project_id in self._sources
        }
        for goldsubj, srchits in zip(self._gold_subjects, self._source_hits):
            weighted_hits = []
            for project_id, hits in srchits.items():
                weighted_hits.append(
                    annif.suggestion.WeightedSuggestion(
                        hits=hits,
                        weight=weights[project_id],
                        subjects=self._backend.project.subjects,
                    )
                )
            batch.evaluate(
                annif.util.merge_hits(
                    weighted_hits, len(self._backend.project.subjects)
                ),
                goldsubj,
            )
        results = batch.results(metrics=[self._metric])
        return results[self._metric]

    def _postprocess(self, study):
        line = self._format_cfg_line(self._normalize(study.best_params))
        return hyperopt.HPRecommendation(lines=[line], score=study.best_value)


class EnsembleBackend(BaseEnsembleBackend, hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that combines results from multiple projects"""

    name = "ensemble"

    @property
    def is_trained(self):
        sources_trained = self._get_sources_attribute("is_trained")
        return all(sources_trained)

    @property
    def modification_time(self):
        mtimes = self._get_sources_attribute("modification_time")
        return max(filter(None, mtimes), default=None)

    def get_hp_optimizer(self, corpus, metric):
        return EnsembleOptimizer(self, corpus, metric)

    def _train(self, corpus, params, jobs=0):
        raise NotSupportedException("Training ensemble backend is not possible.")
