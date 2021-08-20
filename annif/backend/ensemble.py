"""Ensemble backend that combines results from multiple projects"""


import annif.parallel
import annif.suggestion
import annif.util
import annif.eval
from . import backend
from . import hyperopt
from annif.exception import NotSupportedException


class BaseEnsembleBackend(backend.AnnifBackend):
    """Base class for ensemble backends"""

    def _get_sources_attribute(self, attr):
        params = self._get_backend_params(None)
        sources = annif.util.parse_sources(params['sources'])
        return [getattr(self.project.registry.get_project(project_id), attr)
                for project_id, _ in sources]

    def initialize(self, parallel=False):
        # initialize all the source projects
        params = self._get_backend_params(None)
        for project_id, _ in annif.util.parse_sources(params['sources']):
            project = self.project.registry.get_project(project_id)
            project.initialize(parallel)

    def _normalize_hits(self, hits, source_project):
        """Hook for processing hits from backends. Intended to be overridden
        by subclasses."""
        return hits

    def _suggest_with_sources(self, text, sources):
        hits_from_sources = []
        for project_id, weight in sources:
            source_project = self.project.registry.get_project(project_id)
            hits = source_project.suggest(text)
            self.debug(
                'Got {} hits from project {}, weight {}'.format(
                    len(hits), source_project.project_id, weight))
            norm_hits = self._normalize_hits(hits, source_project)
            hits_from_sources.append(
                annif.suggestion.WeightedSuggestion(
                    hits=norm_hits,
                    weight=weight,
                    subjects=source_project.subjects))
        return hits_from_sources

    def _merge_hits_from_sources(self, hits_from_sources, params):
        """Hook for merging hits from sources. Can be overridden by
        subclasses."""
        return annif.util.merge_hits(hits_from_sources, self.project.subjects)

    def _suggest(self, text, params):
        sources = annif.util.parse_sources(params['sources'])
        hits_from_sources = self._suggest_with_sources(text, sources)
        merged_hits = self._merge_hits_from_sources(hits_from_sources, params)
        self.debug('{} hits after merging'.format(len(merged_hits)))
        return merged_hits


class EnsembleOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the ensemble backend"""

    def __init__(self, backend, corpus, metric):
        super().__init__(backend, corpus, metric)
        self._sources = [project_id for project_id, _
                         in annif.util.parse_sources(
                             backend.config_params['sources'])]

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
            limit=int(self._backend.params['limit']),
            threshold=0.0)

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        with pool_class(jobs) as pool:
            for hits, uris, labels in pool.imap_unordered(
                    psmap.suggest, self._corpus.documents):
                self._gold_subjects.append(
                    annif.corpus.SubjectSet((uris, labels)))
                self._source_hits.append(hits)

    def _normalize(self, hps):
        total = sum(hps.values())
        return {source: hps[source] / total for source in hps}

    def _format_cfg_line(self, hps):
        return 'sources=' + ','.join([f"{src}:{weight:.4f}"
                                      for src, weight in hps.items()])

    def _objective(self, trial):
        batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        weights = {project_id: trial.suggest_uniform(project_id, 0.0, 1.0)
                   for project_id in self._sources}
        for goldsubj, srchits in zip(self._gold_subjects, self._source_hits):
            weighted_hits = []
            for project_id, hits in srchits.items():
                weighted_hits.append(annif.suggestion.WeightedSuggestion(
                    hits=hits,
                    weight=weights[project_id],
                    subjects=self._backend.project.subjects))
            batch.evaluate(
                annif.util.merge_hits(
                    weighted_hits,
                    self._backend.project.subjects),
                goldsubj)
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
        sources_trained = self._get_sources_attribute('is_trained')
        return all(sources_trained)

    @property
    def modification_time(self):
        mtimes = self._get_sources_attribute('modification_time')
        return max(filter(None, mtimes), default=None)

    def get_hp_optimizer(self, corpus, metric):
        return EnsembleOptimizer(self, corpus, metric)

    def _train(self, corpus, params, jobs=0):
        raise NotSupportedException(
            'Training ensemble backend is not possible.')
