"""Ensemble backend that combines results from multiple projects"""


import annif.suggestion
import annif.project
import annif.util
import annif.eval
from . import hyperopt
from annif.exception import NotSupportedException


class EnsembleOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the ensemble backend"""

    def __init__(self, backend, corpus, metric):
        super().__init__(backend, corpus, metric)
        self._sources = [project_id for project_id, _
                         in annif.util.parse_sources(
                             backend.config_params['sources'])]

    def _prepare(self):
        self._gold_subjects = []
        self._source_hits = []

        for doc in self._corpus.documents:
            self._gold_subjects.append(
                annif.corpus.SubjectSet((doc.uris, doc.labels)))
            srchits = {}
            for project_id in self._sources:
                source_project = annif.project.get_project(project_id)
                hits = source_project.suggest(doc.text)
                srchits[project_id] = hits
            self._source_hits.append(srchits)

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
                    hits=hits, weight=weights[project_id]))
            batch.evaluate(
                annif.util.merge_hits(
                    weighted_hits,
                    self._backend.project.subjects),
                goldsubj)
        results = batch.results()
        return results[self._metric]

    def _postprocess(self, study):
        line = self._format_cfg_line(self._normalize(study.best_params))
        return hyperopt.HPRecommendation(lines=[line], score=study.best_value)


class EnsembleBackend(hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that combines results from multiple projects"""
    name = "ensemble"

    def get_hp_optimizer(self, corpus, metric):
        return EnsembleOptimizer(self, corpus, metric)

    def _normalize_hits(self, hits, source_project):
        """Hook for processing hits from backends. Intended to be overridden
        by subclasses."""
        return hits

    def _suggest_with_sources(self, text, sources):
        hits_from_sources = []
        for project_id, weight in sources:
            source_project = annif.project.get_project(project_id)
            hits = source_project.suggest(text)
            self.debug(
                'Got {} hits from project {}, weight {}'.format(
                    len(hits), source_project.project_id, weight))
            norm_hits = self._normalize_hits(hits, source_project)
            hits_from_sources.append(
                annif.suggestion.WeightedSuggestion(
                    hits=norm_hits, weight=weight))
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

    def _train(self, corpus, params):
        raise NotSupportedException('Training ensemble model is not possible.')
