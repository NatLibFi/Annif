"""Ensemble backend that combines results from multiple projects"""


from hyperopt import hp
import annif.suggestion
import annif.project
import annif.util
import annif.eval
from . import hyperopt
from annif.exception import NotSupportedException


class EnsembleOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the ensemble backend"""

    def __init__(self, backend, corpus):
        super().__init__(backend, corpus)
        self._sources = [project_id for project_id, _
                         in annif.util.parse_sources(
                             backend.config_params['sources'])]

    def get_hp_space(self):
        space = {}
        for project_id in self._sources:
            space[project_id] = hp.uniform(project_id, 0.0, 1.0)
        return space

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

    def _test(self, hps):
        batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        for goldsubj, srchits in zip(self._gold_subjects, self._source_hits):
            weighted_hits = []
            for project_id, hits in srchits.items():
                weighted_hits.append(annif.suggestion.WeightedSuggestion(
                    hits=hits, weight=hps[project_id]))
            batch.evaluate(
                annif.util.merge_hits(
                    weighted_hits,
                    self._backend.project.subjects),
                goldsubj)
        results = batch.results()
        return 1 - results['NDCG']

    def _postprocess(self, best, trials):
        total = sum(best.values())
        scaled = {source: best[source] / total for source in best}
        lines = 'sources=' + ','.join([f"{src}:{weight:.4f}"
                                       for src, weight in scaled.items()])
        score = 1 - trials.best_trial['result']['loss']
        return hyperopt.HPRecommendation(lines=[lines], score=score)


class EnsembleBackend(hyperopt.AnnifHyperoptBackend):
    """Ensemble backend that combines results from multiple projects"""
    name = "ensemble"

    def get_hp_optimizer(self, corpus):
        return EnsembleOptimizer(self, corpus)

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
