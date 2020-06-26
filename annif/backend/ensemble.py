"""Ensemble backend that combines results from multiple projects"""


import annif.suggestion
import annif.util
from . import backend
from annif.exception import NotSupportedException


class EnsembleBackend(backend.AnnifBackend):
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

    def _get_sources_attribute(self, attr):
        params = self._get_backend_params(None)
        sources = annif.util.parse_sources(params['sources'])
        return [getattr(self.project.registry.get_project(project_id), attr)
                for project_id, _ in sources]

    def initialize(self):
        # initialize all the source projects
        params = self._get_backend_params(None)
        for project_id, _ in annif.util.parse_sources(params['sources']):
            project = self.project.registry.get_project(project_id)
            project.initialize()

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
                'Got {} hits from project {}'.format(
                    len(hits), source_project.project_id))
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

    def _train(self, corpus, params):
        raise NotSupportedException('Training ensemble model is not possible.')
