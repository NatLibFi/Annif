"""Ensemble backend that combines results from multiple projects"""


import annif.suggestion
import annif.project
import annif.util
from . import backend


class EnsembleBackend(backend.AnnifBackend):
    """Ensemble backend that combines results from multiple projects"""
    name = "ensemble"

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
                'Got {} hits from project {}'.format(
                    len(hits), source_project.project_id))
            norm_hits = self._normalize_hits(hits, source_project)
            hits_from_sources.append(
                annif.suggestion.WeightedSuggestion(
                    hits=norm_hits, weight=weight))
        return hits_from_sources

    def _merge_hits_from_sources(self, hits_from_sources, project, params):
        """Hook for merging hits from sources. Can be overridden by
        subclasses."""
        return annif.util.merge_hits(hits_from_sources, project.subjects)

    def _suggest(self, text, project, params):
        sources = annif.util.parse_sources(params['sources'])
        hits_from_sources = self._suggest_with_sources(text, sources)
        merged_hits = self._merge_hits_from_sources(hits_from_sources,
                                                    project,
                                                    params)
        self.debug('{} hits after merging'.format(len(merged_hits)))
        return merged_hits
