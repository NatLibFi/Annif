"""Ensemble backend that combines results from multiple projects"""


import annif.hit
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

    def _analyze_with_sources(self, text, sources):
        hits_from_sources = []
        for project_id, weight in sources:
            source_project = annif.project.get_project(project_id)
            hits = source_project.analyze(text)
            self.debug(
                'Got {} hits from project {}'.format(
                    len(hits), source_project.project_id))
            norm_hits = self._normalize_hits(hits, source_project)
            hits_from_sources.append(
                annif.hit.WeightedHits(
                    hits=norm_hits, weight=weight))
        return hits_from_sources

    def _analyze(self, text, project, params):
        sources = annif.util.parse_sources(params['sources'])
        hits_from_sources = self._analyze_with_sources(text, sources)
        merged_hits = annif.util.merge_hits(
            hits_from_sources, project.subjects)
        self.debug('{} hits after merging'.format(len(merged_hits)))
        return merged_hits
