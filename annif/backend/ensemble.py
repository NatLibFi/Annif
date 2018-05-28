"""Ensemble backend that combines results from multiple projects"""


import annif.hit
import annif.project
import annif.util
from . import backend


def parse_sources(sources_string):
    sources = []
    for srcdef in sources_string.strip().split(','):
        srcval = srcdef.strip().split(':')
        src_id = srcval[0]
        if len(srcval) > 1:
            weight = float(srcval[1])
        else:
            weight = 1.0
        project = annif.project.get_project(src_id)
        sources.append((project, weight))
    return sources


class EnsembleBackend(backend.AnnifBackend):
    name = "ensemble"

    def _analyze_with_sources(self, text, sources):
        hits_from_sources = []
        for project, weight in sources:
            hits = [hit for hit in project.analyze(text) if hit.score > 0.0]
            self.debug(
                'Got {} hits from project {}'.format(
                    len(hits), project.project_id))
            hits_from_sources.append(
                annif.hit.WeightedHits(
                    hits=hits, weight=weight))
        return hits_from_sources

    def _analyze(self, text, project, params):
        sources = parse_sources(params['sources'])
        hits_from_sources = self._analyze_with_sources(text, sources)
        merged_hits = annif.util.merge_hits(hits_from_sources)
        self.debug('{} hits after merging'.format(len(merged_hits)))
        return merged_hits
