"""Ensemble backend that combines results from multiple projects"""


import collections
import annif.project
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
        hits_by_uri = collections.defaultdict(list)
        for project, weight in sources:
            hits = [hit for hit in project.analyze(text) if hit.score > 0.0]
            self.debug(
                'Got {} hits from project {}'.format(
                    len(hits), project.project_id))
            for hit in hits:
                hits_by_uri[hit.uri].append((hit.score * weight, hit))
        return hits_by_uri

    def _merge_hits(self, hits_by_uri, totalweight):
        merged_hits = []
        for score_hits in hits_by_uri.values():
            total = sum([sh[0] for sh in score_hits]) / totalweight
            hit = score_hits[0][1]._replace(score=total)
            merged_hits.append(hit)
        return merged_hits

    def _analyze(self, text, project, params):
        sources = parse_sources(params['sources'])
        totalweight = sum((src[1] for src in sources))
        hits_by_uri = self._analyze_with_sources(text, sources)
        merged_hits = self._merge_hits(hits_by_uri, totalweight)
        self.debug('{} hits after merging'.format(len(merged_hits)))
        return merged_hits
