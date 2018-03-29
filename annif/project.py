"""Project management functionality for Annif"""

import collections
import configparser
import logging
from flask import current_app
import annif
import annif.analyzer
import annif.hit
import annif.backend
from annif import logger


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    def __init__(self, project_id, config, all_backends):
        self.project_id = project_id
        self.language = config['language']
        self.analyzer_spec = config['analyzer']
        self.backends = self._initialize_backends(config['backends'],
                                                  all_backends)
        self._analyzer = None

    def _initialize_backends(self, backends_configuration, all_backends):
        backends = []
        for backenddef in backends_configuration.split(','):
            bedefs = backenddef.strip().split(':')
            backend_id = bedefs[0]
            if len(bedefs) > 1:
                weight = float(bedefs[1])
            else:
                weight = 1.0
            backend = all_backends[backend_id]
            backends.append((backend, weight))
        return backends

    def _analyze_with_backends(self, text, backend_params):
        if backend_params is None:
            backend_params = {}
        hits_by_uri = collections.defaultdict(list)
        for backend, weight in self.backends:
            beparams = backend_params.get(backend.backend_id, {})
            hits = [
                hit for hit in backend.analyze(
                    text,
                    project=self,
                    params=beparams) if hit.score > 0.0]
            logger.debug(
                'Got {} hits from backend {}'.format(
                    len(hits), backend.backend_id))
            for hit in hits:
                hits_by_uri[hit.uri].append((hit.score * weight, hit))
        return hits_by_uri

    @classmethod
    def _merge_hits(cls, hits_by_uri):
        merged_hits = []
        for score_hits in hits_by_uri.values():
            total = sum([sh[0] for sh in score_hits])
            hit = annif.hit.AnalysisHit(
                score_hits[0][1].uri, score_hits[0][1].label, total)
            merged_hits.append(hit)
        return merged_hits

    @classmethod
    def _filter_hits(cls, hits, limit, threshold):
        hits.sort(key=lambda hit: hit.score, reverse=True)
        hits = hits[:limit]
        logger.debug(
            '{} hits after applying limit {}'.format(
                len(hits), limit))
        hits = [hit for hit in hits if hit.score >= threshold]
        logger.debug(
            '{} hits after applying threshold {}'.format(
                len(hits), threshold))
        return hits

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = annif.analyzer.get_analyzer(self.analyzer_spec)
        return self._analyzer

    def analyze(self, text, limit=10, threshold=0.0, backend_params=None):
        """Analyze the given text by passing it to backends and joining the
        results. Returns a list of AnalysisHit objects ordered by decreasing
        score. The limit parameter defines the maximum number of hits to
        return. Only hits whose score is over the threshold are returned."""

        logger.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        hits_by_uri = self._analyze_with_backends(text, backend_params)
        merged_hits = self._merge_hits(hits_by_uri)
        logger.debug('{} hits after merging'.format(len(merged_hits)))
        return self._filter_hits(merged_hits, limit, threshold)

    def load_subjects(self, subjects):
        for backend, weight in self.backends:
            logger.debug(
                'Loading subjects for backend {}'.format(
                    backend.backend_id))
            backend.load_subjects(subjects, project=self)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'language': self.language,
                'backends': [{'backend_id': be[0].backend_id,
                              'weight': be[1]} for be in self.backends]
                }


def _create_projects(projects_file, backends):
    config = configparser.ConfigParser()
    with open(projects_file) as projf:
        config.read_file(projf)

    # create AnnifProject objects from the configuration file
    projects = {}
    for project_id in config.sections():
        projects[project_id] = AnnifProject(project_id,
                                            config[project_id], backends)
    return projects


def init_projects(app, backends):
    projects_file = app.config['PROJECTS_FILE']
    app.annif_projects = _create_projects(projects_file, backends)


def get_projects():
    """return the available projects as a dict of project_id -> AnnifProject"""
    return current_app.annif_projects


def get_project(project_id):
    """return the definition of a single Project by project_id"""
    projects = get_projects()
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
