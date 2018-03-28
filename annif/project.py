"""Project management functionality for Annif"""

import collections
import configparser
import logging
from flask import current_app
import annif
import annif.hit
import annif.backend
from annif import logger


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    def __init__(self, project_id, config):
        self.project_id = project_id
        self.language = config['language']
        self.backends = self._initialize_backends(config['backends'])

    def _initialize_backends(self, backends_configuration):
        backends = []
        for backenddef in backends_configuration.split(','):
            bedefs = backenddef.strip().split(':')
            backend_id = bedefs[0]
            if len(bedefs) > 1:
                weight = float(bedefs[1])
            else:
                weight = 1.0
            backend = annif.backend.get_backend(backend_id)
            backends.append((backend, weight))
        return backends

    def _analyze_with_backends(self, text, backend_params):
        hits_by_uri = collections.defaultdict(list)
        for backend, weight in self.backends:
            beparams = backend_params.get(backend.backend_id, {})
            hits = [hit for hit in backend.analyze(text, params=beparams)
                    if hit.score > 0.0]
            logger.debug(
                'Got {} hits from backend {}'.format(
                    len(hits), backend.backend_id))
            for hit in hits:
                hits_by_uri[hit.uri].append((hit.score * weight, hit))
        return hits_by_uri

    def _merge_hits(self, hits_by_uri):
        merged_hits = []
        for score_hits in hits_by_uri.values():
            total = sum([sh[0] for sh in score_hits])
            hit = annif.hit.AnalysisHit(
                score_hits[0][1].uri, score_hits[0][1].label, total)
            merged_hits.append(hit)
        return merged_hits

    def analyze(self, text, limit=10, threshold=0.0, backend_params={}):
        """Analyze the given text by passing it to backends and joining the
        results. Returns a list of AnalysisHit objects ordered by decreasing
        score. The limit parameter defines the maximum number of hits to return.
        Only hits whose score is over the threshold are returned."""

        logger.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        hits_by_uri = self._analyze_with_backends(text, backend_params)
        merged_hits = self._merge_hits(hits_by_uri)

        logger.debug('{} hits after merging'.format(len(merged_hits)))
        merged_hits.sort(key=lambda hit: hit.score, reverse=True)
        merged_hits = merged_hits[:limit]
        logger.debug(
            '{} hits after applying limit {}'.format(
                len(merged_hits), limit))
        merged_hits = [hit for hit in merged_hits if hit.score >= threshold]
        logger.debug(
            '{} hits after applying threshold {}'.format(
                len(merged_hits), threshold))
        return merged_hits

    def load_subjects(self, subjects):
        for backend, weight in self.backends:
            logger.debug(
                'Loading subjects for backend {}'.format(
                    backend.backend_id))
            backend.load_subjects(subjects)

    def dump(self):
        """return this project as a dict"""
        return {'project_id': self.project_id,
                'language': self.language,
                'backends': [{'backend_id': be[0].backend_id,
                              'weight': be[1]} for be in self.backends]
                }


def get_projects():
    """return the available projects as a dict of project_id -> AnnifProject"""
    projects_file = current_app.config['PROJECTS_FILE']
    config = configparser.ConfigParser()
    with open(projects_file) as projf:
        config.read_file(projf)

    # create AnnifProject objects from the configuration file
    projects = {}
    for project_id in config.sections():
        projects[project_id] = AnnifProject(project_id, config[project_id])
    return projects


def get_project(project_id):
    """return the definition of a single Project by project_id"""
    projects = get_projects()
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
