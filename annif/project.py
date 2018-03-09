"""Project management functionality for Annif"""

import collections
import configparser
import annif
import annif.hit
import annif.backend


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    def __init__(self, project_id, config):
        self.project_id = project_id
        self.language = config['language']
        self.analyzer = config['analyzer']
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

    def analyze(self, text, limit=10, threshold=0.0):
        """Analyze the given text by passing it to backends and joining the
        results. Returns a list of AnalysisHit objects ordered by decreasing
        score. The limit parameter defines the maximum number of hits to return.
        Only hits whose score is over the threshold are returned."""

        hits_by_uri = collections.defaultdict(list)
        for backend, weight in self.backends:
            hits = backend.analyze(text)
            for hit in hits:
                hits_by_uri[hit.uri].append((hit.score * weight, hit))

        merged_hits = []
        for score_hits in hits_by_uri.values():
            total = sum([sh[0] for sh in score_hits])
            hit = annif.hit.AnalysisHit(
                score_hits[0].uri, score_hits[0].label, total)
            merged_hits.append(hit)

        merged_hits.sort(key=lambda hit: hit.score, reverse=True)
        merged_hits = merged_hits[:limit]
        return [hit for hit in merged_hits if hit.score > threshold]


def get_projects():
    """return the available projects as a dict of project_id -> AnnifProject"""
    projects_file = annif.cxapp.app.config['PROJECTS_FILE']
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
