"""Project management functionality for Annif"""

import configparser
import annif
import annif.backend


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    def __init__(self, project_id, language, analyzer, backends):
        self.project_id = project_id
        self.language = language
        self.analyzer = analyzer
        self.backends = self._initialize_backends(backends)

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


def get_projects():
    """return the available projects as a dict of project_id -> AnnifProject"""
    projects_file = annif.cxapp.app.config['PROJECTS_FILE']
    config = configparser.ConfigParser()
    with open(projects_file) as projf:
        config.read_file(projf)

    # create AnnifProject objects from the configuration file
    projects = {}
    for project_id in config.sections():
        projects[project_id] = AnnifProject(
            project_id,
            language=config[project_id]['language'],
            analyzer=config[project_id]['analyzer'],
            backends=config[project_id]['backends'])
    return projects


def get_project(project_id):
    """return the definition of a single Project by project_id"""
    projects = get_projects()
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
