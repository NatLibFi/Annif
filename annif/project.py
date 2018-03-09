"""Project management functionality for Annif"""

import configparser
import annif


class AnnifProject:
    """Class representing the configuration of a single Annif project."""

    def __init__(self, project_id, language, analyzer):
        self.project_id = project_id
        self.language = language
        self.analyzer = analyzer


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
            analyzer=config[project_id]['analyzer'])
    return projects


def get_project(project_id):
    """return the definition of a single Project by project_id"""
    projects = get_projects()
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
