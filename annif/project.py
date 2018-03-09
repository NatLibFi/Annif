import annif
import configparser

class AnnifProject:
    def __init__(self, project_id, language, analyzer):
        self.project_id = project_id
        self.language = language
        self.analyzer = analyzer


def get_projects():
    """return the available projects as a dict of project_id -> AnnifProject"""
    projects_file = annif.cxapp.app.config['PROJECTS_FILE']
    config = configparser.ConfigParser()
    with open(projects_file) as f:
        config.read_file(f)

    # create AnnifProject objects from the configuration file
    projects = {}
    for project_id in config.sections():
        projects[project_id] = AnnifProject(project_id,
                                            language=config[project_id]['language'],
                                            analyzer=config[project_id]['analyzer'])
    return projects
