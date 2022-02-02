"""Registry that keeps track of Annif projects"""

import collections
import configparser
import os.path
from flask import current_app
import tomli
import annif
import annif.util
from annif.exception import ConfigurationException
from annif.project import Access, AnnifProject

logger = annif.logger


class AnnifConfigCFG:
    def __init__(self, filename):
        self._config = configparser.ConfigParser()
        self._config.optionxform = annif.util.identity
        with open(filename, encoding='utf-8-sig') as projf:
            try:
                self._config.read_file(projf)
            except (configparser.DuplicateOptionError,
                    configparser.DuplicateSectionError) as err:
                raise ConfigurationException(err)

    @property
    def project_ids(self):
        return self._config.sections()

    def __getitem__(self, key):
        return self._config[key]


class AnnifConfigTOML:
    def __init__(self, filename):
        with open(filename, "rb") as projf:
            try:
                self._config = tomli.load(projf)
            except tomli.TOMLDecodeError as err:
                raise ConfigurationException(err)

    @property
    def project_ids(self):
        return self._config.keys()

    def __getitem__(self, key):
        return self._config[key]


class AnnifRegistry:
    """Class that keeps track of the Annif projects"""

    # Note: The individual projects are stored in a shared static variable,
    # keyed by the "registry ID" which is unique to the registry instance.
    # This is done to make it possible to serialize AnnifRegistry instances
    # without including the potentially huge project objects (which contain
    # backends with large models, vocabularies with lots of concepts etc).
    # Serialized AnnifRegistry instances can then be passed between
    # processes when using the multiprocessing module.
    _projects = {}

    def __init__(self, projects_file, datadir, init_projects):
        self._rid = id(self)
        self._projects[self._rid] = \
            self._create_projects(projects_file, datadir)
        if init_projects:
            for project in self._projects[self._rid].values():
                project.initialize()

    def _create_projects(self, projects_file, datadir):
        if projects_file:
            if not os.path.exists(projects_file):
                logger.warning(
                    f'Project configuration file "{projects_file}" is ' +
                    'missing. Please provide one. ' +
                    'You can set the path to the project configuration ' +
                    'file using the ANNIF_PROJECTS environment ' +
                    'variable or the command-line option "--projects".')
                return {}
        else:
            if os.path.exists('projects.cfg'):
                projects_file = 'projects.cfg'
            elif os.path.exists('projects.toml'):
                projects_file = 'projects.toml'
            else:
                logger.warning(
                    'Could not find project configuration file ' +
                    '"projects.cfg" or "projects.toml". ' +
                    'You can set the path to the project configuration ' +
                    'file using the ANNIF_PROJECTS environment ' +
                    'variable or the command-line option "--projects".')
                return {}

        if projects_file.endswith('.toml'):  # TOML format
            config = AnnifConfigTOML(projects_file)
        else:  # classic INI style format
            config = AnnifConfigCFG(projects_file)

        # create AnnifProject objects from the configuration file
        projects = collections.OrderedDict()
        for project_id in config.project_ids:
            projects[project_id] = AnnifProject(project_id,
                                                config[project_id],
                                                datadir,
                                                self)
        return projects

    def get_projects(self, min_access=Access.private):
        """Return the available projects as a dict of project_id ->
        AnnifProject. The min_access parameter may be used to set the minimum
        access level required for the returned projects."""

        return {project_id: project
                for project_id, project in self._projects[self._rid].items()
                if project.access >= min_access}

    def get_project(self, project_id, min_access=Access.private):
        """return the definition of a single Project by project_id"""

        projects = self.get_projects(min_access)
        try:
            return projects[project_id]
        except KeyError:
            raise ValueError("No such project {}".format(project_id))


def initialize_projects(app):
    projects_file = app.config['PROJECTS_FILE']
    datadir = app.config['DATADIR']
    init_projects = app.config['INITIALIZE_PROJECTS']
    app.annif_registry = AnnifRegistry(projects_file, datadir, init_projects)


def get_projects(min_access=Access.private):
    """Return the available projects as a dict of project_id ->
    AnnifProject. The min_access parameter may be used to set the minimum
    access level required for the returned projects."""
    if not hasattr(current_app, 'annif_registry'):
        initialize_projects(current_app)

    return current_app.annif_registry.get_projects(min_access)


def get_project(project_id, min_access=Access.private):
    """return the definition of a single Project by project_id"""

    projects = get_projects(min_access)
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError("No such project {}".format(project_id))
