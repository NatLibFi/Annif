"""Registry that keeps track of Annif projects"""

import collections
import re
from flask import current_app
import annif
from annif.config import parse_config
from annif.project import Access, AnnifProject
from annif.vocab import AnnifVocabulary
from annif.util import parse_args

logger = annif.logger


class AnnifRegistry:
    """Class that keeps track of the Annif projects and vocabularies"""

    # Note: The individual projects and vocabularies are stored in shared
    # static variables, keyed by the "registry ID" which is unique to the
    # registry instance. This is done to make it possible to serialize
    # AnnifRegistry instances without including the potentially huge objects
    # (which contain backends with large models, vocabularies with lots of
    # concepts etc). Serialized AnnifRegistry instances can then be passed
    # between processes when using the multiprocessing module.
    _projects = {}
    _vocabs = {}

    def __init__(self, projects_config_path, datadir, init_projects):
        self._rid = id(self)
        self._datadir = datadir
        self._projects[self._rid] = \
            self._create_projects(projects_config_path)
        self._vocabs[self._rid] = {}
        if init_projects:
            for project in self._projects[self._rid].values():
                project.initialize()

    def _create_projects(self, projects_config_path):
        # parse the configuration
        config = parse_config(projects_config_path)

        # handle the case where the config file doesn't exist
        if config is None:
            return {}

        # create AnnifProject objects from the configuration file
        projects = collections.OrderedDict()
        for project_id in config.project_ids:
            projects[project_id] = AnnifProject(project_id,
                                                config[project_id],
                                                self._datadir,
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

    def get_vocab(self, vocab_spec, default_language):
        """Return an (AnnifVocabulary, language) pair corresponding to the
        vocab_spec. If no language information is specified, use the given
        default language."""

        match = re.match(r'(\w+)(\((.*)\))?', vocab_spec)
        if match is None:
            raise ValueError(
                f"Invalid vocabulary specification: {vocab_spec}")
        vocab_id = match.group(1)
        posargs, kwargs = parse_args(match.group(3))
        language = posargs[0] if posargs else default_language
        vocab_key = (vocab_id, language)

        if vocab_key not in self._vocabs[self._rid]:
            self._vocabs[self._rid][vocab_key] = AnnifVocabulary(
                vocab_id, self._datadir)
        return self._vocabs[self._rid][vocab_key], language


def initialize_projects(app):
    projects_config_path = app.config['PROJECTS_CONFIG_PATH']
    datadir = app.config['DATADIR']
    init_projects = app.config['INITIALIZE_PROJECTS']
    app.annif_registry = AnnifRegistry(projects_config_path, datadir,
                                       init_projects)


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
