"""Registry that keeps track of Annif projects"""
from __future__ import annotations

import re

from flask import Flask, current_app

import annif
from annif.config import parse_config
from annif.exception import ConfigurationException
from annif.project import Access, AnnifProject
from annif.util import parse_args
from annif.vocab import AnnifVocabulary

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

    def __init__(
        self, projects_config_path: str, datadir: str, init_projects: bool
    ) -> None:
        self._rid = id(self)
        self._projects_config_path = projects_config_path
        self._datadir = datadir
        self._init_vars()
        if init_projects:
            for project in self._projects[self._rid].values():
                project.initialize()

    def _init_vars(self) -> None:
        # initialize the static variables, if necessary
        if self._rid not in self._projects:
            self._projects[self._rid] = self._create_projects()
            self._vocabs[self._rid] = {}

    def _create_projects(self) -> dict:
        # parse the configuration
        config = parse_config(self._projects_config_path)

        # handle the case where the config file doesn't exist
        if config is None:
            return {}

        # create AnnifProject objects from the configuration file
        projects = dict()
        for project_id in config.project_ids:
            projects[project_id] = AnnifProject(
                project_id, config[project_id], self._datadir, self
            )
        return projects

    def get_projects(
        self, min_access: Access = Access.private
    ) -> dict[str, AnnifProject]:
        """Return the available projects as a dict of project_id ->
        AnnifProject. The min_access parameter may be used to set the minimum
        access level required for the returned projects."""

        self._init_vars()
        return {
            project_id: project
            for project_id, project in self._projects[self._rid].items()
            if project.access >= min_access
        }

    def get_project(
        self, project_id: str, min_access: Access = Access.private
    ) -> AnnifProject:
        """return the definition of a single Project by project_id"""

        projects = self.get_projects(min_access)
        try:
            return projects[project_id]
        except KeyError:
            raise ValueError("No such project {}".format(project_id))

    def get_vocab(
        self, vocab_spec: str, default_language: str | None
    ) -> tuple[AnnifVocabulary, None] | tuple[AnnifVocabulary, str]:
        """Return an (AnnifVocabulary, language) pair corresponding to the
        vocab_spec. If no language information is specified, use the given
        default language."""

        match = re.match(r"([\w-]+)(\((.*)\))?$", vocab_spec)
        if match is None:
            raise ValueError(f"Invalid vocabulary specification: {vocab_spec}")
        vocab_id = match.group(1)
        posargs, kwargs = parse_args(match.group(3))
        language = posargs[0] if posargs else default_language
        vocab_key = (vocab_id, language)

        self._init_vars()
        if vocab_key not in self._vocabs[self._rid]:
            self._vocabs[self._rid][vocab_key] = AnnifVocabulary(
                vocab_id, self._datadir
            )
        return self._vocabs[self._rid][vocab_key], language


def initialize_projects(app: Flask) -> None:
    projects_config_path = app.config["PROJECTS_CONFIG_PATH"]
    datadir = app.config["DATADIR"]
    init_projects = app.config["INITIALIZE_PROJECTS"]
    app.annif_registry = AnnifRegistry(projects_config_path, datadir, init_projects)


def get_projects(min_access: Access = Access.private) -> dict[str, AnnifProject]:
    """Return the available projects as a dict of project_id ->
    AnnifProject. The min_access parameter may be used to set the minimum
    access level required for the returned projects."""
    if not hasattr(current_app, "annif_registry"):
        initialize_projects(current_app)

    return current_app.annif_registry.get_projects(min_access)


def get_project(project_id: str, min_access: Access = Access.private) -> AnnifProject:
    """return the definition of a single Project by project_id"""

    projects = get_projects(min_access)
    try:
        return projects[project_id]
    except KeyError:
        raise ValueError(f"No such project '{project_id}'")


def get_vocabs(min_access: Access = Access.private) -> dict[str, AnnifVocabulary]:
    """Return the available vocabularies as a dict of vocab_id ->
    AnnifVocabulary. The min_access parameter may be used to set the minimum
    access level required for the returned vocabularies."""

    vocabs = {}
    for proj in get_projects(min_access).values():
        try:
            vocabs[proj.vocab.vocab_id] = proj.vocab
        except ConfigurationException:
            pass

    return vocabs


def get_vocab(vocab_id: str, min_access: Access = Access.private) -> AnnifVocabulary:
    """return a single AnnifVocabulary by vocabulary id"""

    vocabs = get_vocabs(min_access)
    try:
        return vocabs[vocab_id]
    except KeyError:
        raise ValueError(f"No such vocabulary '{vocab_id}'")
