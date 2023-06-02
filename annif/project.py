"""Project management functionality for Annif"""
from __future__ import annotations

import enum
import os.path
from shutil import rmtree
from typing import TYPE_CHECKING

import annif
import annif.analyzer
import annif.backend
import annif.corpus
import annif.transform
from annif.datadir import DatadirMixin
from annif.exception import (
    AnnifException,
    ConfigurationException,
    NotInitializedException,
    NotSupportedException,
)

if TYPE_CHECKING:
    from collections import defaultdict
    from configparser import SectionProxy
    from datetime import datetime

    from click.utils import LazyFile

    from annif.analyzer import Analyzer
    from annif.backend import AnnifBackend
    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus.document import DocumentCorpus
    from annif.corpus.subject import SubjectIndex
    from annif.registry import AnnifRegistry
    from annif.transform.transform import TransformChain
    from annif.vocab import AnnifVocabulary

logger = annif.logger


class Access(enum.IntEnum):
    """Enumeration of access levels for projects"""

    private = 1
    hidden = 2
    public = 3


class AnnifProject(DatadirMixin):
    """Class representing the configuration of a single Annif project."""

    # defaults for uninitialized instances
    _transform = None
    _analyzer = None
    _backend = None
    _vocab = None
    _vocab_lang = None
    initialized = False

    # default values for configuration settings
    DEFAULT_ACCESS = "public"

    def __init__(
        self,
        project_id: str,
        config: dict[str, str] | SectionProxy,
        datadir: str,
        registry: AnnifRegistry,
    ) -> None:
        DatadirMixin.__init__(self, datadir, "projects", project_id)
        self.project_id = project_id
        self.name = config.get("name", project_id)
        self.language = config["language"]
        self.analyzer_spec = config.get("analyzer", None)
        self.transform_spec = config.get("transform", "pass")
        self.vocab_spec = config.get("vocab", None)
        self.config = config
        self._base_datadir = datadir
        self.registry = registry
        self._init_access()

    def _init_access(self) -> None:
        access = self.config.get("access", self.DEFAULT_ACCESS)
        try:
            self.access = getattr(Access, access)
        except AttributeError:
            raise ConfigurationException(
                "'{}' is not a valid access setting".format(access),
                project_id=self.project_id,
            )

    def _initialize_analyzer(self) -> None:
        if not self.analyzer_spec:
            return  # not configured, so assume it's not needed
        analyzer = self.analyzer
        logger.debug(
            "Project '%s': initialized analyzer: %s", self.project_id, str(analyzer)
        )

    def _initialize_subjects(self) -> None:
        try:
            subjects = self.subjects
            logger.debug(
                "Project '%s': initialized subjects: %s", self.project_id, str(subjects)
            )
        except AnnifException as err:
            logger.warning(err.format_message())

    def _initialize_backend(self, parallel: bool) -> None:
        logger.debug("Project '%s': initializing backend", self.project_id)
        try:
            if not self.backend:
                logger.debug("Cannot initialize backend: does not exist")
                return
            self.backend.initialize(parallel)
        except AnnifException as err:
            logger.warning(err.format_message())

    def initialize(self, parallel: bool = False) -> None:
        """Initialize this project and its backend so that they are ready to
        be used. If parallel is True, expect that the project will be used
        for parallel processing."""

        if self.initialized:
            return

        logger.debug("Initializing project '%s'", self.project_id)

        self._initialize_analyzer()
        self._initialize_subjects()
        self._initialize_backend(parallel)

        self.initialized = True

    def _suggest_with_backend(
        self,
        texts: list[str],
        backend_params: defaultdict[str, dict] | None,
    ) -> annif.suggestion.SuggestionBatch:
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        return self.backend.suggest(texts, beparams)

    @property
    def analyzer(self) -> Analyzer:
        if self._analyzer is None:
            if self.analyzer_spec:
                self._analyzer = annif.analyzer.get_analyzer(self.analyzer_spec)
            else:
                raise ConfigurationException(
                    "analyzer setting is missing", project_id=self.project_id
                )
        return self._analyzer

    @property
    def transform(self) -> TransformChain:
        if self._transform is None:
            self._transform = annif.transform.get_transform(
                self.transform_spec, project=self
            )
        return self._transform

    @property
    def backend(self) -> AnnifBackend | None:
        if self._backend is None:
            if "backend" not in self.config:
                raise ConfigurationException(
                    "backend setting is missing", project_id=self.project_id
                )
            backend_id = self.config["backend"]
            try:
                backend_class = annif.backend.get_backend(backend_id)
                self._backend = backend_class(
                    backend_id, config_params=self.config, project=self
                )
            except ValueError:
                logger.warning(
                    "Could not create backend %s, "
                    "make sure you've installed optional dependencies",
                    backend_id,
                )
        return self._backend

    def _initialize_vocab(self) -> None:
        if self.vocab_spec is None:
            raise ConfigurationException(
                "vocab setting is missing", project_id=self.project_id
            )
        self._vocab, self._vocab_lang = self.registry.get_vocab(
            self.vocab_spec, self.language
        )

    @property
    def vocab(self) -> AnnifVocabulary:
        if self._vocab is None:
            self._initialize_vocab()
        return self._vocab

    @property
    def vocab_lang(self) -> str:
        if self._vocab_lang is None:
            self._initialize_vocab()
        return self._vocab_lang

    @property
    def subjects(self) -> SubjectIndex:
        return self.vocab.subjects

    def _get_info(self, key: str) -> bool | datetime | None:
        try:
            be = self.backend
            if be is not None:
                return getattr(be, key)
        except AnnifException as err:
            logger.warning(err.format_message())
            return None

    @property
    def is_trained(self) -> bool | None:
        return self._get_info("is_trained")

    @property
    def modification_time(self) -> datetime | None:
        return self._get_info("modification_time")

    def suggest_corpus(
        self,
        corpus: DocumentCorpus,
        backend_params: defaultdict[str, dict] | None = None,
    ) -> annif.suggestion.SuggestionResults:
        """Suggest subjects for the given documents corpus in batches of documents."""
        suggestions = (
            self.suggest([doc.text for doc in doc_batch], backend_params)
            for doc_batch in corpus.doc_batches
        )
        import annif.suggestion

        return annif.suggestion.SuggestionResults(suggestions)

    def suggest(
        self,
        texts: list[str],
        backend_params: defaultdict[str, dict] | None = None,
    ) -> annif.suggestion.SuggestionBatch:
        """Suggest subjects for the given documents batch."""
        if not self.is_trained:
            if self.is_trained is None:
                logger.warning("Could not get train state information.")
            else:
                raise NotInitializedException("Project is not trained.")
        texts = [self.transform.transform_text(text) for text in texts]
        return self._suggest_with_backend(texts, backend_params)

    def train(
        self,
        corpus: DocumentCorpus,
        backend_params: defaultdict[str, dict] | None = None,
        jobs: int = 0,
    ) -> None:
        """train the project using documents from a metadata source"""
        if corpus != "cached":
            corpus = self.transform.transform_corpus(corpus)
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        self.backend.train(corpus, beparams, jobs)

    def learn(
        self,
        corpus: DocumentCorpus,
        backend_params: defaultdict[str, dict] | None = None,
    ) -> None:
        """further train the project using documents from a metadata source"""
        if backend_params is None:
            backend_params = {}
        beparams = backend_params.get(self.backend.backend_id, {})
        corpus = self.transform.transform_corpus(corpus)
        if isinstance(self.backend, annif.backend.backend.AnnifLearningBackend):
            self.backend.learn(corpus, beparams)
        else:
            raise NotSupportedException(
                "Learning not supported by backend", project_id=self.project_id
            )

    def hyperopt(
        self,
        corpus: DocumentCorpus,
        trials: int,
        jobs: int,
        metric: str,
        results_file: LazyFile | None,
    ) -> HPRecommendation:
        """optimize the hyperparameters of the project using a validation
        corpus against a given metric"""
        if isinstance(self.backend, annif.backend.hyperopt.AnnifHyperoptBackend):
            optimizer = self.backend.get_hp_optimizer(corpus, metric)
            return optimizer.optimize(trials, jobs, results_file)

        raise NotSupportedException(
            "Hyperparameter optimization not supported " "by backend",
            project_id=self.project_id,
        )

    def dump(self) -> dict[str, str | dict | bool | datetime | None]:
        """return this project as a dict"""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "language": self.language,
            "backend": {"backend_id": self.config.get("backend")},
            "is_trained": self.is_trained,
            "modification_time": self.modification_time,
        }

    def remove_model_data(self) -> None:
        """remove the data of this project"""
        datadir_path = self._datadir_path
        if os.path.isdir(datadir_path):
            rmtree(datadir_path)
            logger.info("Removed model data for project {}.".format(self.project_id))
        else:
            logger.warning(
                "No model data to remove for project {}.".format(self.project_id)
            )
