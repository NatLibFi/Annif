"""Common functionality for backends."""
from __future__ import annotations

import abc
import os.path
from datetime import datetime, timezone
from glob import glob
from typing import TYPE_CHECKING, Any

from annif import logger
from annif.suggestion import SuggestionBatch

if TYPE_CHECKING:
    from configparser import SectionProxy

    from annif.corpus.document import DocumentCorpus
    from annif.project import AnnifProject


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None

    DEFAULT_PARAMETERS = {"limit": 100}

    def __init__(
        self,
        backend_id: str,
        config_params: dict[str, Any] | SectionProxy,
        project: AnnifProject,
    ) -> None:
        """Initialize backend with specific parameters. The
        parameters are a dict. Keys and values depend on the specific
        backend type."""
        self.backend_id = backend_id
        self.config_params = config_params
        self.project = project
        self.datadir = project.datadir

    def default_params(self) -> dict[str, Any]:
        params = AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)  # Optional backend specific parameters
        return params

    @property
    def params(self) -> dict[str, Any]:
        params = {}
        params.update(self.default_params())
        params.update(self.config_params)
        return params

    @property
    def _model_file_paths(self) -> list:
        all_paths = glob(os.path.join(self.datadir, "*"))
        ignore_patterns = ("*-train*", "tmp-*", "vectorizer")
        ignore_paths = [
            path
            for igp in ignore_patterns
            for path in glob(os.path.join(self.datadir, igp))
        ]
        return list(set(all_paths) - set(ignore_paths))

    @property
    def is_trained(self) -> bool:
        return bool(self._model_file_paths)

    @property
    def modification_time(self) -> datetime | None:
        mtimes = [
            datetime.utcfromtimestamp(os.path.getmtime(p))
            for p in self._model_file_paths
        ]
        most_recent = max(mtimes, default=None)
        if most_recent is None:
            return None
        return most_recent.replace(tzinfo=timezone.utc)

    def _get_backend_params(
        self,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        backend_params = dict(self.params)
        if params is not None:
            backend_params.update(params)
        return backend_params

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        """This method can be overridden by backends. It implements
        the train functionality, with pre-processed parameters."""
        pass  # default is to do nothing, subclasses may override

    def train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any] | None = None,
        jobs: int = 0,
    ) -> None:
        """Train the model on the given document or subject corpus."""
        beparams = self._get_backend_params(params)
        return self._train(corpus, params=beparams, jobs=jobs)

    def initialize(self, parallel: bool = False) -> None:
        """This method can be overridden by backends. It should cause the
        backend to pre-load all data it needs during operation.
        If parallel is True, the backend should expect to be used for
        parallel operation."""
        pass

    def _suggest(self, text, params):
        """Either this method or _suggest_batch should be implemented by by
        backends.  It implements the suggest functionality for a single
        document, with pre-processed parameters."""
        pass  # pragma: no cover

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        """This method can be implemented by backends to use batching of documents in
        their operations. This default implementation uses the regular suggest
        functionality."""
        return SuggestionBatch.from_sequence(
            [self._suggest(text, params) for text in texts],
            self.project.subjects,
            limit=int(params.get("limit")),
        )

    def suggest(
        self,
        texts: list[str],
        params: dict[str, Any] | None = None,
    ) -> SuggestionBatch:
        """Suggest subjects for the input documents and return a list of subject sets
        represented as a list of SubjectSuggestion objects."""
        beparams = self._get_backend_params(params)
        self.initialize()
        return self._suggest_batch(texts, params=beparams)

    def debug(self, message: str) -> None:
        """Log a debug message from this backend"""
        logger.debug("Backend {}: {}".format(self.backend_id, message))

    def info(self, message: str) -> None:
        """Log an info message from this backend"""
        logger.info("Backend {}: {}".format(self.backend_id, message))

    def warning(self, message: str) -> None:
        """Log a warning message from this backend"""
        logger.warning("Backend {}: {}".format(self.backend_id, message))


class AnnifLearningBackend(AnnifBackend):
    """Base class for Annif backends that can perform online learning"""

    @abc.abstractmethod
    def _learn(self, corpus, params):
        """This method should implemented by backends. It implements the learn
        functionality, with pre-processed parameters."""
        pass  # pragma: no cover

    def learn(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Further train the model on the given document or subject corpus."""
        beparams = self._get_backend_params(params)
        return self._learn(corpus, params=beparams)
