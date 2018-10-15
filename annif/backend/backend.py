"""Common functionality for backends."""

import abc
import os
import os.path
from annif import logger


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None
    needs_subject_index = False
    needs_subject_vectorizer = False

    def __init__(self, backend_id, params, datadir):
        """Initialize backend with specific parameters. The
        parameters are a dict. Keys and values depend on the specific
        backend type."""
        self.backend_id = backend_id
        self.params = params
        self._datadir = datadir

    def _get_datadir(self):
        """return the path of the directory where this backend can store its
        data files"""
        if not os.path.exists(self._datadir):
            os.makedirs(self._datadir)
        return self._datadir

    def load_corpus(self, corpus, project):
        """load the given document or subject corpus"""
        pass  # default is to do nothing, subclasses may override

    def initialize(self):
        """This method can be overridden by backends. It should cause the
        backend to pre-load all data it needs during operation."""
        pass

    @abc.abstractmethod
    def _analyze(self, text, project, params):
        """This method should implemented by backends. It implements
        the analyze functionality, with pre-processed parameters."""
        pass

    def analyze(self, text, project, params=None):
        """Analyze some input text and return a list of subjects represented
        as a list of AnalysisHit objects."""
        beparams = dict(self.params)
        if params is not None:
            beparams.update(params)
        return self._analyze(text, project, params=beparams)

    def debug(self, message):
        """Log a debug message from this backend"""
        logger.debug("Backend {}: {}".format(self.backend_id, message))

    def info(self, message):
        """Log an info message from this backend"""
        logger.info("Backend {}: {}".format(self.backend_id, message))

    def warning(self, message):
        """Log a warning message from this backend"""
        logger.warning("Backend {}: {}".format(self.backend_id, message))
