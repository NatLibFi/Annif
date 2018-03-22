"""Common functionality for backends."""

import abc
import os
import os.path
import annif
from annif import logger


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None

    def __init__(self, backend_id, params):
        """Initialize backend with specific parameters. The
        parameters are a dict. Keys and values depend on the specific
        backend type."""
        self.backend_id = backend_id
        self.params = params
        self._datadir = None

    def _get_datadir(self):
        """return the path of the directory where this backend can store its
        data files"""
        if self._datadir is None:
            self._datadir = os.path.join(
                annif.cxapp.app.config['DATADIR'],
                'backends',
                self.backend_id)
            if not os.path.exists(self._datadir):
                os.makedirs(self._datadir)
        return self._datadir

    def load_subjects(self, subjects, analyzer):
        """load the given subjects analyzed using the given analyzer"""
        pass  # default is to do nothing, subclasses may override

    @abc.abstractmethod
    def analyze(self, text):
        """Analyze some input text and return a list of subjects represented
        as a list of AnalysisHit objects."""
        pass

    def debug(self, message):
        """Log a debug message from this backend"""
        logger.debug("Backend {}: {}".format(self.backend_id, message))

    def info(self, message):
        """Log an info message from this backend"""
        logger.info("Backend {}: {}".format(self.backend_id, message))

    def warning(self, message):
        """Log a warning message from this backend"""
        logger.warning("Backend {}: {}".format(self.backend_id, message))
