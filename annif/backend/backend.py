"""Common functionality for backends."""

import abc


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None

    def __init__(self, backend_id, config):
        """Initialize backend with a specific configuration. The
        configuration is a dict. Keys and values depend on the specific
        backend type."""
        self.backend_id = backend_id
        self.config = config

    @abc.abstractmethod
    def analyze(self, text):
        """Analyze some input text and return a list of subjects represented
        as a list of AnalysisHit objects."""
        pass
