"""Common functionality for backends."""

import abc


class BackendAnalysisHit:
    """A single hit from a backend resulting from analysis."""

    def __init__(self, uri, label, score):
        self.uri = uri
        self.label = label
        self.score = score


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None

    def __init__(self, config):
        """Initialize backend with a specific configuration. The
        configuration is a dict. Keys and values depend on the specific
        backend type."""
        self.config = config

    @abc.abstractmethod
    def analyze(self, text):
        """Analyze some input text and return a list of subjects represented
        as a list of BackendAnalysisHit objects."""
        pass
