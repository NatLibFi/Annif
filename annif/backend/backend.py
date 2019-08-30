"""Common functionality for backends."""

import abc
from annif import logger


class AnnifBackend(metaclass=abc.ABCMeta):
    """Base class for Annif backends that perform analysis. The
    non-implemented methods should be overridden in subclasses."""

    name = None
    needs_subject_index = False
    needs_subject_vectorizer = False

    DEFAULT_PARAMS = {
        'limit': 100,
    }

    def __init__(self, backend_id, params, datadir):
        """Initialize backend with specific parameters. The
        parameters are a dict. Keys and values depend on the specific
        backend type."""
        self.backend_id = backend_id
        self.params = params
        self.datadir = datadir
        self.fill_params_with_defaults()

    def fill_params_with_defaults(self):
        """Set the parameters that are not provided in the projects config file
        with default values defined in the backend class and its parents."""
        for source_cls in type(self).mro()[:-1]:  # omit the object class
            for default_param, default_value in \
                    source_cls.DEFAULT_PARAMS.items():
                if default_param not in self.params:
                    self.debug(
                        "parameter {} not set, using default value {} from {}"
                        .format(default_param, default_value,
                                source_cls.__name__))
                    self.params[default_param] = str(default_value)

    def train(self, corpus, project):
        """train the model on the given document or subject corpus"""
        pass  # default is to do nothing, subclasses may override

    def initialize(self):
        """This method can be overridden by backends. It should cause the
        backend to pre-load all data it needs during operation."""
        pass

    @abc.abstractmethod
    def _suggest(self, text, project, params):
        """This method should implemented by backends. It implements
        the suggest functionality, with pre-processed parameters."""
        pass  # pragma: no cover

    def suggest(self, text, project, params=None):
        """Suggest subjects for the input text and return a list of subjects
        represented as a list of SubjectSuggestion objects."""
        self.initialize()
        beparams = dict(self.params)
        if params:
            beparams.update(params)
        return self._suggest(text, project, params=beparams)

    def debug(self, message):
        """Log a debug message from this backend"""
        logger.debug("Backend {}: {}".format(self.backend_id, message))

    def info(self, message):
        """Log an info message from this backend"""
        logger.info("Backend {}: {}".format(self.backend_id, message))

    def warning(self, message):
        """Log a warning message from this backend"""
        logger.warning("Backend {}: {}".format(self.backend_id, message))


class AnnifLearningBackend(AnnifBackend):
    """Base class for Annif backends that can perform online learning"""

    @abc.abstractmethod
    def learn(self, corpus, project):
        """further train the model on the given document or subject corpus"""
        pass  # pragma: no cover
