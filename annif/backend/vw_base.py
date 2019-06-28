"""Base class for Vowpal Wabbit based Annif backends"""

import abc
import os
from vowpalwabbit import pyvw
import annif.util
from annif.exception import ConfigurationException
from annif.exception import NotInitializedException
from . import backend


class VWBaseBackend(backend.AnnifLearningBackend, metaclass=abc.ABCMeta):
    """Base class for Vowpal Wabbit based Annif backends"""

    # Parameters for VW based backends
    # each param specifier is a pair (allowed_values, default_value)
    # where allowed_values is either a type or a list of allowed values
    # and default_value may be None, to let VW decide by itself
    VW_PARAMS = {}  # needs to be specified in subclasses

    MODEL_FILE = 'vw-model'
    TRAIN_FILE = 'vw-train.txt'

    # defaults for uninitialized instances
    _model = None

    def initialize(self):
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            if not os.path.exists(path):
                raise NotInitializedException(
                    'model {} not found'.format(path),
                    backend_id=self.backend_id)
            self.debug('loading VW model from {}'.format(path))
            params = self._create_params({'i': path, 'quiet': True})
            if 'passes' in params:
                # don't confuse the model with passes
                del params['passes']
            self.debug("model parameters: {}".format(params))
            self._model = pyvw.vw(**params)
            self.debug('loaded model {}'.format(str(self._model)))

    def _convert_param(self, param, val):
        pspec, _ = self.VW_PARAMS[param]
        if isinstance(pspec, list):
            if val in pspec:
                return val
            raise ConfigurationException(
                "{} is not a valid value for {} (allowed: {})".format(
                    val, param, ', '.join(pspec)), backend_id=self.backend_id)
        try:
            return pspec(val)
        except ValueError:
            raise ConfigurationException(
                "The {} value {} cannot be converted to {}".format(
                    param, val, pspec), backend_id=self.backend_id)

    def _create_params(self, params):
        params = params.copy()  # don't mutate the original dict
        params.update({param: defaultval
                       for param, (_, defaultval) in self.VW_PARAMS.items()
                       if defaultval is not None})
        params.update({param: self._convert_param(param, val)
                       for param, val in self.params.items()
                       if param in self.VW_PARAMS})
        return params

    @staticmethod
    def _write_train_file(examples, filename):
        with open(filename, 'w', encoding='utf-8') as trainfile:
            for ex in examples:
                print(ex, file=trainfile)

    def _create_train_file(self, corpus, project):
        self.info('creating VW train file')
        examples = self._create_examples(corpus, project)
        annif.util.atomic_save(examples,
                               self.datadir,
                               self.TRAIN_FILE,
                               method=self._write_train_file)

    @abc.abstractmethod
    def _create_examples(self, corpus, project):
        """This method should be implemented by concrete backends. It
        should return a sequence of strings formatted according to the VW
        input format."""
        pass  # pragma: no cover

    def _create_model(self, project, initial_params={}):
        initial_params = initial_params.copy()  # don't mutate the original
        trainpath = os.path.join(self.datadir, self.TRAIN_FILE)
        initial_params['data'] = trainpath
        params = self._create_params(initial_params)
        if params.get('passes', 1) > 1:
            # need a cache file when there are multiple passes
            params.update({'cache': True, 'kill_cache': True})
        self.debug("model parameters: {}".format(params))
        self._model = pyvw.vw(**params)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)

    def train(self, corpus, project):
        self.info("creating VW model")
        self._create_train_file(corpus, project)
        self._create_model(project)

    def learn(self, corpus, project):
        self.initialize()
        for example in self._create_examples(corpus, project):
            self._model.learn(example)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._model.save(modelpath)
