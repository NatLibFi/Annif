"""Base class for Vowpal Wabbit based Annif backends"""

import os
from vowpalwabbit import pyvw
from annif.exception import ConfigurationException
from annif.exception import NotInitializedException
from . import backend


class VWBaseBackend(backend.AnnifLearningBackend):
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
        params.update({param: defaultval
                       for param, (_, defaultval) in self.VW_PARAMS.items()
                       if defaultval is not None})
        params.update({param: self._convert_param(param, val)
                       for param, val in self.params.items()
                       if param in self.VW_PARAMS})
        return params
