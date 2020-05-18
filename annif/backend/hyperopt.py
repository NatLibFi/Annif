"""Hyperparameter optimization functionality for backends"""

import abc
import collections
import hyperopt
from .backend import AnnifBackend
from annif import logger
from logging import DEBUG


HPRecommendation = collections.namedtuple('HPRecommendation', 'lines score')


class HyperparameterOptimizer:
    """Base class for hyperparameter optimizers"""

    def __init__(self, backend, corpus, metric):
        self._backend = backend
        self._corpus = corpus
        self._metric = metric

    @abc.abstractmethod
    def get_hp_space(self):
        """Get the hyperparameter space definition of this backend"""
        pass  # pragma: no cover

    def _prepare(self):
        """Prepare the optimizer for hyperparameter evaluation"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _test(self, hps):
        """Evaluate a set of hyperparameters"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _postprocess(self, best, trials):
        """Convert the trial results into hyperparameter recommendations"""
        pass  # pragma: no cover

    def optimize(self, n_trials):
        """Find the optimal hyperparameters by testing up to the given number
        of hyperparameter combinations"""

        self._prepare()
        space = self.get_hp_space()
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
            verbose=not logger.isEnabledFor(DEBUG),
            fn=self._test,
            space=space,
            algo=hyperopt.tpe.suggest,
            max_evals=n_trials,
            trials=trials)
        return self._postprocess(best, trials)


class AnnifHyperoptBackend(AnnifBackend):
    """Base class for Annif backends that can perform hyperparameter
    optimization"""

    @abc.abstractmethod
    def get_hp_optimizer(self, corpus):
        """Get a HyperparameterOptimizer object that can look for
        optimal hyperparameter combinations for the given corpus"""

        pass  # pragma: no cover
