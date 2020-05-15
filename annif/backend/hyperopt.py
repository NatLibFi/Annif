"""Hyperparameter optimization functionality for backends"""

import abc
import hyperopt
from .backend import AnnifBackend


class HyperparameterOptimizer:
    """Base class for hyperparameter optimizers"""

    def __init__(self, backend, corpus):
        self._backend = backend
        self._corpus = corpus

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

    def optimize(self, n_trials):
        """Find the optimal hyperparameters by testing up to the given number of
        hyperparameter combinations"""

        self._prepare()
        space = self.get_hp_space()
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
            show_progressbar=False,
            fn=self._test,
            space=space,
            algo=hyperopt.tpe.suggest,
            max_evals=n_trials,
            trials=trials)
        return (best, 1 - trials.best_trial['result']['loss'])


class AnnifHyperoptBackend(AnnifBackend):
    """Base class for Annif backends that can perform hyperparameter
    optimization"""

    @abc.abstractmethod
    def get_hp_optimizer(self, corpus):
        """Get a HyperparameterOptimizer object that can look for
        optimal hyperparameter combinations for the given corpus"""

        pass  # pragma: no cover
