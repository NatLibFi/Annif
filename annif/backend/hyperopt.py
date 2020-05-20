"""Hyperparameter optimization functionality for backends"""

import abc
import collections
import optuna
from .backend import AnnifBackend
from annif import logger


HPRecommendation = collections.namedtuple('HPRecommendation', 'lines score')


class HyperparameterOptimizer:
    """Base class for hyperparameter optimizers"""

    def __init__(self, backend, corpus, metric):
        self._backend = backend
        self._corpus = corpus
        self._metric = metric

    def _prepare(self):
        """Prepare the optimizer for hyperparameter evaluation"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _objective(self, trial):
        """Objective function to optimize"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _postprocess(self, study):
        """Convert the study results into hyperparameter recommendations"""
        pass  # pragma: no cover

    def optimize(self, n_trials, n_jobs):
        """Find the optimal hyperparameters by testing up to the given number
        of hyperparameter combinations"""

        self._prepare()
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective,
                       n_trials=n_trials,
                       n_jobs=n_jobs,
                       gc_after_trial=False,
                       show_progress_bar=True)
        return self._postprocess(study)


class AnnifHyperoptBackend(AnnifBackend):
    """Base class for Annif backends that can perform hyperparameter
    optimization"""

    @abc.abstractmethod
    def get_hp_optimizer(self, corpus, metric):
        """Get a HyperparameterOptimizer object that can look for
        optimal hyperparameter combinations for the given corpus,
        measured using the given metric"""

        pass  # pragma: no cover
