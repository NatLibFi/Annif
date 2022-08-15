"""Hyperparameter optimization functionality for backends"""

import abc
import collections
import warnings
import optuna
import optuna.exceptions
from .backend import AnnifBackend


HPRecommendation = collections.namedtuple('HPRecommendation', 'lines score')


class TrialWriter:
    """Object that writes hyperparameter optimization trial results into a
    TSV file."""

    def __init__(self, results_file, normalize_func):
        self.results_file = results_file
        self.normalize_func = normalize_func
        self.header_written = False

    def write(self, study, trial):
        """Write the results of one trial into the results file.  On the
        first run, write the header line first."""

        if not self.header_written:
            param_names = list(trial.params.keys())
            print('\t'.join(['trial', 'value'] + param_names),
                  file=self.results_file)
            self.header_written = True
        print('\t'.join((str(e) for e in [trial.number, trial.value] +
                         list(self.normalize_func(trial.params).values()))),
              file=self.results_file)


class HyperparameterOptimizer:
    """Base class for hyperparameter optimizers"""

    def __init__(self, backend, corpus, metric):
        self._backend = backend
        self._corpus = corpus
        self._metric = metric

    def _prepare(self, n_jobs=1):
        """Prepare the optimizer for hyperparameter evaluation.  Up to
        n_jobs parallel threads or processes may be used during the
        operation."""

        pass  # pragma: no cover

    @abc.abstractmethod
    def _objective(self, trial):
        """Objective function to optimize"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _postprocess(self, study):
        """Convert the study results into hyperparameter recommendations"""
        pass  # pragma: no cover

    def _normalize(self, hps):
        """Normalize the given raw hyperparameters. Intended to be overridden
        by subclasses when necessary. The default is to keep them as-is."""
        return hps

    def optimize(self, n_trials, n_jobs, results_file):
        """Find the optimal hyperparameters by testing up to the given number
        of hyperparameter combinations"""

        self._prepare(n_jobs)

        if results_file:
            callbacks = [TrialWriter(results_file, self._normalize).write]
        else:
            callbacks = []

        study = optuna.create_study(direction='maximize')
        # silence the ExperimentalWarning when using the Optuna progress bar
        warnings.filterwarnings("ignore",
                                category=optuna.exceptions.ExperimentalWarning)
        study.optimize(self._objective,
                       n_trials=n_trials,
                       n_jobs=n_jobs,
                       callbacks=callbacks,
                       gc_after_trial=False,
                       show_progress_bar=(n_jobs == 1))
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
