"""Hyperparameter optimization functionality for backends"""

import abc
import collections
import warnings
import optuna
import optuna.exceptions
from .backend import AnnifBackend
from annif import logger


HPRecommendation = collections.namedtuple('HPRecommendation', 'lines score')


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

    def _write_trials_header(self, results_file, param_names):
        print('\t'.join(['trial', 'value'] + param_names), file=results_file)

    def _write_trial(self, results_file, trial):
        print('\t'.join((str(e) for e in [trial.number, trial.value] +
                         list(trial.params.values()))),
              file=results_file)

    def optimize(self, n_trials, n_jobs, results_file):
        """Find the optimal hyperparameters by testing up to the given number
        of hyperparameter combinations"""

        self._prepare(n_jobs)
        study = optuna.create_study(direction='maximize')
        # silence the ExperimentalWarning when using the Optuna progress bar
        warnings.filterwarnings("ignore",
                                category=optuna.exceptions.ExperimentalWarning)
        study.optimize(self._objective,
                       n_trials=n_trials,
                       n_jobs=n_jobs,
                       gc_after_trial=False,
                       show_progress_bar=(n_jobs == 1))
        if results_file:
            self._write_trials_header(results_file,
                                      list(study.best_params.keys()))
            for trial in study.trials:
                self._write_trial(results_file, trial)
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
