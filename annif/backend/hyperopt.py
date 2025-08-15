"""Hyperparameter optimization functionality for backends"""

from __future__ import annotations

import abc
import collections
import tempfile
from typing import TYPE_CHECKING, Any, Callable

import optuna
import optuna.exceptions

import annif.parallel

from .backend import AnnifBackend

if TYPE_CHECKING:
    from click.utils import LazyFile
    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.corpus.document import DocumentCorpus

HPRecommendation = collections.namedtuple("HPRecommendation", "lines score")


class TrialWriter:
    """Object that writes hyperparameter optimization trial results into a
    TSV file."""

    def __init__(self, results_file: LazyFile, normalize_func: Callable) -> None:
        self.results_file = results_file
        self.normalize_func = normalize_func
        self.header_written = False

    def write(self, trial_data: dict[str, Any]) -> None:
        """Write the results of one trial into the results file.  On the
        first run, write the header line first."""

        if not self.header_written:
            param_names = list(trial_data["params"].keys())
            print("\t".join(["trial", "value"] + param_names), file=self.results_file)
            self.header_written = True
        print(
            "\t".join(
                (
                    str(e)
                    for e in [trial_data["number"], trial_data["value"]]
                    + list(self.normalize_func(trial_data["params"]).values())
                )
            ),
            file=self.results_file,
        )


class HyperparameterOptimizer:
    """Base class for hyperparameter optimizers"""

    def __init__(
        self, backend: AnnifBackend, corpus: DocumentCorpus, metric: str
    ) -> None:
        self._backend = backend
        self._corpus = corpus
        self._metric = metric

    def _prepare(self, n_jobs: int = 1):
        """Prepare the optimizer for hyperparameter evaluation.  Up to
        n_jobs parallel threads or processes may be used during the
        operation."""

        pass  # pragma: no cover

    @abc.abstractmethod
    def _objective(self, trial: Trial) -> float:
        """Objective function to optimize"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _postprocess(self, study: Study) -> HPRecommendation:
        """Convert the study results into hyperparameter recommendations"""
        pass  # pragma: no cover

    def _normalize(self, hps: dict[str, float]) -> dict[str, float]:
        """Normalize the given raw hyperparameters. Intended to be overridden
        by subclasses when necessary. The default is to keep them as-is."""
        return hps

    def _run_trial(
        self, trial_id: int, storage_url: str, study_name: str
    ) -> dict[str, Any]:

        # use a callback to set the completed trial, to avoid race conditions
        completed_trial = []

        def set_trial_callback(study: Study, trial: Trial) -> None:
            completed_trial.append(trial)

        study = optuna.load_study(storage=storage_url, study_name=study_name)
        study.optimize(
            self._objective,
            n_trials=1,
            callbacks=[set_trial_callback],
        )

        return {
            "number": completed_trial[0].number,
            "value": completed_trial[0].value,
            "params": completed_trial[0].params,
        }

    def optimize(
        self, n_trials: int, n_jobs: int, results_file: LazyFile | None
    ) -> HPRecommendation:
        """Find the optimal hyperparameters by testing up to the given number
        of hyperparameter combinations"""

        self._prepare(n_jobs)

        writer = TrialWriter(results_file, self._normalize) if results_file else None
        write_callback = writer.write if writer else None

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        storage_url = f"sqlite:///{temp_db.name}"

        study = optuna.create_study(direction="maximize", storage=storage_url)

        jobs, pool_class = annif.parallel.get_pool(n_jobs)
        with pool_class(jobs) as pool:
            for i in range(n_trials):
                pool.apply_async(
                    self._run_trial,
                    args=(i, storage_url, study.study_name),
                    callback=write_callback,
                )
            pool.close()
            pool.join()

        return self._postprocess(study)


class AnnifHyperoptBackend(AnnifBackend):
    """Base class for Annif backends that can perform hyperparameter
    optimization"""

    @abc.abstractmethod
    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str):
        """Get a HyperparameterOptimizer object that can look for
        optimal hyperparameter combinations for the given corpus,
        measured using the given metric"""

        pass  # pragma: no cover
