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


class HPObjective(annif.parallel.BaseWorker):
    """Base class for hyperparameter optimizer objective functions"""

    @classmethod
    def objective(cls, trial: Trial, args) -> float:
        """Objective function to optimize. To be implemented by subclasses."""

        pass  # pragma: no cover

    @classmethod
    def _objective_wrapper(cls, trial: Trial) -> float:
        return cls.objective(trial, cls.args)

    @classmethod
    def run_trial(
        cls, trial_id: int, storage_url: str, study_name: str
    ) -> dict[str, Any]:

        # use a callback to set the completed trial, to avoid race conditions
        completed_trial = []

        def set_trial_callback(study: Study, trial: Trial) -> None:
            completed_trial.append(trial)

        study = optuna.load_study(storage=storage_url, study_name=study_name)
        study.optimize(
            cls._objective_wrapper,
            n_trials=1,
            callbacks=[set_trial_callback],
        )

        return {
            "number": completed_trial[0].number,
            "value": completed_trial[0].value,
            "params": completed_trial[0].params,
        }


class _FixedParamsTrial:
    """A minimal Trial stand-in for evaluating fixed hyperparameter
    combinations outside of Optuna's trial loop: the suggest_* methods
    return the given values and record the distributions so the trial
    can be registered in the study."""

    def __init__(self, params: dict[str, float]) -> None:
        self._params = params
        self.distributions: dict[str, Any] = {}

    def suggest_float(
        self,
        name: str,
        low: float | None = None,
        high: float | None = None,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        self.distributions[name] = optuna.distributions.FloatDistribution(
            low, high, step=step, log=log
        )
        return self._params[name]


class HyperparameterOptimizer:
    """Base class for hyperparameter optimizers"""

    def __init__(
        self,
        backend: AnnifBackend,
        corpus: DocumentCorpus,
        metric: str,
        objective: HPObjective,
    ) -> None:
        self._backend = backend
        self._corpus = corpus
        self._metric = metric
        self._objective = objective

    def _initial_trials(self) -> list[dict[str, float]]:
        """Return a list of hyperparameter combinations to evaluate first,
        before the sampler starts proposing its own. Intended to be
        overridden by subclasses when necessary. The default is to have no
        initial trials."""
        return []

    def _prepare(self, n_jobs: int = 1):
        """Prepare the optimizer for hyperparameter evaluation.  Up to
        n_jobs parallel threads or processes may be used during the
        operation. The return value will be passed to the objective function."""

        pass  # pragma: no cover

    @abc.abstractmethod
    def _postprocess(self, study: Study) -> HPRecommendation:
        """Convert the study results into hyperparameter recommendations"""
        pass  # pragma: no cover

    def _normalize(self, hps: dict[str, float]) -> dict[str, float]:
        """Normalize the given raw hyperparameters. Intended to be overridden
        by subclasses when necessary. The default is to keep them as-is."""
        return hps

    def optimize(
        self, n_trials: int, n_jobs: int, results_file: LazyFile | None
    ) -> HPRecommendation:
        """Find the optimal hyperparameters by testing up to the given number
        of hyperparameter combinations"""

        objective_args = self._prepare(n_jobs)
        self._objective.init(objective_args)

        writer = TrialWriter(results_file, self._normalize) if results_file else None
        write_callback = writer.write if writer else None

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        storage_url = f"sqlite:///{temp_db.name}"

        study = optuna.create_study(direction="maximize", storage=storage_url)

        # evaluate the fixed initial trials first so that the sampler
        # conditions on them from its first iteration; these are useful
        # for covering regions of the search space that the sampler's
        # random startup phase would rarely visit on its own
        best_index = None
        best_value = None
        for index, params in enumerate(self._initial_trials(), start=1):
            fixed_trial = _FixedParamsTrial(params)
            value = self._objective.objective(fixed_trial, objective_args)
            trial = optuna.trial.create_trial(
                value=value, params=params, distributions=fixed_trial.distributions
            )
            study.add_trial(trial)
            if best_value is None or value > best_value:
                best_value = value
                best_index = index
            self._backend.info(
                f"initial trial {index} finished with value: {value} and "
                f"parameters: {params}. Best so far is initial trial "
                f"{best_index} with value: {best_value}."
            )
            if write_callback:
                write_callback(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                    }
                )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)
        with pool_class(jobs) as pool:
            for i in range(n_trials):
                pool.apply_async(
                    self._objective.run_trial,
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
