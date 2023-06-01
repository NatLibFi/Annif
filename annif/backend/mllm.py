"""Maui-like Lexical Matching backend"""
from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

import annif.eval
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.lexical.mllm import MLLMModel
from annif.suggestion import vector_to_suggestions

from . import backend, hyperopt

if TYPE_CHECKING:
    from collections.abc import Iterator

    from optuna.study.study import Study
    from optuna.trial import Trial

    from annif.backend.hyperopt import HPRecommendation
    from annif.corpus.document import DocumentCorpus
    from annif.lexical.mllm import Candidate


class MLLMOptimizer(hyperopt.HyperparameterOptimizer):
    """Hyperparameter optimizer for the MLLM backend"""

    def _prepare(self, n_jobs: int = 1) -> None:
        self._backend.initialize()
        self._train_x, self._train_y = self._backend._load_train_data()
        self._candidates = []
        self._gold_subjects = []

        # TODO parallelize generation of candidates
        for doc in self._corpus.documents:
            candidates = self._backend._generate_candidates(doc.text)
            self._candidates.append(candidates)
            self._gold_subjects.append(doc.subject_set)

    def _objective(self, trial: Trial) -> float:
        params = {
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 100, 2000),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "limit": 100,
        }
        model = self._backend._model._create_classifier(params)
        model.fit(self._train_x, self._train_y)

        batch = annif.eval.EvaluationBatch(self._backend.project.subjects)
        for goldsubj, candidates in zip(self._gold_subjects, self._candidates):
            if candidates:
                features = self._backend._model._candidates_to_features(candidates)
                scores = model.predict_proba(features)
                ranking = self._backend._model._prediction_to_list(scores, candidates)
            else:
                ranking = []
            results = self._backend._prediction_to_result(ranking, params)
            batch.evaluate_many([results], [goldsubj])
        results = batch.results(metrics=[self._metric])
        return results[self._metric]

    def _postprocess(self, study: Study) -> HPRecommendation:
        bp = study.best_params
        lines = [
            f"min_samples_leaf={bp['min_samples_leaf']}",
            f"max_leaf_nodes={bp['max_leaf_nodes']}",
            f"max_samples={bp['max_samples']:.4f}",
        ]
        return hyperopt.HPRecommendation(lines=lines, score=study.best_value)


class MLLMBackend(hyperopt.AnnifHyperoptBackend):
    """Maui-like Lexical Matching backend for Annif"""

    name = "mllm"

    # defaults for unitialized instances
    _model = None

    MODEL_FILE = "mllm-model.gz"
    TRAIN_FILE = "mllm-train.gz"

    DEFAULT_PARAMETERS = {
        "min_samples_leaf": 20,
        "max_leaf_nodes": 1000,
        "max_samples": 0.9,
        "use_hidden_labels": False,
    }

    def get_hp_optimizer(self, corpus: DocumentCorpus, metric: str) -> MLLMOptimizer:
        return MLLMOptimizer(self, corpus, metric)

    def default_params(self) -> dict[str, Any]:
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _load_model(self) -> MLLMModel:
        path = os.path.join(self.datadir, self.MODEL_FILE)
        self.debug("loading model from {}".format(path))
        if os.path.exists(path):
            return MLLMModel.load(path)
        else:
            raise NotInitializedException(
                "model {} not found".format(path), backend_id=self.backend_id
            )

    def _load_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.datadir, self.TRAIN_FILE)
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise NotInitializedException(
                "train data file {} not found".format(path), backend_id=self.backend_id
            )

    def initialize(self, parallel: bool = False) -> None:
        if self._model is None:
            self._model = self._load_model()

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        self.info("starting train")
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "training backend {} with no documents".format(self.backend_id)
                )
            self.info("preparing training data")
            self._model = MLLMModel()
            train_data = self._model.prepare_train(
                corpus, self.project.vocab, self.project.analyzer, params, jobs
            )
            annif.util.atomic_save(
                train_data, self.datadir, self.TRAIN_FILE, method=joblib.dump
            )
        else:
            self.info("reusing cached training data from previous run")
            self._model = self._load_model()
            train_data = self._load_train_data()

        self.info("training model")
        self._model.train(train_data[0], train_data[1], params)

        self.info("saving model")
        annif.util.atomic_save(self._model, self.datadir, self.MODEL_FILE)

    def _generate_candidates(self, text: str) -> list[Candidate]:
        return self._model.generate_candidates(text, self.project.analyzer)

    def _prediction_to_result(
        self,
        prediction: list[tuple[np.float64, int]],
        params: dict[str, Any],
    ) -> Iterator:
        vector = np.zeros(len(self.project.subjects), dtype=np.float32)
        for score, subject_id in prediction:
            vector[subject_id] = score
        return vector_to_suggestions(vector, int(params["limit"]))

    def _suggest(self, text: str, params: dict[str, Any]) -> Iterator:
        candidates = self._generate_candidates(text)
        prediction = self._model.predict(candidates)
        return self._prediction_to_result(prediction, params)
