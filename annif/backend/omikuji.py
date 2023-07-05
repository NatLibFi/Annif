"""Annif backend using the Omikuji classifier"""
from __future__ import annotations

import os.path
import shutil
from typing import TYPE_CHECKING, Any

import omikuji

import annif.util
from annif.exception import (
    NotInitializedException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SubjectSuggestion, SuggestionBatch

from . import backend, mixins

if TYPE_CHECKING:
    from scipy.sparse._csr import csr_matrix

    from annif.corpus.document import DocumentCorpus


class OmikujiBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """Omikuji based backend for Annif"""

    name = "omikuji"

    # defaults for uninitialized instances
    _model = None

    TRAIN_FILE = "omikuji-train.txt"
    MODEL_FILE = "omikuji-model"

    DEFAULT_PARAMETERS = {
        "min_df": 1,
        "ngram": 1,
        "cluster_balanced": True,
        "cluster_k": 2,
        "max_depth": 20,
        "collapse_every_n_layers": 0,
    }

    def _initialize_model(self) -> None:
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug("loading model from {}".format(path))
            if os.path.exists(path):
                try:
                    self._model = omikuji.Model.load(path)
                except RuntimeError:
                    raise OperationFailedException(
                        "Omikuji models trained on Annif versions older than "
                        "0.56 cannot be loaded. Please retrain your project."
                    )
            else:
                raise NotInitializedException(
                    "model {} not found".format(path), backend_id=self.backend_id
                )

    def initialize(self, parallel: bool = False) -> None:
        self.initialize_vectorizer()
        self._initialize_model()

    def _create_train_file(self, veccorpus: csr_matrix, corpus: DocumentCorpus) -> None:
        self.info("creating train file")
        path = os.path.join(self.datadir, self.TRAIN_FILE)
        with open(path, "w", encoding="utf-8") as trainfile:
            # Extreme Classification Repository format header line
            # We don't yet know the number of samples, as some may be skipped
            print(
                "00000000",
                len(self.vectorizer.vocabulary_),
                len(self.project.subjects),
                file=trainfile,
            )
            n_samples = 0
            for doc, vector in zip(corpus.documents, veccorpus):
                subject_ids = [str(subject_id) for subject_id in doc.subject_set]
                feature_values = [
                    "{}:{}".format(col, vector[row, col])
                    for row, col in zip(*vector.nonzero())
                ]
                if not subject_ids or not feature_values:
                    continue  # noqa
                print(",".join(subject_ids), " ".join(feature_values), file=trainfile)
                n_samples += 1
            # replace the number of samples value at the beginning
            trainfile.seek(0)
            print("{:08d}".format(n_samples), end="", file=trainfile)

    def _create_model(self, params: dict[str, Any], jobs: int) -> None:
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        model_path = os.path.join(self.datadir, self.MODEL_FILE)
        hyper_param = omikuji.Model.default_hyper_param()

        hyper_param.cluster_balanced = annif.util.boolean(params["cluster_balanced"])
        hyper_param.cluster_k = int(params["cluster_k"])
        hyper_param.max_depth = int(params["max_depth"])
        hyper_param.collapse_every_n_layers = int(params["collapse_every_n_layers"])

        self._model = omikuji.Model.train_on_data(train_path, hyper_param, jobs or None)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        self._model.save(os.path.join(self.datadir, self.MODEL_FILE))

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "Cannot train omikuji project with no documents"
                )
            input = (doc.text for doc in corpus.documents)
            vecparams = {
                "min_df": int(params["min_df"]),
                "tokenizer": self.project.analyzer.tokenize_words,
                "ngram_range": (1, int(params["ngram"])),
            }
            veccorpus = self.create_vectorizer(input, vecparams)
            self._create_train_file(veccorpus, corpus)
        else:
            self.info("Reusing cached training data from previous run.")
        self._create_model(params, jobs)

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        vector = self.vectorizer.transform(texts)
        limit = int(params["limit"])

        batch_results = []
        for row in vector:
            if row.nnz == 0:  # All zero vector, empty result
                batch_results.append([])
                continue
            feature_values = [(col, row[0, col]) for col in row.nonzero()[1]]
            results = []
            for subj_id, score in self._model.predict(feature_values, top_k=limit):
                results.append(SubjectSuggestion(subject_id=subj_id, score=score))
            batch_results.append(results)
        return SuggestionBatch.from_sequence(batch_results, self.project.subjects)
