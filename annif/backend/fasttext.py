"""Annif backend using the fastText classifier"""
from __future__ import annotations

import collections
import os.path
from typing import TYPE_CHECKING, Any

import fasttext

import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SubjectSuggestion

from . import backend, mixins

if TYPE_CHECKING:
    from fasttext.FastText import _FastText
    from numpy import ndarray

    from annif.corpus.document import DocumentCorpus


class FastTextBackend(mixins.ChunkingBackend, backend.AnnifBackend):
    """fastText backend for Annif"""

    name = "fasttext"

    FASTTEXT_PARAMS = {
        "lr": float,
        "lrUpdateRate": int,
        "dim": int,
        "ws": int,
        "epoch": int,
        "minCount": int,
        "neg": int,
        "wordNgrams": int,
        "loss": str,
        "bucket": int,
        "minn": int,
        "maxn": int,
        "thread": int,
        "t": float,
        "pretrainedVectors": str,
    }

    DEFAULT_PARAMETERS = {
        "dim": 100,
        "lr": 0.25,
        "epoch": 5,
        "loss": "hs",
    }

    MODEL_FILE = "fasttext-model"
    TRAIN_FILE = "fasttext-train.txt"

    # defaults for uninitialized instances
    _model = None

    def default_params(self) -> dict[str, Any]:
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(mixins.ChunkingBackend.DEFAULT_PARAMETERS)
        params.update(self.DEFAULT_PARAMETERS)
        return params

    @staticmethod
    def _load_model(path: str) -> _FastText:
        # monkey patch fasttext.FastText.eprint to avoid spurious warning
        # see https://github.com/facebookresearch/fastText/issues/1067
        orig_eprint = fasttext.FastText.eprint
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model(path)
        # restore the original eprint
        fasttext.FastText.eprint = orig_eprint
        return model

    def initialize(self, parallel: bool = False) -> None:
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug("loading fastText model from {}".format(path))
            if os.path.exists(path):
                self._model = self._load_model(path)
                self.debug("loaded model {}".format(str(self._model)))
                self.debug("dim: {}".format(self._model.get_dimension()))
            else:
                raise NotInitializedException(
                    "model {} not found".format(path), backend_id=self.backend_id
                )

    @staticmethod
    def _id_to_label(subject_id: int) -> str:
        return "__label__{:d}".format(subject_id)

    def _label_to_subject_id(self, label: str) -> int:
        labelnum = label.replace("__label__", "")
        return int(labelnum)

    def _write_train_file(self, corpus: DocumentCorpus, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as trainfile:
            for doc in corpus.documents:
                text = self._normalize_text(doc.text)
                if text == "":
                    continue
                labels = [self._id_to_label(sid) for sid in doc.subject_set]
                if labels:
                    print(" ".join(labels), text, file=trainfile)
                else:
                    self.warning(f'no labels for document "{doc.text}"')

    def _normalize_text(self, text: str) -> str:
        return " ".join(self.project.analyzer.tokenize_words(text))

    def _create_train_file(
        self,
        corpus: DocumentCorpus,
    ) -> None:
        self.info("creating fastText training file")

        annif.util.atomic_save(
            corpus, self.datadir, self.TRAIN_FILE, method=self._write_train_file
        )

    def _create_model(self, params: dict[str, Any], jobs: int) -> None:
        self.info("creating fastText model")
        trainpath = os.path.join(self.datadir, self.TRAIN_FILE)
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        params = {
            param: self.FASTTEXT_PARAMS[param](val)
            for param, val in params.items()
            if param in self.FASTTEXT_PARAMS
        }
        if jobs != 0:  # jobs set by user to non-default value
            params["thread"] = jobs
        self.debug("Model parameters: {}".format(params))
        self._model = fasttext.train_supervised(trainpath, **params)
        self._model.save_model(modelpath)

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "training backend {} with no documents".format(self.backend_id)
                )
            self._create_train_file(corpus)
        else:
            self.info("Reusing cached training data from previous run.")
        self._create_model(params, jobs)

    def _predict_chunks(
        self, chunktexts: list[str], limit: int
    ) -> tuple[list[list[str]], list[ndarray]]:
        return self._model.predict(
            list(
                filter(
                    None, [self._normalize_text(chunktext) for chunktext in chunktexts]
                )
            ),
            limit,
        )

    def _suggest_chunks(
        self, chunktexts: list[str], params: dict[str, Any]
    ) -> list[SubjectSuggestion]:
        limit = int(params["limit"])
        chunklabels, chunkscores = self._predict_chunks(chunktexts, limit)
        label_scores = collections.defaultdict(float)
        for labels, scores in zip(chunklabels, chunkscores):
            for label, score in zip(labels, scores):
                label_scores[label] += score
        best_labels = sorted(
            [(score, label) for label, score in label_scores.items()], reverse=True
        )

        results = []
        for score, label in best_labels[:limit]:
            results.append(
                SubjectSuggestion(
                    subject_id=self._label_to_subject_id(label),
                    score=score / len(chunktexts),
                )
            )
        return results
