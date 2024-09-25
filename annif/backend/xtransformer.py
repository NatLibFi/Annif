"""Annif backend using the transformer variant of pecos."""

import logging
import os.path as osp
from sys import stdout
from typing import Any

import numpy as np
import scipy.sparse as sp
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc.xtransformer import matcher, model
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText

from annif.corpus.document import DocumentCorpus
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SubjectSuggestion, SuggestionBatch
from annif.util import (
    apply_param_parse_config,
    atomic_save,
    atomic_save_folder,
    boolean,
)

from . import backend, mixins


class XTransformerBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """XTransformer based backend for Annif"""

    name = "xtransformer"
    needs_subject_index = True

    _model = None

    train_X_file = "xtransformer-train-X.npz"
    train_y_file = "xtransformer-train-y.npz"
    train_txt_file = "xtransformer-train-raw.txt"
    model_folder = "xtransformer-model"

    PARAM_CONFIG = {
        "min_df": int,
        "ngram": int,
        "fix_clustering": boolean,
        "nr_splits": int,
        "min_codes": int,
        "max_leaf_size": int,
        "imbalanced_ratio": float,
        "imbalanced_depth": int,
        "max_match_clusters": int,
        "do_fine_tune": boolean,
        "model_shortcut": str,
        "beam_size": int,
        "limit": int,
        "post_processor": str,
        "negative_sampling": str,
        "ensemble_method": str,
        "threshold": float,
        "loss_function": str,
        "truncate_length": int,
        "hidden_droput_prob": float,
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "learning_rate": float,
        "weight_decay": float,
        "adam_epsilon": float,
        "num_train_epochs": int,
        "max_steps": int,
        "lr_schedule": str,
        "warmup_steps": int,
        "logging_steps": int,
        "save_steps": int,
        "max_active_matching_labels": int,
        "max_num_labels_in_gpu": int,
        "use_gpu": boolean,
        "bootstrap_model": str,
    }

    DEFAULT_PARAMETERS = {
        "min_df": 1,
        "ngram": 1,
        "fix_clustering": False,
        "nr_splits": 16,
        "min_codes": None,
        "max_leaf_size": 100,
        "imbalanced_ratio": 0.0,
        "imbalanced_depth": 100,
        "max_match_clusters": 32768,
        "do_fine_tune": True,
        "model_shortcut": "distilbert-base-multilingual-uncased",
        "beam_size": 20,
        "limit": 100,
        "post_processor": "sigmoid",
        "negative_sampling": "tfn",
        "ensemble_method": "transformer-only",
        "threshold": 0.1,
        "loss_function": "squared-hinge",
        "truncate_length": 128,
        "hidden_droput_prob": 0.1,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "num_train_epochs": 1,
        "max_steps": 0,
        "lr_schedule": "linear",
        "warmup_steps": 0,
        "logging_steps": 100,
        "save_steps": 1000,
        "max_active_matching_labels": None,
        "max_num_labels_in_gpu": 65536,
        "use_gpu": True,
        "bootstrap_model": "linear",
    }

    def _initialize_model(self):
        if self._model is None:
            path = osp.join(self.datadir, self.model_folder)
            self.debug("loading model from {}".format(path))
            if osp.exists(path):
                self._model = XTransformer.load(path)
            else:
                raise NotInitializedException(
                    "model {} not found".format(path), backend_id=self.backend_id
                )

    def initialize(self, parallel: bool = False) -> None:
        self.initialize_vectorizer()
        self._initialize_model()

    def default_params(self):
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def _create_train_files(self, veccorpus, corpus):
        self.info("creating train file")
        Xs = []
        ys = []
        txt_pth = osp.join(self.datadir, self.train_txt_file)
        with open(txt_pth, "w", encoding="utf-8") as txt_file:
            for doc, vector in zip(corpus.documents, veccorpus):
                subject_set = doc.subject_set
                if not (subject_set and doc.text):
                    continue  # noqa
                print(" ".join(doc.text.split()), file=txt_file)
                Xs.append(sp.csr_matrix(vector, dtype=np.float32).sorted_indices())
                ys.append(
                    sp.csr_matrix(
                        (
                            np.ones(len(subject_set)),
                            (np.zeros(len(subject_set)), [s for s in subject_set]),
                        ),
                        shape=(1, len(self.project.subjects)),
                        dtype=np.float32,
                    ).sorted_indices()
                )
        atomic_save(
            sp.vstack(Xs, format="csr"),
            self.datadir,
            self.train_X_file,
            method=lambda mtrx, target: sp.save_npz(target, mtrx, compressed=True),
        )
        atomic_save(
            sp.vstack(ys, format="csr"),
            self.datadir,
            self.train_y_file,
            method=lambda mtrx, target: sp.save_npz(target, mtrx, compressed=True),
        )

    def _create_model(self, params, jobs):
        train_txts = Preprocessor.load_data_from_file(
            osp.join(self.datadir, self.train_txt_file),
            label_text_path=None,
            text_pos=0,
        )["corpus"]
        train_X = sp.load_npz(osp.join(self.datadir, self.train_X_file))
        train_y = sp.load_npz(osp.join(self.datadir, self.train_y_file))
        model_path = osp.join(self.datadir, self.model_folder)
        new_params = apply_param_parse_config(self.PARAM_CONFIG, self.params)
        new_params["only_topk"] = new_params.pop("limit")
        train_params = XTransformer.TrainParams.from_dict(
            new_params, recursive=True
        ).to_dict()
        pred_params = XTransformer.PredParams.from_dict(
            new_params, recursive=True
        ).to_dict()

        self.info("Start training")
        # enable progress
        matcher.LOGGER.setLevel(logging.DEBUG)
        matcher.LOGGER.addHandler(logging.StreamHandler(stream=stdout))
        model.LOGGER.setLevel(logging.DEBUG)
        model.LOGGER.addHandler(logging.StreamHandler(stream=stdout))
        self._model = XTransformer.train(
            MLProblemWithText(train_txts, train_y, X_feat=train_X),
            clustering=None,
            val_prob=None,
            train_params=train_params,
            pred_params=pred_params,
            beam_size=int(params["beam_size"]),
            steps_scale=None,
            label_feat=None,
        )
        atomic_save_folder(self._model, model_path)

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus == "cached":
            self.info("Reusing cached training data from previous run.")
        else:
            if corpus.is_empty():
                raise NotSupportedException("Cannot t project with no documents")
            input = (doc.text for doc in corpus.documents)
            vecparams = {
                "min_df": int(params["min_df"]),
                "tokenizer": self.project.analyzer.tokenize_words,
                "ngram_range": (1, int(params["ngram"])),
            }
            veccorpus = self.create_vectorizer(input, vecparams)
            self._create_train_files(veccorpus, corpus)
        self._create_model(params, jobs)

    def _suggest_batch(
        self, texts: list[str], params: dict[str, Any]
    ) -> SuggestionBatch:
        vector = self.vectorizer.transform(texts)
        if vector.nnz == 0:  # All zero vector, empty result
            return list()
        new_params = apply_param_parse_config(self.PARAM_CONFIG, params)
        prediction = self._model.predict(
            texts,
            X_feat=vector.sorted_indices(),
            batch_size=new_params["batch_size"],
            use_gpu=True,
            only_top_k=new_params["limit"],
            post_processor=new_params["post_processor"],
        )
        current_batchsize = prediction.get_shape()[0]
        batch_result = []
        for i in range(current_batchsize):
            results = []
            row = prediction.getrow(i)
            for idx, score in zip(row.indices, row.data):
                results.append(SubjectSuggestion(subject_id=idx, score=score))
            batch_result.append(results)
        return SuggestionBatch.from_sequence(batch_result, self.project.subjects)
