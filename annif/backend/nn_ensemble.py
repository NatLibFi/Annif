"""Neural network based ensemble backend that combines results from multiple
projects."""
from __future__ import annotations

import os.path
import shutil
from io import BytesIO
from typing import TYPE_CHECKING, Any

import joblib
import keras.backend as K
import lmdb
import numpy as np
from keras.layers import Add, Dense, Dropout, Flatten, Input, Layer
from keras.models import Model
from keras.saving import load_model
from keras.utils import Sequence
from scipy.sparse import csc_matrix, csr_matrix

import annif.corpus
import annif.parallel
import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SuggestionBatch, vector_to_suggestions

from . import backend, ensemble

if TYPE_CHECKING:
    from tensorflow.python.framework.ops import EagerTensor

    from annif.corpus.document import DocumentCorpus


def idx_to_key(idx: int) -> bytes:
    """convert an integer index to a binary key for use in LMDB"""
    return b"%08d" % idx


def key_to_idx(key: memoryview | bytes) -> int:
    """convert a binary LMDB key to an integer index"""
    return int(key)


class LMDBSequence(Sequence):
    """A sequence of samples stored in a LMDB database."""

    def __init__(self, txn, batch_size):
        self._txn = txn
        cursor = txn.cursor()
        if cursor.last():
            # Counter holds the number of samples in the database
            self._counter = key_to_idx(cursor.key()) + 1
        else:  # empty database
            self._counter = 0
        self._batch_size = batch_size

    def add_sample(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        # use zero-padded 8-digit key
        key = idx_to_key(self._counter)
        self._counter += 1
        # convert the sample into a sparse matrix and serialize it as bytes
        sample = (csc_matrix(inputs), csr_matrix(targets))
        buf = BytesIO()
        joblib.dump(sample, buf)
        buf.seek(0)
        self._txn.put(key, buf.read())

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """get a particular batch of samples"""
        cursor = self._txn.cursor()
        first_key = idx * self._batch_size
        cursor.set_key(idx_to_key(first_key))
        input_arrays = []
        target_arrays = []
        for key, value in cursor.iternext():
            if key_to_idx(key) >= (first_key + self._batch_size):
                break
            input_csr, target_csr = joblib.load(BytesIO(value))
            input_arrays.append(input_csr.toarray())
            target_arrays.append(target_csr.toarray().flatten())
        return np.array(input_arrays), np.array(target_arrays)

    def __len__(self) -> int:
        """return the number of available batches"""
        return int(np.ceil(self._counter / self._batch_size))


class MeanLayer(Layer):
    """Custom Keras layer that calculates mean values along the 2nd axis."""

    def call(self, inputs: EagerTensor) -> EagerTensor:
        return K.mean(inputs, axis=2)


class NNEnsembleBackend(backend.AnnifLearningBackend, ensemble.BaseEnsembleBackend):
    """Neural network ensemble backend that combines results from multiple
    projects"""

    name = "nn_ensemble"

    MODEL_FILE = "nn-model.keras"
    LMDB_FILE = "nn-train.mdb"

    DEFAULT_PARAMETERS = {
        "nodes": 100,
        "dropout_rate": 0.2,
        "optimizer": "adam",
        "epochs": 10,
        "learn-epochs": 1,
        "lmdb_map_size": 1024 * 1024 * 1024,
    }

    # defaults for uninitialized instances
    _model = None

    def initialize(self, parallel: bool = False) -> None:
        super().initialize(parallel)
        if self._model is not None:
            return  # already initialized
        if parallel:
            # Don't load TF model just before parallel execution,
            # since it won't work after forking worker processes
            return
        model_filename = os.path.join(self.datadir, self.MODEL_FILE)
        if not os.path.exists(model_filename):
            raise NotInitializedException(
                "model file {} not found".format(model_filename),
                backend_id=self.backend_id,
            )
        self.debug("loading Keras model from {}".format(model_filename))
        self._model = load_model(
            model_filename, custom_objects={"MeanLayer": MeanLayer}
        )

    def _merge_source_batches(
        self,
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        src_weight = dict(sources)
        score_vectors = np.array(
            [
                [
                    np.sqrt(suggestions.as_vector())
                    * src_weight[project_id]
                    * len(batch_by_source)
                    for suggestions in batch
                ]
                for project_id, batch in batch_by_source.items()
            ],
            dtype=np.float32,
        ).transpose(1, 2, 0)
        prediction = self._model(score_vectors).numpy()
        return SuggestionBatch.from_sequence(
            [
                vector_to_suggestions(row, limit=int(params["limit"]))
                for row in prediction
            ],
            self.project.subjects,
        )

    def _create_model(self, sources: list[tuple[str, float]]) -> None:
        self.info("creating NN ensemble model")

        inputs = Input(shape=(len(self.project.subjects), len(sources)))

        flat_input = Flatten()(inputs)
        drop_input = Dropout(rate=float(self.params["dropout_rate"]))(flat_input)
        hidden = Dense(int(self.params["nodes"]), activation="relu")(drop_input)
        drop_hidden = Dropout(rate=float(self.params["dropout_rate"]))(hidden)
        delta = Dense(
            len(self.project.subjects),
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )(drop_hidden)

        mean = MeanLayer()(inputs)

        predictions = Add()([mean, delta])

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.compile(
            optimizer=self.params["optimizer"],
            loss="binary_crossentropy",
            metrics=["top_k_categorical_accuracy"],
        )
        if "lr" in self.params:
            self._model.optimizer.learning_rate.assign(float(self.params["lr"]))

        summary = []
        self._model.summary(print_fn=summary.append)
        self.debug("Created model: \n" + "\n".join(summary))

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        sources = annif.util.parse_sources(self.params["sources"])
        self._create_model(sources)
        self._fit_model(
            corpus,
            epochs=int(params["epochs"]),
            lmdb_map_size=int(params["lmdb_map_size"]),
            n_jobs=jobs,
        )

    def _corpus_to_vectors(
        self,
        corpus: DocumentCorpus,
        seq: LMDBSequence,
        n_jobs: int,
    ) -> None:
        # pass corpus through all source projects
        sources = dict(annif.util.parse_sources(self.params["sources"]))

        # initialize the source projects before forking, to save memory
        self.info(f"Initializing source projects: {', '.join(sources.keys())}")
        for project_id in sources.keys():
            project = self.project.registry.get_project(project_id)
            project.initialize(parallel=True)

        psmap = annif.parallel.ProjectSuggestMap(
            self.project.registry,
            list(sources.keys()),
            backend_params=None,
            limit=None,
            threshold=0.0,
        )

        jobs, pool_class = annif.parallel.get_pool(n_jobs)

        self.info("Processing training documents...")
        with pool_class(jobs) as pool:
            for hits, subject_set in pool.imap_unordered(
                psmap.suggest, corpus.documents
            ):
                doc_scores = []
                for project_id, p_hits in hits.items():
                    vector = p_hits.as_vector()
                    doc_scores.append(
                        np.sqrt(vector) * sources[project_id] * len(sources)
                    )
                score_vector = np.array(doc_scores, dtype=np.float32).transpose()
                true_vector = subject_set.as_vector(len(self.project.subjects))
                seq.add_sample(score_vector, true_vector)

    def _open_lmdb(self, cached, lmdb_map_size):
        lmdb_path = os.path.join(self.datadir, self.LMDB_FILE)
        if not cached and os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        return lmdb.open(lmdb_path, map_size=lmdb_map_size, writemap=True)

    def _fit_model(
        self,
        corpus: DocumentCorpus,
        epochs: int,
        lmdb_map_size: int,
        n_jobs: int = 1,
    ) -> None:
        env = self._open_lmdb(corpus == "cached", lmdb_map_size)
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "Cannot train nn_ensemble project with no documents"
                )
            with env.begin(write=True, buffers=True) as txn:
                seq = LMDBSequence(txn, batch_size=32)
                self._corpus_to_vectors(corpus, seq, n_jobs)
        else:
            self.info("Reusing cached training data from previous run.")
        # fit the model using a read-only view of the LMDB
        self.info("Training neural network model...")
        with env.begin(buffers=True) as txn:
            seq = LMDBSequence(txn, batch_size=32)
            self._model.fit(seq, verbose=True, epochs=epochs)

        annif.util.atomic_save(self._model, self.datadir, self.MODEL_FILE)

    def _learn(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
    ) -> None:
        self.initialize()
        self._fit_model(
            corpus, int(params["learn-epochs"]), int(params["lmdb_map_size"])
        )
