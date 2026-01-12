"""Neural network based ensemble backend that combines results from multiple
projects."""

from __future__ import annotations

import importlib
import json
import os.path
import shutil
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING, Any

import joblib
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix, csr_matrix
from torch.utils.data import DataLoader, Dataset

import annif.corpus
import annif.parallel
import annif.util
from annif.exception import (
    NotInitializedException,
    NotSupportedException,
    OperationFailedException,
)
from annif.suggestion import SuggestionBatch, vector_to_suggestions

from . import backend, ensemble

if TYPE_CHECKING:
    from annif.corpus.document import DocumentCorpus

logger = annif.logger


def idx_to_key(idx: int) -> bytes:
    """convert an integer index to a binary key for use in LMDB"""
    return b"%08d" % idx


def key_to_idx(key: memoryview | bytes) -> int:
    """convert a binary LMDB key to an integer index"""
    return int(key)


class LMDBDataset(Dataset):
    """A sequence of samples stored in a LMDB database."""

    def __init__(self, txn):
        super().__init__()
        self._txn = txn
        cursor = txn.cursor()
        if cursor.last():
            # Counter holds the number of samples in the database
            self._counter = key_to_idx(cursor.key()) + 1
        else:  # empty database
            self._counter = 0

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
        """get a particular sample"""
        cursor = self._txn.cursor()
        cursor.set_key(idx_to_key(idx))
        value = cursor.value()
        input_csr, target_csr = joblib.load(BytesIO(value))
        input_tensor = torch.from_numpy(input_csr.toarray())
        target_tensor = torch.from_numpy(target_csr.toarray()[0]).float()
        return input_tensor, target_tensor

    def __len__(self) -> int:
        """return the number of available samples"""
        return self._counter


class NNEnsembleModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float
    ):
        super().__init__()
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "dropout_rate": dropout_rate,
        }
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.delta_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=1)
        x = self.flatten(inputs)
        x = self.dropout1(x)
        x = F.relu(self.hidden(x))
        x = self.dropout2(x)
        delta = self.delta_layer(x)
        return mean + delta

    def save(self, filepath):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_config": self.model_config,
                "model_class": self.__class__.__name__,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath, map_location="cpu"):
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        config = checkpoint["model_config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


class NNEnsembleBackend(backend.AnnifLearningBackend, ensemble.BaseEnsembleBackend):
    """Neural network ensemble backend that combines results from multiple
    projects"""

    name = "nn_ensemble"

    MODEL_FILE = "nn-model.pt"
    LMDB_FILE = "nn-train.mdb"

    DEFAULT_PARAMETERS = {
        "nodes": 100,
        "dropout_rate": 0.2,
        "optimizer": "adam",
        "lr": 0.001,
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
            # Don't load model just before parallel execution,
            # since it won't work after forking worker processes
            return
        model_filename = os.path.join(self.datadir, self.MODEL_FILE)
        if not os.path.exists(model_filename):
            raise NotInitializedException(
                "model file {} not found".format(model_filename),
                backend_id=self.backend_id,
            )
        self.debug("loading model from {}".format(model_filename))
        try:
            self._model = NNEnsembleModel.load(model_filename)
        except Exception as err:
            metadata = self.get_model_metadata(model_filename)
            message = (
                f"loading model from {model_filename}; "
                f"model metadata: {metadata}; "
                f'Original error message: "{err}"'
            )
            raise OperationFailedException(message, backend_id=self.backend_id)

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
        )
        score_vector_tensor = torch.from_numpy(score_vectors.swapaxes(0, 1))
        with torch.no_grad():
            prediction = self._model(score_vector_tensor)
        return SuggestionBatch.from_sequence(
            [
                vector_to_suggestions(row, limit=int(params["limit"]))
                for row in prediction
            ],
            self.project.subjects,
        )

    def _create_model(self, sources: list[tuple[str, float]]) -> None:
        self.info("creating NN ensemble model")

        # Create PyTorch model
        input_dim = len(self.project.subjects) * len(sources)
        hidden_dim = int(self.params["nodes"])
        output_dim = len(self.project.subjects)
        dropout_rate = float(self.params["dropout_rate"])

        self._model = NNEnsembleModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )

    #        summary = []
    #        self._model.summary(print_fn=summary.append)
    #        self.debug("Created model: \n" + "\n".join(summary))

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
        seq: LMDBDataset,
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
                score_vector = np.array(doc_scores, dtype=np.float32)
                true_vector = subject_set.as_vector(len(self.project.subjects))
                seq.add_sample(score_vector, true_vector)

    def _open_lmdb(self, cached, lmdb_map_size):
        lmdb_path = os.path.join(self.datadir, self.LMDB_FILE)
        if not cached and os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        return lmdb.open(lmdb_path, map_size=lmdb_map_size, writemap=True, mode=0o775)

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
                seq = LMDBDataset(txn)
                self._corpus_to_vectors(corpus, seq, n_jobs)
        else:
            self.info("Reusing cached training data from previous run.")

        # fit the model using a read-only view of the LMDB
        self.info("Training neural network model...")
        with env.begin(buffers=True) as txn:
            dataset = LMDBDataset(txn)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Training loop
            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=float(self.params["lr"])
            )
            criterion = nn.BCEWithLogitsLoss()

            self._model.train()
            for epoch in range(epochs):
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

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

    def get_model_metadata(self, model_filename: str) -> dict | None:
        """Read metadata from Keras model files."""

        try:
            with zipfile.ZipFile(model_filename, "r") as zip:
                with zip.open("metadata.json") as metadata_file:
                    metadata_str = metadata_file.read().decode("utf-8")
                    metadata = json.loads(metadata_str)
                    return metadata
        except Exception:
            self.warning(f"Failed to read metadata from {model_filename}")
            return None
