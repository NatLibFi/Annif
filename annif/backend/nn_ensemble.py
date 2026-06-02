"""Neural network based ensemble backend that combines results from multiple
projects."""

from __future__ import annotations

import copy
import os.path
import shutil
import sys
from io import BytesIO
from typing import TYPE_CHECKING, Any

import joblib
import lmdb
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        sample = (csr_matrix(inputs), csr_matrix(targets))
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
        input_tensor = torch.log1p(torch.from_numpy(input_csr.toarray()))
        target_tensor = torch.log1p(torch.from_numpy(target_csr.toarray()[0]).float())
        return input_tensor, target_tensor

    def get_subset(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Fetch a fixed set of samples by index and stack into batch tensors.

        Returns (inputs, targets) where inputs is (B, M, N) and targets is (B, N).
        """
        inputs_list, targets_list = [], []
        for idx in indices:
            inp, tgt = self[idx]
            inputs_list.append(inp)
            targets_list.append(tgt)
        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(targets_list, dim=0)
        return inputs, targets

    def __len__(self) -> int:
        """return the number of available samples"""
        return self._counter


class NNEnsembleModel(nn.Module):
    def __init__(self, n_sources: int, n_subjects: int, source_weights: list[float]):
        super().__init__()
        self.model_config = {
            "n_sources": n_sources,
            "n_subjects": n_subjects,
            "source_weights": source_weights,
        }
        # per-concept/source weights
        init_weights = torch.tensor(source_weights, dtype=torch.float32)
        init_weights = init_weights / init_weights.sum()
        self.weights = nn.Parameter(
            init_weights[:, None].expand(-1, n_subjects).contiguous()
        )
        # bias decomposition: global + per-label delta
        self.bias_global = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias_delta = nn.Parameter(torch.zeros(n_subjects, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor):
        weighted = inputs * self.weights.unsqueeze(0)
        return weighted.sum(1) + self.bias_global + self.bias_delta

    def save(self, filepath):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_class": self.__class__.__name__,
                "model_config": self.model_config,
                "pytorch_version": str(torch.__version__),
                "python_version": sys.version,
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


class EarlyStopping:
    def __init__(self, patience: int):
        self._patience = patience
        self._best_metric = None
        self._no_improvement_count = 0
        self._stop_early = False
        self.best_state = None
        self.best_epoch = 0

    def __call__(self, model, metric, epoch):
        if self._best_metric is None or metric > self._best_metric:
            self._best_metric = metric
            self._no_improvement_count = 0
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self._no_improvement_count += 1
            if self._no_improvement_count > self._patience:
                self._stop_early = True

        return self._stop_early


@torch.no_grad()
def ndcg_batch(preds: torch.Tensor, targets: torch.Tensor):
    """
    preds:   (B, N) float
    targets: (B, N) {0,1}

    Returns: mean NDCG across the batch
    """
    sorted_idx = torch.argsort(preds, dim=1, descending=True)
    sorted_targets = torch.gather(targets, 1, sorted_idx)

    L = sorted_targets.size(1)
    ranks = torch.arange(1, L + 1, device=preds.device)
    discounts = 1.0 / torch.log2(ranks + 1)

    dcg = (sorted_targets * discounts).sum(dim=1)

    ideal_sorted = torch.sort(targets, dim=1, descending=True).values[:, :L]
    idcg = (ideal_sorted * discounts).sum(dim=1)

    ndcg = dcg / torch.clamp(idcg, min=1e-8)

    return ndcg.mean()


class NNEnsembleBackend(backend.AnnifLearningBackend, ensemble.BaseEnsembleBackend):
    """Neural network ensemble backend that combines results from multiple
    projects"""

    name = "nn_ensemble"

    MODEL_FILE = "nn-model.pt"
    LMDB_FILE = "nn-train.mdb"

    EVAL_BATCH_SIZE = 512
    EARLY_STOPPING_PATIENCE = 2
    EARLY_STOP_EVAL_ROWS = 512
    EARLY_STOP_SEED = 1337
    PRED_SCALE = 20

    DEFAULT_PARAMETERS = {
        "lr": 0.003,
        "max-epochs": 50,
        "learn-epochs": 1,
        "batch-size": 256,
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
            message = (
                f"loading model from {model_filename}; "
                f'original error message: "{err}"'
            )
            raise OperationFailedException(message, backend_id=self.backend_id)

    def _merge_source_batches(
        self,
        batch_by_source: dict[str, SuggestionBatch],
        sources: list[tuple[str, float]],
        params: dict[str, Any],
    ) -> SuggestionBatch:
        score_vectors = np.array(
            [
                [suggestions.as_vector() for suggestions in batch]
                for project_id, batch in batch_by_source.items()
            ],
            dtype=np.float32,
        )
        score_vector_tensor = torch.log1p(
            torch.from_numpy(score_vectors.swapaxes(0, 1))
        )
        with torch.no_grad():
            prediction = self._model(score_vector_tensor)
        # use sigmoid to ensure [0..1] range, scaled to spread out values
        scaled_pred = torch.sigmoid(prediction * self.PRED_SCALE)
        return SuggestionBatch.from_sequence(
            [
                vector_to_suggestions(row, limit=int(params["limit"]))
                for row in scaled_pred.detach().numpy()
            ],
            self.project.subjects,
        )

    def _create_model(self, sources: list[tuple[str, float]]) -> None:
        self.info("creating NN ensemble model")

        # Create PyTorch model

        self._model = NNEnsembleModel(
            n_sources=len(sources),
            n_subjects=len(self.project.subjects),
            source_weights=[src[1] for src in sources],
        )

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
            max_epochs=int(params["max-epochs"]),
            batch_size=int(params["batch-size"]),
            lr=float(params["lr"]),
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
                score_vector = np.array(
                    [p_hits.as_vector() for project_id, p_hits in hits.items()],
                    dtype=np.float32,
                )
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
        max_epochs: int,
        batch_size: int,
        lr: float,
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
        self.info("Training model...")
        with env.begin(buffers=True) as txn:
            dataset = LMDBDataset(txn)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            # Deterministic eval subset for early stopping
            rng = np.random.default_rng(self.EARLY_STOP_SEED)
            n_samples = len(dataset)
            n_eval = min(self.EARLY_STOP_EVAL_ROWS, n_samples)
            eval_indices = rng.choice(n_samples, size=n_eval, replace=False)
            eval_inputs, eval_targets = dataset.get_subset(eval_indices.tolist())

            # Training loop
            optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=lr,
                weight_decay=0.0,
                eps=1e-08,
            )
            criterion = nn.BCEWithLogitsLoss()
            early_stopping = EarlyStopping(patience=self.EARLY_STOPPING_PATIENCE)

            for epoch in range(max_epochs):
                self._model.train()
                tqdm_loader = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch + 1}/{max_epochs}",
                )
                for inputs, targets in tqdm_loader:
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                # evaluate for early stopping
                with torch.no_grad():
                    outputs = self._model(eval_inputs)
                ndcg = ndcg_batch(outputs, eval_targets)
                self.info(f"Epoch {epoch + 1}/{max_epochs}: NDCG={ndcg:.4f}")
                if early_stopping(self._model, ndcg, epoch):
                    best = early_stopping.best_epoch + 1
                    self.info(f"Model no longer improving, using best epoch {best}.")
                    break

            # Restore best model weights
            self._model.load_state_dict(early_stopping.best_state)

        annif.util.atomic_save(self._model, self.datadir, self.MODEL_FILE)

    def _learn(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
    ) -> None:
        self.initialize()
        self._fit_model(
            corpus,
            max_epochs=int(params["learn-epochs"]),
            batch_size=int(params["batch-size"]),
            lr=float(params["lr"]),
            lmdb_map_size=int(params["lmdb_map_size"]),
        )
