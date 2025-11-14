import os
from typing import Any

import joblib
import numpy as np
from ebm4subjects.ebm_model import EbmModel

from annif.analyzer.analyzer import Analyzer
from annif.corpus.document import Document, DocumentCorpus
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SuggestionBatch, vector_to_suggestions
from annif.util import atomic_save

from . import backend


class EbmBackend(backend.AnnifBackend):
    name = "ebm"

    EBM_PARAMETERS = {
        "embedding_model_name": str,
        "embedding_dimensions": int,
        "max_chunk_count": int,
        "max_chunk_length": int,
        "chunking_jobs": int,
        "max_sentence_count": int,
        "hnsw_index_params": dict[str, Any],
        "candidates_per_chunk": int,
        "candidates_per_doc": int,
        "query_jobs": int,
        "xgb_shrinkage": float,
        "xgb_interaction_depth": int,
        "xgb_subsample": float,
        "xgb_rounds": int,
        "xgb_jobs": int,
        "duck_db_threads": int,
        "use_altLabels": bool,
        "model_args": dict[str, Any],
        "encode_args_vocab": dict[str, Any],
        "encode_args_documents": dict[str, Any],
    }

    DEFAULT_PARAMETERS = {
        "embedding_model_name": "BAAI/bge-m3",
        "embedding_dimensions": 1024,
        "max_chunk_count": 100,
        "max_chunk_length": 50,
        "chunking_jobs": 1,
        "max_sentence_count": 100,
        "hnsw_index_params": {"M": 32, "ef_construction": 256, "ef_search": 256},
        "candidates_per_chunk": 20,
        "candidates_per_doc": 100,
        "query_jobs": 1,
        "xgb_shrinkage": 0.03,
        "xgb_interaction_depth": 5,
        "xgb_subsample": 0.7,
        "xgb_rounds": 300,
        "xgb_jobs": 1,
        "duckdb_threads": 1,
        "use_altLabels": True,
        "model_args": {"device": "cpu", "trust_remote_code": False},
        "encode_args_vocab": {"batch_size": 32, "show_progress_bar": True},
        "encode_args_documents": {"batch_size": 32, "show_progress_bar": True},
    }

    DB_FILE = "ebm-duck.db"
    MODEL_FILE = "ebm-model.gz"
    TRAIN_FILE = "ebm-train.gz"

    _analyzer = Analyzer()

    _model = None

    def initialize(self, parallel: bool = False) -> None:
        if self._model is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)

            self.debug(f"loading model from {path}")
            if os.path.exists(path):
                self._model = EbmModel.load(path)
                self._model.init_logger(logger=self)
                self.debug("loaded model")
            else:
                raise NotInitializedException(
                    f"model not found at {path}", backend_id=self.backend_id
                )

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        self.info("starting train")
        self._model = EbmModel(
            db_path=os.path.join(self.datadir, self.DB_FILE),
            embedding_model_name=params["embedding_model_name"],
            embedding_dimensions=params["embedding_dimensions"],
            chunk_tokenizer=self._analyzer,
            max_chunk_count=params["max_chunk_count"],
            max_chunk_length=params["max_chunk_length"],
            chunking_jobs=params["chunking_jobs"],
            max_sentence_count=params["max_sentence_count"],
            hnsw_index_params=params["hnsw_index_params"],
            candidates_per_chunk=params["candidates_per_chunk"],
            candidates_per_doc=params["candidates_per_doc"],
            query_jobs=params["query_jobs"],
            xgb_shrinkage=params["xgb_shrinkage"],
            xgb_interaction_depth=params["xgb_interaction_depth"],
            xgb_subsample=params["xgb_subsample"],
            xgb_rounds=params["xgb_rounds"],
            xgb_jobs=params["xgb_jobs"],
            duckdb_threads=jobs if jobs else params["duckdb_threads"],
            use_altLabels=params["use_altLabels"],
            model_args=params["model_args"],
            encode_args_vocab=params["encode_args_vocab"],
            encode_args_documents=params["encode_args_documents"],
            logger=self,
        )

        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    f"training backend {self.backend_id} with no documents"
                )

            self.info("creating vector database")
            self._model.create_vector_db(
                vocab_in_path=os.path.join(
                    self.project.vocab.datadir, self.project.vocab.INDEX_FILENAME_TTL
                ),
                force=True,
            )

            self.info("preparing training data")
            doc_ids = []
            texts = []
            label_ids = []
            for doc_id, doc in enumerate(corpus.documents):
                for subject_id in [
                    subject_id for subject_id in getattr(doc, "subject_set")
                ]:
                    doc_ids.append(doc_id)
                    texts.append(getattr(doc, "text"))
                    label_ids.append(self.project.subjects[subject_id].uri)

            train_data = self._model.prepare_train(
                doc_ids=doc_ids,
                label_ids=label_ids,
                texts=texts,
                n_jobs=jobs,
            )

            atomic_save(
                obj=train_data,
                dirname=self.datadir,
                filename=self.TRAIN_FILE,
                method=joblib.dump,
            )

        else:
            self.info("reusing cached training data from previous run")
            if not os.path.exists(self._model.db_path):
                raise NotInitializedException(
                    f"database file {self._model.db_path} not found",
                    backend_id=self.backend_id,
                )
            if not os.path.exists(os.path.join(self.datadir, self.TRAIN_FILE)):
                raise NotInitializedException(
                    f"train data file {self.TRAIN_FILE} not found",
                    backend_id=self.backend_id,
                )

            train_data = joblib.load(os.path.join(self.datadir, self.TRAIN_FILE))

        self.info("training model")
        self._model.train(train_data, jobs)

        self.info("saving model")
        atomic_save(self._model, self.datadir, self.MODEL_FILE)

    def _suggest_batch(
        self, documents: list[Document], params: dict[str, Any]
    ) -> SuggestionBatch:
        candidates = self._model.generate_candidates_batch(
            texts=[doc.text for doc in documents],
            doc_ids=[i for i in range(len(documents))],
        )

        predictions = self._model.predict(candidates)

        suggestions = []
        for doc_predictions in predictions:
            vector = np.zeros(len(self.project.subjects), dtype=np.float32)
            for row in doc_predictions.iter_rows(named=True):
                position = self.project.subjects._uri_idx.get(row["label_id"], 0)
                vector[position] = row["score"]
            suggestions.append(vector_to_suggestions(vector, int(params["limit"])))

        return SuggestionBatch.from_sequence(
            suggestions,
            self.project.subjects,
            limit=int(params.get("limit")),
        )
