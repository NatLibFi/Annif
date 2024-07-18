"""TODO"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Any

import numpy as np
import tiktoken
from openai import AzureOpenAI  # Try using huggingface client
from qdrant_client import QdrantClient, models

from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import vector_to_suggestions

from . import backend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annif.corpus.document import DocumentCorpus


class Vectorizer:
    def __init__(self, endpoint, model):
        self.model = model
        self.client = AzureOpenAI(  # TODO Try AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
        )

    def transform(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            # dimensions=dimensions,  # TODO Try with reduced dimensions
        )
        return response.data[0].embedding


class EmbeddingsBackend(backend.AnnifBackend):
    """TODO xxx cector space similarity based backend for Annif"""

    name = "embeddings"
    is_trained = True

    # defaults for uninitialized instances
    _index = None

    DB_FILE = "qdrant-db"
    COLLECTION_NAME = "index-collection"
    BASE_MODEL = "text-embedding-3-large"
    VECTOR_DIMENSIONS = 3072  # For text-embedding-3-large
    MAX_TOKENS = 8192  # For text-embedding-3-large

    encoding = tiktoken.encoding_for_model(BASE_MODEL)

    def _initialize_index(self) -> None:
        if self._index is None:
            path = os.path.join(self.datadir, self.DB_FILE)
            self.debug("loading similarity index from {}".format(path))
            if os.path.exists(path):
                self.qdclient = QdrantClient(path=path)
            else:
                raise NotInitializedException(
                    "similarity index {} not found".format(path),
                    backend_id=self.backend_id,
                )

    def initialize(
        self,
    ) -> None:
        self.vectorizer = Vectorizer(self.params["endpoint"], self.params["model"])
        self._initialize_index()

    def _create_index(self, corpus) -> None:
        self.vectorizer = Vectorizer(self.params["endpoint"], self.params["model"])
        self.info("creating similarity index")
        path = os.path.join(self.datadir, self.DB_FILE)

        self.qdclient = QdrantClient(path=path)
        self.qdclient.recreate_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=self.VECTOR_DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
        )

        veccorpus = (
            (
                doc.subject_set,
                self.vectorizer.transform(self._truncate_text(" ".join(doc.text))),
            )
            for doc in corpus.documents
        )

        subject_sets, vectors = zip(*veccorpus)
        payloads = [{"subjects": [sid for sid in ss]} for ss in subject_sets]
        ids = list(range(len(vectors)))
        self.qdclient.upsert(
            collection_name=self.COLLECTION_NAME,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            ),
        )
        print(self.qdclient.get_collection(collection_name=self.COLLECTION_NAME))

    def _truncate_text(self, text):
        """truncate text so it contains at most MAX_TOKENS according to the OpenAI
        tokenizer"""
        tokens = self.encoding.encode(text)
        return self.encoding.decode(tokens[: self.MAX_TOKENS])

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus == "cached":
            raise NotSupportedException(
                "Training embeddings project from cached data not supported."
            )
        if corpus.is_empty():
            raise NotSupportedException(
                "Cannot train embeddings project with no documents"
            )
        self.info("transforming subject corpus")
        self._create_index(corpus)

    def _suggest(self, text: str, params: dict[str, Any]) -> Iterator:
        self.debug(
            'Suggesting subjects for text "{}..." (len={})'.format(text[:20], len(text))
        )
        truncated_text = self._truncate_text(" ".join(text))
        vector = self.vectorizer.transform(truncated_text)
        # print(vector[:5])
        info = self.qdclient.get_collection(collection_name=self.COLLECTION_NAME)
        self.debug(f"Collection info: {info}")
        results = self._search(vector, params)
        # print(results)
        return self._prediction_to_result(results, params)

    def _search(self, vector, params):
        result = self.qdclient.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=vector,
            # score_threshold=1.0,  # TODO parameterize this
            limit=int(params["limit"]),
            # search_params=models.SearchParams(hnsw_ef=128, exact=False),  # TODO This
        )
        return [(sp.payload["subjects"], sp.score) for sp in result]

    def _combine_search_results(self, results):
        combined = []
        for res in results:
            sids, weight = res[0], res[1]
            combined.extend([sid * weight for sid in sids])
        return combined

    # From backend/mllm.py
    def _prediction_to_result(
        self,
        results,
        params,
    ) -> Iterator:
        vector = np.zeros(len(self.project.subjects), dtype=np.float32)
        for subject_ids, score in results:
            for sid in subject_ids:
                vector[sid] += score
        return vector_to_suggestions(vector, int(params["limit"]))
