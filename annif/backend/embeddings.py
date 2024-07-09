"""TODO"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Any

import numpy as np
import tiktoken
from openai import AzureOpenAI  # Try using huggingface client

# import annif.util
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
    """Semantic vector space similarity based backend for Annif"""

    name = "embeddings"
    _index = None

    INDEX_FILE = "emdeddings-index.npy"
    BASE_MODEL = "text-embedding-3-large"
    VECTOR_DIMENSIONS = 3072  # For text-embedding-3-large
    MAX_TOKENS = 8192  # For text-embedding-3-large

    encoding = tiktoken.encoding_for_model(BASE_MODEL)

    def _initialize_index(self) -> None:
        if self._index is None:
            path = os.path.join(self.datadir, self.INDEX_FILE)
            self.debug("loading similarity index from {}".format(path))
            if os.path.exists(path):
                self._index = np.load(path, allow_pickle=True)
            else:
                raise NotInitializedException(
                    "similarity index {} not found".format(path),
                    backend_id=self.backend_id,
                )

    def initialize(
        self,
        parallel: bool = False,
    ) -> None:
        self.vectorizer = Vectorizer(self.params["endpoint"], self.params["model"])
        self._initialize_index()

    def _create_index(self, corpus) -> None:
        self.vectorizer = Vectorizer(self.params["endpoint"], self.params["model"])
        self.info("creating similarity index")
        path = os.path.join(self.datadir, self.INDEX_FILE)

        subject_vectors = np.zeros((len(self.project.subjects), self.VECTOR_DIMENSIONS))
        for doc in corpus.documents:
            vec = self.vectorizer.transform(self._truncate_text(doc.text))
            for sid in doc.subject_set:
                subject_vectors[sid, :] = subject_vectors[sid, :] + vec

        row_norms = np.linalg.norm(subject_vectors, axis=1, keepdims=True)

        # Avoid division by zero: Only normalize non-zero rows
        self._index = np.where(row_norms == 0, 0, subject_vectors / row_norms)
        np.save(path, self._index, allow_pickle=True)

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
        truncated_text = self._truncate_text(text)
        vector = self.vectorizer.transform(truncated_text)

        cosine_similarity = np.dot(self._index, np.array(vector))
        return vector_to_suggestions(cosine_similarity, int(params["limit"]))
