"""Annif backend mixins that can be used to implement features"""

from __future__ import annotations

import abc
import os.path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import scipy.sparse as sp
from pecos.utils.featurization.text.vectorizers import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import annif.util
from annif.exception import NotInitializedException

if TYPE_CHECKING:
    from collections.abc import Iterable

    from scipy.sparse._csr import csr_matrix

    from annif.corpus import Document
    from annif.suggestion import SubjectSuggestion


class ChunkingBackend(metaclass=abc.ABCMeta):
    """Annif backend mixin that implements chunking of input"""

    DEFAULT_PARAMETERS = {"chunksize": 1}

    def default_params(self) -> dict[str, Any]:
        return self.DEFAULT_PARAMETERS

    @abc.abstractmethod
    def _suggest_chunks(
        self, chunktexts: list[str], params: dict[str, Any]
    ) -> list[SubjectSuggestion]:
        """Suggest subjects for the chunked text; should be implemented by
        the subclass inheriting this mixin"""

        pass  # pragma: no cover

    def _suggest(
        self, doc: Document, params: dict[str, Any]
    ) -> list[SubjectSuggestion]:
        self.debug(
            'Suggesting subjects for text "{}..." (len={})'.format(
                doc.text[:20], len(doc.text)
            )
        )
        sentences = self.project.analyzer.tokenize_sentences(doc.text)
        self.debug("Found {} sentences".format(len(sentences)))
        chunksize = int(params["chunksize"])
        chunktexts = []
        for i in range(0, len(sentences), chunksize):
            chunktexts.append(" ".join(sentences[i : i + chunksize]))
        self.debug("Split sentences into {} chunks".format(len(chunktexts)))
        if len(chunktexts) == 0:  # no input, empty result
            return []
        return self._suggest_chunks(chunktexts, params)

class TfidfVectorizerMixin:
    """Annif backend mixin that implements TfidfVectorizer functionality"""

    VECTORIZER_FILE = "vectorizer"

    vectorizer = None

    def initialize_vectorizer(self) -> None:
        if self.vectorizer is None:
            path = os.path.join(self.datadir, self.VECTORIZER_FILE)
            if os.path.exists(path):
                self.debug("loading vectorizer from {}".format(path))
                self.vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    backend_id=self.backend_id,
                )

    def create_vectorizer(
        self, input: Iterable[str], params: dict[str, Any] = None
    ) -> csr_matrix:
        self.info("creating vectorizer")
        if params is None:
            params = {}
        # avoid UserWarning when overriding tokenizer
        if "tokenizer" in params:
            params["token_pattern"] = None
        self.vectorizer = TfidfVectorizer(**params)
        veccorpus = self.vectorizer.fit_transform(input)
        annif.util.atomic_save(
            self.vectorizer, self.datadir, self.VECTORIZER_FILE, method=joblib.dump
        )
        return veccorpus

class PecosTfidfVectorizerMixin:
    """Annif backend mixin that implements TfidfVectorizer functionality from Pecos"""

    VECTORIZER_FILE = "vectorizer"

    vectorizer = None

    def initialize_vectorizer(self) -> None:
        if self.vectorizer is None:
            path = os.path.join(self.datadir, self.VECTORIZER_FILE)
            if os.path.exists(path):
                self.debug("loading vectorizer from {}".format(path))
                self.vectorizer = Vectorizer.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    backend_id=self.backend_id,
                )
            
    def vectorizer_dict(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a vectorizer configuration dictionary from the given parameters."""
        
        config = {
            "base_vect_configs": [
                {
                    "ngram_range": params.get("ngram_range", [1, 1]),
                    "max_df_ratio": 0.98,
                    "analyzer": "word",
                    "min_df_cnt": params.get("min_df", 1),
                }
            ]
        }
        return {"type": "tfidf", "kwargs": {**config}} 
    

    def create_vectorizer(
        self, input: Iterable[str], params: dict[str, Any] = None
    ) -> csr_matrix:

        self.info("creating Pecos vectorizer")
        if params is None:
            params = {}
        data = list(input)
        vectorizer_config = self.vectorizer_dict(params)
        self.vectorizer = Vectorizer.train(data, vectorizer_config, np.float32)
        self.vectorizer.save(os.path.join(self.datadir, self.VECTORIZER_FILE))
        veccorpus = self.vectorizer.predict(
            data,
            threads=params.get("threads", -1)
        )

        # # Fix scikit-learn requirement: enforce int32 indices
        # if sp.issparse(veccorpus):
        #     if veccorpus.indices.dtype != np.int32:
        #         veccorpus.indices = veccorpus.indices.astype(np.int32, copy=False)
        #     if veccorpus.indptr.dtype != np.int32:
        #         veccorpus.indptr = veccorpus.indptr.astype(np.int32, copy=False)
        
        return veccorpus
