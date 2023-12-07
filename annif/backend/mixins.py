"""Annif backend mixins that can be used to implement features"""
from __future__ import annotations

import abc
import os.path
from typing import TYPE_CHECKING, Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import annif.util
from annif.exception import NotInitializedException

if TYPE_CHECKING:
    from collections.abc import Iterable

    from scipy.sparse._csr import csr_matrix

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

    def _suggest(self, text: str, params: dict[str, Any]) -> list[SubjectSuggestion]:
        self.debug(
            'Suggesting subjects for text "{}..." (len={})'.format(text[:20], len(text))
        )
        sentences = self.project.analyzer.tokenize_sentences(text)
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
