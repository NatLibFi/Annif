"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

from __future__ import annotations

import os.path
import tempfile
from typing import TYPE_CHECKING, Any

from scipy.sparse import load_npz, save_npz
from sklearn.preprocessing import normalize

import annif.util
from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import vector_to_suggestions

from . import backend, mixins

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annif.corpus import Document, DocumentCorpus


class SubjectBuffer:
    """A file-backed buffer to store and retrieve subject text."""

    BUFFER_SIZE = 100

    def __init__(self, tempdir: str, subject_id: int) -> None:
        filename = "{:08d}.txt".format(subject_id)
        self._path = os.path.join(tempdir, filename)
        self._buffer = []
        self._created = False

    def flush(self) -> None:
        if self._created:
            mode = "a"
        else:
            mode = "w"

        with open(self._path, mode, encoding="utf-8") as subjfile:
            for text in self._buffer:
                print(text, file=subjfile)

        self._buffer = []
        self._created = True

    def write(self, text: str) -> None:
        self._buffer.append(text)
        if len(self._buffer) >= self.BUFFER_SIZE:
            self.flush()

    def read(self) -> str:
        if not self._created:
            # file was never created - we can simply return the buffer content
            return "\n".join(self._buffer)
        else:
            with open(self._path, "r", encoding="utf-8") as subjfile:
                return subjfile.read() + "\n" + "\n".join(self._buffer)


class TFIDFBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""

    name = "tfidf"

    # defaults for uninitialized instances
    _tfidf_matrix = None

    MATRIX_FILE = "tfidf-matrix.npz"

    def _generate_subjects_from_documents(
        self, corpus: DocumentCorpus
    ) -> Iterator[str]:
        with tempfile.TemporaryDirectory() as tempdir:
            subject_buffer = {}
            for subject_id in range(len(self.project.subjects)):
                subject_buffer[subject_id] = SubjectBuffer(tempdir, subject_id)

            for doc in corpus.documents:
                tokens = self.project.analyzer.tokenize_words(doc.text)
                for subject_id in doc.subject_set:
                    subject_buffer[subject_id].write(" ".join(tokens))

            for sid in range(len(self.project.subjects)):
                yield subject_buffer[sid].read()

    def _initialize_index(self) -> None:
        if self._tfidf_matrix is None:
            path = os.path.join(self.datadir, self.MATRIX_FILE)
            self.debug("loading tf-idf matrix from {}".format(path))
            if os.path.exists(path):
                self._tfidf_matrix = load_npz(path)
            else:
                raise NotInitializedException(
                    "tf-idf matrix {} not found".format(path),
                    backend_id=self.backend_id,
                )

    def initialize(self, parallel: bool = False) -> None:
        self.initialize_vectorizer()
        self._initialize_index()

    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus == "cached":
            raise NotSupportedException(
                "Training tfidf project from cached data not supported."
            )
        if corpus.is_empty():
            raise NotSupportedException("Cannot train tfidf project with no documents")
        self.info("transforming subject corpus")
        subjects = self._generate_subjects_from_documents(corpus)
        self._tfidf_matrix = normalize(self.create_vectorizer(subjects))
        self.info("saving tf-idf matrix")
        annif.util.atomic_save(
            self._tfidf_matrix,
            self.datadir,
            self.MATRIX_FILE,
            lambda obj, filename: save_npz(filename, obj),
        )

    def _suggest(self, doc: Document, params: dict[str, Any]) -> Iterator:
        self.debug(
            'Suggesting subjects for text "{}..." (len={})'.format(
                doc.text[:20], len(doc.text)
            )
        )
        tokens = self.project.analyzer.tokenize_words(doc.text)
        query_vector = normalize(self.vectorizer.transform([" ".join(tokens)]))

        # Compute cosine similarity between query and indexed corpus
        similarities = (query_vector @ self._tfidf_matrix.T).toarray().flatten()

        return vector_to_suggestions(similarities, int(params["limit"]))
