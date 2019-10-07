"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import tempfile
import joblib
import gensim.similarities
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import annif.util
from annif.suggestion import VectorSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend


class SubjectBuffer:
    """A file-backed buffer to store and retrieve subject text."""

    BUFFER_SIZE = 100

    def __init__(self, tempdir, subject_id):
        filename = '{:08d}.txt'.format(subject_id)
        self._path = os.path.join(tempdir, filename)
        self._buffer = []
        self._created = False

    def flush(self):
        if self._created:
            mode = 'a'
        else:
            mode = 'w'

        with open(self._path, mode, encoding='utf-8') as subjfile:
            for text in self._buffer:
                print(text, file=subjfile)

        self._buffer = []
        self._created = True

    def write(self, text):
        self._buffer.append(text)
        if len(self._buffer) >= self.BUFFER_SIZE:
            self.flush()

    def read(self):
        if not self._created:
            # file was never created - we can simply return the buffer content
            return "\n".join(self._buffer)
        else:
            with open(self._path, 'r', encoding='utf-8') as subjfile:
                return subjfile.read() + "\n" + "\n".join(self._buffer)


class TFIDFBackend(backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""
    name = "tfidf"
    needs_subject_index = True

    # defaults for uninitialized instances
    _vectorizer = None
    _index = None

    VECTORIZER_FILE = 'vectorizer'
    INDEX_FILE = 'tfidf-index'

    def _generate_subjects_from_documents(self, corpus, project):
        with tempfile.TemporaryDirectory() as tempdir:
            subject_buffer = {}
            for subject_id in range(len(project.subjects)):
                subject_buffer[subject_id] = SubjectBuffer(tempdir,
                                                           subject_id)

            for doc in corpus.documents:
                tokens = project.analyzer.tokenize_words(doc.text)
                for uri in doc.uris:
                    subject_id = project.subjects.by_uri(uri)
                    if subject_id is None:
                        continue
                    subject_buffer[subject_id].write(" ".join(tokens))

            for sid in range(len(project.subjects)):
                yield subject_buffer[sid].read()

    def _initialize_vectorizer(self):
        if self._vectorizer is None:
            path = os.path.join(self.datadir, self.VECTORIZER_FILE)
            if os.path.exists(path):
                self.debug('loading vectorizer from {}'.format(path))
                self._vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    backend_id=self.backend_id)

    def _initialize_index(self):
        if self._index is None:
            path = os.path.join(self.datadir, self.INDEX_FILE)
            self.debug('loading similarity index from {}'.format(path))
            if os.path.exists(path):
                self._index = gensim.similarities.SparseMatrixSimilarity.load(
                    path)
            else:
                raise NotInitializedException(
                    'similarity index {} not found'.format(path),
                    backend_id=self.backend_id)

    def initialize(self):
        self._initialize_vectorizer()
        self._initialize_index()

    def _create_index(self, veccorpus):
        self.info('creating similarity index')
        gscorpus = Sparse2Corpus(veccorpus, documents_columns=False)
        self._index = gensim.similarities.SparseMatrixSimilarity(
            gscorpus, num_features=len(self._vectorizer.vocabulary_))
        annif.util.atomic_save(
            self._index,
            self.datadir,
            self.INDEX_FILE)

    def train(self, corpus, project):
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train tfidf project with no documents')
        self.info('transforming subject corpus')
        subjects = self._generate_subjects_from_documents(corpus, project)
        self.info('creating vectorizer')
        self._vectorizer = TfidfVectorizer()
        veccorpus = self._vectorizer.fit_transform(subjects)
        annif.util.atomic_save(
            self._vectorizer,
            self.datadir,
            self.VECTORIZER_FILE,
            method=joblib.dump)
        self._create_index(veccorpus)

    def _suggest(self, text, project, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        tokens = project.analyzer.tokenize_words(text)
        vectors = self._vectorizer.transform([" ".join(tokens)])
        docsim = self._index[vectors[0]]
        fullresult = VectorSuggestionResult(docsim, project.subjects)
        return fullresult.filter(limit=int(self.params['limit']))
