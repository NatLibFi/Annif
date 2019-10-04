"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import tempfile
import joblib
import gensim.similarities
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import annif.util
from annif.corpus.subject import SubjectDirectory
from annif.suggestion import VectorSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend


class SubjectWriter:
    """Writes a single subject file into a SubjectDirectory, performing
    buffering to limit the number of I/O operations."""

    _buffer = None

    BUFFER_SIZE = 100

    def __init__(self, path, uri, label):
        self._path = path
        self._buffer = ["{} {}".format(uri, label)]
        self._created = False

    def _flush(self):
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
            self._flush()

    def close(self):
        self._flush()


class TFIDFBackend(backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""
    name = "tfidf"
    needs_subject_index = True

    # defaults for uninitialized instances
    _vectorizer = None
    _index = None

    VECTORIZER_FILE = 'vectorizer'
    INDEX_FILE = 'tfidf-index'

    _temp_directory = None
    _subject_writer = None

    def _subject_filename(self, subject_id):
        filename = '{:08d}.txt'.format(subject_id)
        return os.path.join(self._temp_directory.name, filename)

    def _create_subject(self, subject_id, uri, label):
        filename = self._subject_filename(subject_id)
        self._subject_writer[subject_id] = SubjectWriter(filename, uri, label)

    def _add_text_to_subject(self, subject_id, text):
        self._subject_writer[subject_id].write(text)

    def _generate_subjects_from_documents(self, corpus, project):
        self._temp_directory = tempfile.TemporaryDirectory()
        self._subject_writer = {}

        for subject_id, subject_info in enumerate(project.subjects):
            uri, label = subject_info
            self._create_subject(subject_id, uri, label)

        for doc in corpus.documents:
            for uri in doc.uris:
                subject_id = project.subjects.by_uri(uri)
                if subject_id is None:
                    continue
                self._add_text_to_subject(subject_id, doc.text)

        for subject_id, _ in enumerate(project.subjects):
            self._subject_writer[subject_id].close()

        return SubjectDirectory(self._temp_directory.name)

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
        subjects = self._generate_subjects_from_documents(
            corpus, project).subjects
        self.info('creating vectorizer')
        self._vectorizer = TfidfVectorizer(
            tokenizer=project.analyzer.tokenize_words)
        veccorpus = self._vectorizer.fit_transform(
            (subj.text for subj in subjects))
        annif.util.atomic_save(
            self._vectorizer,
            self.datadir,
            self.VECTORIZER_FILE,
            method=joblib.dump)
        self._create_index(veccorpus)

    def _suggest(self, text, project, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        vectors = self._vectorizer.transform([text])
        docsim = self._index[vectors[0]]
        fullresult = VectorSuggestionResult(docsim, project.subjects)
        return fullresult.filter(limit=int(self.params['limit']))
