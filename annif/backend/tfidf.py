"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import tempfile
import annif.util
from annif.suggestion import VectorSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend
from . import mixins

# Filter UserWarnings due to not-installed python-Levenshtein package
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import gensim.similarities
    from gensim.matutils import Sparse2Corpus


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


class TFIDFBackend(mixins.TfidfVectorizerMixin, backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""
    name = "tfidf"
    needs_subject_index = True

    # defaults for uninitialized instances
    _index = None

    INDEX_FILE = 'tfidf-index'

    def _generate_subjects_from_documents(self, corpus):
        with tempfile.TemporaryDirectory() as tempdir:
            subject_buffer = {}
            for subject_id in range(len(self.project.subjects)):
                subject_buffer[subject_id] = SubjectBuffer(tempdir,
                                                           subject_id)

            for doc in corpus.documents:
                tokens = self.project.analyzer.tokenize_words(doc.text)
                for uri in doc.uris:
                    subject_id = self.project.subjects.by_uri(uri)
                    if subject_id is None:
                        continue
                    subject_buffer[subject_id].write(" ".join(tokens))

            for sid in range(len(self.project.subjects)):
                yield subject_buffer[sid].read()

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
        self.initialize_vectorizer()
        self._initialize_index()

    def _create_index(self, veccorpus):
        self.info('creating similarity index')
        gscorpus = Sparse2Corpus(veccorpus, documents_columns=False)
        self._index = gensim.similarities.SparseMatrixSimilarity(
            gscorpus, num_features=len(self.vectorizer.vocabulary_))
        annif.util.atomic_save(
            self._index,
            self.datadir,
            self.INDEX_FILE)

    def _train(self, corpus, params):
        if corpus == 'cached':
            raise NotSupportedException(
                'Training tfidf project from cached data not supported.')
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train tfidf project with no documents')
        self.info('transforming subject corpus')
        subjects = self._generate_subjects_from_documents(corpus)
        veccorpus = self.create_vectorizer(subjects)
        self._create_index(veccorpus)

    def _suggest(self, text, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        tokens = self.project.analyzer.tokenize_words(text)
        vectors = self.vectorizer.transform([" ".join(tokens)])
        docsim = self._index[vectors[0]]
        fullresult = VectorSuggestionResult(docsim)
        return fullresult.filter(self.project.subjects,
                                 limit=int(params['limit']))
