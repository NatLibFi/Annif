"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import joblib
import gensim.similarities
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import annif.util
from annif.suggestion import VectorSuggestionResult
from annif.exception import NotInitializedException, NotSupportedException
from . import backend


class TFIDFBackend(backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""
    name = "tfidf"
    needs_subject_index = True

    # defaults for uninitialized instances
    _vectorizer = None
    _index = None

    VECTORIZER_FILE = 'vectorizer'
    INDEX_FILE = 'tfidf-index'

    def _create_vectorizer(self, corpus, project):
        if corpus.is_empty():
            raise NotSupportedException(
                'Cannot train tfidf project with no documents')
        self.info('transforming subject corpus')
        subjects = corpus.subjects
        self.info('creating vectorizer')
        self._vectorizer = TfidfVectorizer(
            tokenizer=project.analyzer.tokenize_words)
        self._vectorizer.fit((subj.text for subj in subjects))
        annif.util.atomic_save(
            self._vectorizer,
            self.datadir,
            'vectorizer',
            method=joblib.dump)

    def initialize(self):
        if self._vectorizer is None:
            path = os.path.join(self.datadir, self.VECTORIZER_FILE)
            if os.path.exists(path):
                self.debug('loading vectorizer from {}'.format(path))
                self._vectorizer = joblib.load(path)
            else:
                raise NotInitializedException(
                    "vectorizer file '{}' not found".format(path),
                    backend_id=self.backend_id)
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

    def train(self, corpus, project):
        self._create_vectorizer(corpus, project)
        self.info('creating similarity index')
        veccorpus = self._vectorizer.transform(
            (subj.text for subj in corpus.subjects))
        gscorpus = Sparse2Corpus(veccorpus, documents_columns=False)
        self._index = gensim.similarities.SparseMatrixSimilarity(
            gscorpus, num_features=len(self._vectorizer.vocabulary_))
        annif.util.atomic_save(
            self._index,
            self.datadir,
            self.INDEX_FILE)

    def _suggest(self, text, project, params):
        self.debug('Suggesting subjects for text "{}..." (len={})'.format(
            text[:20], len(text)))
        vectors = self._vectorizer.transform([text])
        docsim = self._index[vectors[0]]
        fullresult = VectorSuggestionResult(docsim, project.subjects)
        return fullresult.filter(limit=int(self.params['limit']))
