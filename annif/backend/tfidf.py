"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import gensim.similarities
from gensim.matutils import Sparse2Corpus
import annif.util
from annif.hit import VectorAnalysisResult
from annif.exception import NotInitializedException
from . import backend


class TFIDFBackend(backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""
    name = "tfidf"
    needs_subject_index = True
    needs_subject_vectorizer = True

    # defaults for uninitialized instances
    _index = None

    INDEX_FILE = 'tfidf-index'

    def initialize(self):
        if self._index is None:
            path = os.path.join(self._get_datadir(), self.INDEX_FILE)
            self.debug('loading similarity index from {}'.format(path))
            if os.path.exists(path):
                self._index = gensim.similarities.SparseMatrixSimilarity.load(
                    path)
            else:
                raise NotInitializedException(
                    'similarity index {} not found'.format(path),
                    backend_id=self.backend_id)

    def load_corpus(self, corpus, project):
        self.info('creating similarity index')
        veccorpus = project.vectorizer.transform(
            (subj.text for subj in corpus.subjects))
        gscorpus = Sparse2Corpus(veccorpus, documents_columns=False)
        self._index = gensim.similarities.SparseMatrixSimilarity(
            gscorpus, num_features=len(project.vectorizer.vocabulary_))
        annif.util.atomic_save(
            self._index,
            self._get_datadir(),
            self.INDEX_FILE)

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        vectors = project.vectorizer.transform([text])
        docsim = self._index[vectors[0]]
        fullresult = VectorAnalysisResult(docsim, project.subjects)
        return fullresult.filter(limit=int(self.params['limit']))
