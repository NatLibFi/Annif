"""Backend that returns most similar subjects based on similarity in LSI
vector space"""

import os.path
import gensim.similarities
from gensim.matutils import Sparse2Corpus
from gensim.models import LsiModel
import annif.util
from annif.hit import VectorAnalysisResult
from annif.exception import NotInitializedException
from . import backend


class LSIBackend(backend.AnnifBackend):
    """TF-IDF vector space similarity based backend for Annif"""
    name = "lsi"
    needs_subject_index = True
    needs_subject_vectorizer = True

    # defaults for uninitialized instances
    _lsi = None
    _index = None

    MODEL_FILE = 'lsi-model'
    INDEX_FILE = 'lsi-index'

    def initialize(self):
        if self._lsi is None:
            path = os.path.join(self._get_datadir(), self.MODEL_FILE)
            self.debug('loading LSI model from {}'.format(path))
            if os.path.exists(path):
                self._lsi = LsiModel.load(path)
            else:
                raise NotInitializedException(
                    'LSI model {} not found'.format(path),
                    backend_id=self.backend_id)
        if self._index is None:
            path = os.path.join(self._get_datadir(), self.INDEX_FILE)
            self.debug('loading similarity index from {}'.format(path))
            if os.path.exists(path):
                self._index = gensim.similarities.MatrixSimilarity.load(path)
            else:
                raise NotInitializedException(
                    'similarity index {} not found'.format(path),
                    backend_id=self.backend_id)

    def load_corpus(self, corpus, project):
        self.info('creating LSI model')
        veccorpus = project.vectorizer.transform(
            (subj.text for subj in corpus.subjects))
        gscorpus = Sparse2Corpus(veccorpus, documents_columns=False)
        self._lsi = LsiModel(
            gscorpus,
            num_topics=int(self.params['num_topics']))
        annif.util.atomic_save(
            self._lsi,
            self._get_datadir(),
            self.MODEL_FILE)
        self.info('creating similarity index')
        self._index = gensim.similarities.MatrixSimilarity(
            self._lsi[gscorpus])
        annif.util.atomic_save(
            self._index,
            self._get_datadir(),
            self.INDEX_FILE)

    def _analyze(self, text, project, params):
        self.initialize()
        self.debug('Analyzing text "{}..." (len={})'.format(
            text[:20], len(text)))
        vectors = project.vectorizer.transform([text])
        corpus = Sparse2Corpus(vectors, documents_columns=False)
        lsi_vector = self._lsi[corpus]
        docsim = self._index[lsi_vector[0]]
        fullresult = VectorAnalysisResult(docsim, project.subjects)
        return fullresult.filter(limit=int(self.params['limit']))
