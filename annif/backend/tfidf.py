"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os
import os.path
import tempfile
import gensim.corpora
import gensim.models
import gensim.similarities
from . import backend


class VectorCorpus:
    """A class that wraps a text corpus so it can be iterated as lists of
    vectors, by using a dictionary to map words to integers."""

    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dictionary = dictionary

    def __iter__(self):
        for doc in self.corpus:
            yield self.dictionary.doc2bow(doc)


class TFIDFBackend(backend.AnnifBackend):
    name = "tfidf"

    # defaults for uninitialized instances
    _dictionary = None
    _tfidf = None
    _index = None

    def _atomic_save(self, obj, dirname, filename):
        tempfd, tempfilename = tempfile.mkstemp(prefix=filename, dir=dirname)
        os.close(tempfd)
        obj.save(tempfilename)
        os.rename(tempfilename, os.path.join(dirname, filename))

    def load_subjects(self, subjects, analyzer):
        corpus = subjects.tokens(analyzer)
        self._dictionary = gensim.corpora.Dictionary(corpus)
        self._atomic_save(self._dictionary, self._get_datadir(), 'dictionary')
        veccorpus = VectorCorpus(corpus, self._dictionary)
        self._tfidf = gensim.models.TfidfModel(veccorpus)
        self._atomic_save(self._tfidf, self._get_datadir(), 'tfidf')
        self._index = gensim.similarities.SparseMatrixSimilarity(
            self._tfidf[veccorpus], num_features=len(self._dictionary))
        self._atomic_save(self._index, self._get_datadir(), 'index')

    def analyze(self, text):
        return []  # TODO
