"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os
import os.path
import tempfile
import gensim.corpora
from . import backend


class TFIDFBackend(backend.AnnifBackend):
    name = "tfidf"

    def _atomic_save(self, obj, dirname, filename):
        tempfd, tempfilename = tempfile.mkstemp(prefix=filename, dir=dirname)
        os.close(tempfd)
        obj.save(tempfilename)
        os.rename(tempfilename, os.path.join(dirname, filename))

    def load_subjects(self, subjects, analyzer):
        corpus = subjects.tokens(analyzer)
        dictionary = gensim.corpora.Dictionary(corpus)
        self._atomic_save(dictionary, self._get_datadir(), 'dictionary')

    def analyze(self, text):
        return []  # TODO
