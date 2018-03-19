"""Backend that returns most similar subjects based on similarity in sparse
TF-IDF normalized bag-of-words vector space"""

import os.path
import gensim.corpora
from . import backend


class TFIDFBackend(backend.AnnifBackend):
    name = "tfidf"

    def load_subjects(self, subjects, analyzer):
        corpus = subjects.tokens(analyzer)
        dictionary = gensim.corpora.Dictionary(corpus)
        dictionary.save(os.path.join(self._get_datadir(), 'dictionary'))

    def analyze(self, text):
        return []  # TODO
