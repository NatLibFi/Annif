"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""

import functools
import simplemma
from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param, **kwargs):
        self.lang = param
        self.langdata = None
        super().__init__(**kwargs)

    def __getstate__(self):
        """Return the state of the object for pickling purposes. The langdata
        field is set to None as it's more efficient to use load_data."""

        return {'lang': self.lang, 'langdata': None}

    @functools.lru_cache(maxsize=500000)
    def _normalize_word(self, word):
        if self.langdata is None:
            self.langdata = simplemma.load_data(self.lang)
        return simplemma.lemmatize(word, self.langdata)
