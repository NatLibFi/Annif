"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""

import simplemma
from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param, **kwargs):
        self.lang = param
        self.langdata = simplemma.load_data(self.lang)
        super().__init__(**kwargs)

    def _normalize_word(self, word):
        return simplemma.lemmatize(word, self.langdata)
