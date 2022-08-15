"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""

import functools
import simplemma
from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param, **kwargs):
        self.lang = param
        super().__init__(**kwargs)

    @functools.lru_cache(maxsize=500000)
    def _normalize_word(self, word):
        return simplemma.lemmatize(word, self.lang)
