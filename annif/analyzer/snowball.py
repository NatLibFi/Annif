"""Snowball analyzer for Annif, based on nltk Snowball stemmer."""

import functools
from . import analyzer


class SnowballAnalyzer(analyzer.Analyzer):
    name = "snowball"

    def __init__(self, param, **kwargs):
        self.param = param
        import nltk.stem.snowball
        self.stemmer = nltk.stem.snowball.SnowballStemmer(param)
        super().__init__(**kwargs)

    @functools.lru_cache(maxsize=500000)
    def _normalize_word(self, word):
        return self.stemmer.stem(word.lower())
