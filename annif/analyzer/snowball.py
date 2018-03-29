"""Snowball analyzer for Annif, based on nltk Snowball stemmer."""

import functools
import nltk.stem.snowball
from . import analyzer


class SnowballAnalyzer(analyzer.Analyzer):
    name = "snowball"

    def __init__(self, param):
        self.param = param
        self.stemmer = nltk.stem.snowball.SnowballStemmer(param)

    @functools.lru_cache(maxsize=500000)
    def normalize_word(self, word):
        return self.stemmer.stem(word.lower())
