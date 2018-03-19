"""Snowball analyzer for Annif, based on nltk Snowball stemmer."""

import nltk.stem.snowball
from . import analyzer


class SnowballAnalyzer(analyzer.Analyzer):
    name = "snowball"

    def __init__(self, name, param):
        self.stemmer = nltk.stem.snowball.SnowballStemmer(param)

    def normalize_word(self, word):
        return self.stemmer.stem(word.lower())
