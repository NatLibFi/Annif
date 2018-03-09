"""English analyzer for Annif, based on nltk Snowball stemmer."""

import nltk.stem.snowball
from . import analyzer


class EnglishAnalyzer(analyzer.Analyzer):
    name = "english"

    def __init__(self):
        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")

    def normalize_word(self, word):
        return self.stemmer.stem(word)
