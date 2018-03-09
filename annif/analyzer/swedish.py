"""Swedish analyzer for Annif, based on nltk Snowball stemmer."""

import nltk.stem.snowball
from . import analyzer


class SwedishAnalyzer(analyzer.Analyzer):
    name = "swedish"

    def __init__(self):
        self.stemmer = nltk.stem.snowball.SnowballStemmer("swedish")

    def normalize_word(self, word):
        return self.stemmer.stem(word)
