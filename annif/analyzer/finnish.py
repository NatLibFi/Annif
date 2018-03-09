"""Finnish analyzer for Annif, based on nltk Snowball stemmer."""

import nltk.stem.snowball
from . import analyzer

class FinnishAnalyzer(analyzer.Analyzer):
    name = "finnish"

    def __init__(self):
        self.stemmer = nltk.stem.snowball.SnowballStemmer("finnish")
    
    def normalize_word(self, word):
        return self.stemmer.stem(word)
