"""Simple analyzer for Annif. Only folds words to lower case."""

from . import analyzer


class SimpleAnalyzer(analyzer.Analyzer):
    name = "simple"

    def __init__(self, param):
        self.param = param

    def normalize_word(self, word):
        return word.lower()
