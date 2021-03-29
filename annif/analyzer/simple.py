"""Simple analyzer for Annif. Only folds words to lower case."""

from . import analyzer


class SimpleAnalyzer(analyzer.Analyzer):
    name = "simple"

    def __init__(self, param, **kwargs):
        self.param = param
        super().__init__(**kwargs)

    def normalize_word(self, word):
        return word.lower()
