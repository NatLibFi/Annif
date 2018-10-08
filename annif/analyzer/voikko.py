"""Snowball analyzer for Annif, based on nltk Snowball stemmer."""

import functools
import voikko.libvoikko
from . import analyzer


class VoikkoAnalyzer(analyzer.Analyzer):
    name = "voikko"

    def __init__(self, param):
        self.param = param
        self.voikko = voikko.libvoikko.Voikko(param)

    @functools.lru_cache(maxsize=500000)
    def normalize_word(self, word):
        result = self.voikko.analyze(word)
        if len(result) > 0 and 'BASEFORM' in result[0]:
            return result[0]['BASEFORM']
        return word
