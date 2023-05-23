"""Snowball analyzer for Annif, based on nltk Snowball stemmer."""
from __future__ import annotations

import functools

from . import analyzer


class SnowballAnalyzer(analyzer.Analyzer):
    name = "snowball"

    def __init__(self, param: str, **kwargs) -> None:
        self.param = param
        import nltk.stem.snowball

        self.stemmer = nltk.stem.snowball.SnowballStemmer(param)
        super().__init__(**kwargs)

    @functools.lru_cache(maxsize=500000)
    def _normalize_word(self, word: str) -> str:
        return self.stemmer.stem(word.lower())
