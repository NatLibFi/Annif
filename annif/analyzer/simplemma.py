"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""
from __future__ import annotations

import simplemma

from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param: str, **kwargs) -> None:
        self.lang = param
        super().__init__(**kwargs)

    def _normalize_word(self, word: str) -> str:
        return simplemma.lemmatize(word, lang=self.lang)
