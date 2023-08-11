"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""
from __future__ import annotations

import annif.simplemma_util

from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param: str, **kwargs) -> None:
        self.lang = param
        super().__init__(**kwargs)

    def _normalize_word(self, word: str) -> str:
        return annif.simplemma_util.lemmatizer.lemmatize(word, lang=self.lang)
