"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""

from __future__ import annotations

from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param: str, **kwargs) -> None:
        import annif.simplemma_util

        self._simplemma_util = annif.simplemma_util
        self.lang = param
        super().__init__(**kwargs)

    def _normalize_word(self, word: str) -> str:
        return self._simplemma_util.lemmatizer.lemmatize(word, lang=self.lang)
