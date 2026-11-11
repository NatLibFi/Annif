"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""

from __future__ import annotations

import annif.simplemma_util

from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param: str, **kwargs) -> None:
        self.lang = param
        self._lemmatizer = None
        super().__init__(**kwargs)

    def _normalize_word(self, word: str) -> str:
        if self._lemmatizer is None:
            self._lemmatizer = annif.simplemma_util.get_lemmatizer()
        return self._lemmatizer.lemmatize(word, lang=self.lang)

    def __getstate__(self) -> dict:
        # The lemmatizer (and its dictionary cache) is not picklable;
        # it is reconstructed lazily in the worker process.
        state = self.__dict__.copy()
        state["_lemmatizer"] = None
        return state
