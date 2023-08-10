"""Simplemma analyzer for Annif, based on simplemma lemmatizer."""

from __future__ import annotations

import simplemma

import annif.langsupport

from . import analyzer


class SimplemmaAnalyzer(analyzer.Analyzer):
    name = "simplemma"

    def __init__(self, param: str, **kwargs) -> None:
        self.lang = param
        self.lemmatizer = simplemma.Lemmatizer(
            lemmatization_strategy=annif.langsupport.lemmatization_strategy
        )
        super().__init__(**kwargs)

    def _normalize_word(self, word: str) -> str:
        return self.lemmatizer.lemmatize(word, lang=self.lang)
