"""Simple analyzer for Annif. Only folds words to lower case."""
from __future__ import annotations

from . import analyzer


class SimpleAnalyzer(analyzer.Analyzer):
    name = "simple"

    def __init__(self, param: None, **kwargs) -> None:
        self.param = param
        super().__init__(**kwargs)

    def _normalize_word(self, word: str) -> str:
        return word.lower()
