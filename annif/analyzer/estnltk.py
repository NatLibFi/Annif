"""EstNLTK analyzer for Annif which uses EstNLTK for lemmatization"""

from __future__ import annotations

import importlib

from . import analyzer


class EstNLTKAnalyzer(analyzer.Analyzer):
    name = "estnltk"

    @staticmethod
    def is_available() -> bool:
        # return True iff EstNLTK is installed
        return importlib.util.find_spec("estnltk") is not None

    def __init__(self, param: str, **kwargs) -> None:
        self.param = param
        super().__init__(**kwargs)

    def tokenize_words(self, text: str, filter: bool = True) -> list[str]:
        import estnltk

        txt = estnltk.Text(text.strip())
        txt.tag_layer()
        return [
            lemma
            for lemma in [lemmas[0] for lemmas in txt.lemma]
            if (not filter or self.is_valid_token(lemma))
        ]
