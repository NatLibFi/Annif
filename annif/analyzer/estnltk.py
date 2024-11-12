"""EstNLTK analyzer for Annif which uses EstNLTK for lemmatization"""

from __future__ import annotations

from . import analyzer


class EstNLTKAnalyzer(analyzer.Analyzer):
    name = "estnltk"

    def __init__(self, param: str, **kwargs) -> None:
        self.param = param
        super().__init__(**kwargs)

    def tokenize_words(self, text: str, filter: bool = True) -> list[str]:
        import estnltk

        txt = estnltk.Text(text.strip())
        txt.tag_layer()
        lemmas = [
            lemma
            for lemma in [l[0] for l in txt.lemma]
            if (not filter or self.is_valid_token(lemma))
        ]
        return lemmas
