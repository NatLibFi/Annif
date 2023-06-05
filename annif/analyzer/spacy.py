"""spaCy analyzer for Annif which uses spaCy for lemmatization"""
from __future__ import annotations

import annif.util
from annif.exception import OperationFailedException

from . import analyzer

_KEY_LOWERCASE = "lowercase"


class SpacyAnalyzer(analyzer.Analyzer):
    name = "spacy"

    def __init__(self, param: str, **kwargs) -> None:
        import spacy

        self.param = param
        try:
            self.nlp = spacy.load(param, exclude=["ner", "parser"])
        except IOError as err:
            raise OperationFailedException(
                f"Loading spaCy model '{param}' failed - "
                + f"please download the model.\n{err}"
            )
        if _KEY_LOWERCASE in kwargs:
            self.lowercase = annif.util.boolean(kwargs[_KEY_LOWERCASE])
        else:
            self.lowercase = False
        super().__init__(**kwargs)

    def tokenize_words(self, text: str, filter: bool = True) -> list[str]:
        lemmas = [
            lemma
            for lemma in (token.lemma_ for token in self.nlp(text.strip()))
            if (not filter or self.is_valid_token(lemma))
        ]
        if self.lowercase:
            return [lemma.lower() for lemma in lemmas]
        else:
            return lemmas
