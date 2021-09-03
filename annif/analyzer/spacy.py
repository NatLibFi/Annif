"""Simple analyzer for Annif. Only folds words to lower case."""

import spacy
from . import analyzer


class SpacyAnalyzer(analyzer.Analyzer):
    name = "spacy"

    def __init__(self, param, **kwargs):
        self.param = param
        self.nlp = spacy.load(param, exclude=['ner', 'parser'])
        super().__init__(**kwargs)

    def tokenize_words(self, text):
        return [lemma for lemma in (token.lemma_ for token in self.nlp(text))
                if self.is_valid_token(lemma)]

    def normalize_word(self, word):
        doc = self.nlp(word)
        return doc[:].lemma_
