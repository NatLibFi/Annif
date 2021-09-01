"""Simple analyzer for Annif. Only folds words to lower case."""

import spacy
from spacy.tokens import Doc, Span
from . import analyzer


class SpacyAnalyzer(analyzer.Analyzer):
    name = "spacy"

    def __init__(self, param, **kwargs):
        self.param = param
        self.nlp = spacy.load(param, exclude=['ner', 'parser'])
        # we need a way to split sentences, now that parser is excluded
        self.nlp.add_pipe('sentencizer')
        super().__init__(**kwargs)

    def tokenize_sentences(self, text):
        doc = self.nlp(text)
        return list(doc.sents)

    def tokenize_words(self, text):
        if not isinstance(text, (Doc, Span)):
            text = self.nlp(text)
        return [lemma for lemma in (token.lemma_ for token in text)
                if self.is_valid_token(lemma)]

    def normalize_word(self, word):
        doc = self.nlp(word)
        return doc[:].lemma_
