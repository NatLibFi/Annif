"""Common functionality for analyzers."""

import abc
import nltk.tokenize


class Analyzer(metaclass=abc.ABCMeta):
    """Base class for language-specific analyzers. The non-implemented
    methods should be overridden in subclasses. Tokenize functions may
    be overridden when necessary."""

    name = None

    def __init__(self, name):
        self.name = name

    def tokenize_sentences(self, text):
        """Tokenize a piece of text (e.g. a document) into sentences."""
        return nltk.tokenize.sent_tokenize(text)

    def tokenize_words(self, text):
        """Tokenize a piece of text (e.g. a sentence) into words."""
        return nltk.tokenize.word_tokenize(text)

    @abc.abstractmethod
    def normalize_word(self, word):
        """Normalize (stem or lemmatize) a word form into a normal form."""
        pass
