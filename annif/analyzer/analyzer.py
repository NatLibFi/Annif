"""Common functionality for analyzers."""

import abc
import unicodedata
import nltk.tokenize


class Analyzer(metaclass=abc.ABCMeta):
    """Base class for language-specific analyzers. The non-implemented
    methods should be overridden in subclasses. Tokenize functions may
    be overridden when necessary."""

    name = None
    TOKEN_MIN_LENGTH = 3

    def tokenize_sentences(self, text):
        """Tokenize a piece of text (e.g. a document) into sentences."""
        return nltk.tokenize.sent_tokenize(text)

    def is_valid_token(self, word):
        """Return True if the word is an acceptable token."""
        if len(word) < self.TOKEN_MIN_LENGTH:
            return False
        for char in word:
            category = unicodedata.category(char)
            if category[0] == 'L':  # letter
                return True
        return False

    def tokenize_words(self, text):
        """Tokenize a piece of text (e.g. a sentence) into words."""
        return [self.normalize_word(word)
                for word in nltk.tokenize.word_tokenize(text)
                if self.is_valid_token(word)]

    @abc.abstractmethod
    def normalize_word(self, word):
        """Normalize (stem or lemmatize) a word form into a normal form."""
        pass  # pragma: no cover
