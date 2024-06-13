"""Wrapper code for using Simplemma functionality in Annif"""

from simplemma import LanguageDetector, Lemmatizer
from simplemma.strategies import DefaultStrategy
from simplemma.strategies.dictionaries import DefaultDictionaryFactory

LANG_CACHE_SIZE = 5  # How many language dictionaries to keep in memory at once (max)

_dictionary_factory = DefaultDictionaryFactory(cache_max_size=LANG_CACHE_SIZE)
_lemmatization_strategy = DefaultStrategy(dictionary_factory=_dictionary_factory)
lemmatizer = Lemmatizer(lemmatization_strategy=_lemmatization_strategy)


def get_language_detector(lang: str) -> LanguageDetector:
    return LanguageDetector(lang, lemmatization_strategy=_lemmatization_strategy)
