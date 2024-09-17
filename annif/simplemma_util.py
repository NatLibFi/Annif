"""Wrapper code for using Simplemma functionality in Annif"""

from typing import Dict, Tuple, Union

from simplemma import LanguageDetector, Lemmatizer
from simplemma.strategies import DefaultStrategy
from simplemma.strategies.dictionaries import DefaultDictionaryFactory

LANG_CACHE_SIZE = 5  # How many language dictionaries to keep in memory at once (max)

_dictionary_factory = DefaultDictionaryFactory(cache_max_size=LANG_CACHE_SIZE)
_lemmatization_strategy = DefaultStrategy(dictionary_factory=_dictionary_factory)
lemmatizer = Lemmatizer(lemmatization_strategy=_lemmatization_strategy)


def get_language_detector(lang: Union[str, Tuple[str, ...]]) -> LanguageDetector:
    return LanguageDetector(lang, lemmatization_strategy=_lemmatization_strategy)


def detect_language(text: str, languages: Tuple[str, ...]) -> Dict[str, float]:
    detector = get_language_detector(languages)
    proportions = detector.proportion_in_each_language(text)
    return dict(sorted(proportions.items(), key=lambda x: x[1], reverse=True))
