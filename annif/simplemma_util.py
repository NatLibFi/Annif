"""Wrapper code for using Simplemma functionality in Annif"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Dict, Tuple, Union

if TYPE_CHECKING:
    from simplemma import LanguageDetector, Lemmatizer

LANG_CACHE_SIZE = 5  # How many language dictionaries to keep in memory at once (max)


@functools.lru_cache(maxsize=1)
def _strategy():
    from simplemma.strategies import DefaultStrategy
    from simplemma.strategies.dictionaries import DefaultDictionaryFactory

    factory = DefaultDictionaryFactory(cache_max_size=LANG_CACHE_SIZE)
    return DefaultStrategy(dictionary_factory=factory)


def get_lemmatizer() -> Lemmatizer:
    """Create a new simplemma lemmatizer instance."""
    from simplemma import Lemmatizer

    return Lemmatizer(lemmatization_strategy=_strategy())


def get_language_detector(lang: Union[str, Tuple[str, ...]]) -> LanguageDetector:
    """Create a new simplemma language detector for the given language(s)."""
    from simplemma import LanguageDetector

    return LanguageDetector(lang, lemmatization_strategy=_strategy())


def detect_language(text: str, languages: Tuple[str, ...]) -> Dict[str, float]:
    detector = get_language_detector(languages)
    proportions = detector.proportion_in_each_language(text)
    return dict(sorted(proportions.items(), key=lambda x: x[1], reverse=True))
