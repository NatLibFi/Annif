"""Language support and language detection functionality for Annif"""

from simplemma.strategies import DefaultStrategy
from simplemma.strategies.dictionaries import DefaultDictionaryFactory

LANG_CACHE_SIZE = 5  # How many language dictionaries to keep in memory at once (max)

dictionary_factory = DefaultDictionaryFactory(cache_max_size=LANG_CACHE_SIZE)
lemmatization_strategy = DefaultStrategy(dictionary_factory=dictionary_factory)
