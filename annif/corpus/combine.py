"""Class for combining multiple corpora so they behave like a single corpus"""

import itertools

from .types import DocumentCorpus


class CombinedCorpus(DocumentCorpus):
    """Class for combining multiple corpora so they behave like a single
    corpus"""

    def __init__(self, corpora):
        self._corpora = corpora

    @property
    def documents(self):
        return itertools.chain.from_iterable(
            [corpus.documents for corpus in self._corpora]
        )
