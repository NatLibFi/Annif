"""Class for combining multiple corpora so they behave like a single corpus"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .types import DocumentCorpus

if TYPE_CHECKING:
    from itertools import chain

    from annif.corpus.document import DocumentFile


class CombinedCorpus(DocumentCorpus):
    """Class for combining multiple corpora so they behave like a single
    corpus"""

    def __init__(self, corpora: List[DocumentFile]) -> None:
        self._corpora = corpora

    @property
    def documents(self) -> itertools.chain:
        return itertools.chain.from_iterable(
            [corpus.documents for corpus in self._corpora]
        )
