"""Class for combining multiple corpora so they behave like a single corpus"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from .types import DocumentCorpus

if TYPE_CHECKING:
    from annif.corpus.document import DocumentFile


class CombinedCorpus(DocumentCorpus):
    """Class for combining multiple corpora so they behave like a single
    corpus"""

    def __init__(self, corpora: list[DocumentFile]) -> None:
        self._corpora = corpora

    @property
    def documents(self) -> itertools.chain:
        return itertools.chain.from_iterable(
            [corpus.documents for corpus in self._corpora]
        )
