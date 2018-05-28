"""Class for combining multiple corpora so they behave like a single corpus"""

import itertools
from .types import DocumentCorpus, SubjectCorpus


class CombinedCorpus(SubjectCorpus, DocumentCorpus):
    """Class for combining multiple corpora so they behave like a single
    corpus"""

    def __init__(self, corpora):
        self._corpora = corpora

    @property
    def subjects(self):
        return itertools.chain.from_iterable(
            [corpus.subjects for corpus in self._corpora])

    @property
    def documents(self):
        return itertools.chain.from_iterable(
            [corpus.documents for corpus in self._corpora])

    def set_subject_index(self, subject_index):
        """Set a subject index for looking up labels that are necessary for
        conversion"""

        for corpus in self._corpora:
            if hasattr(corpus, 'set_subject_index'):
                corpus.set_subject_index(subject_index)
