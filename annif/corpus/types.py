"""Basic types for document and subject corpora"""

import abc
import collections


Document = collections.namedtuple('Document', 'text subject_set')


class DocumentCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for document corpora"""

    @property
    @abc.abstractmethod
    def documents(self):
        """Iterate through the document corpus, yielding Document objects."""
        pass  # pragma: no cover

    def is_empty(self):
        """Check if there are no documents to iterate."""
        try:
            next(self.documents)
            return False
        except StopIteration:
            return True


Subject = collections.namedtuple('Subject', 'uri label notation')


class SubjectCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for subject corpora"""

    @property
    @abc.abstractmethod
    def subjects(self):
        """Iterate through the subject corpus, yielding Subject objects."""
        pass  # pragma: no cover
