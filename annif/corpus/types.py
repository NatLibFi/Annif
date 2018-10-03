"""Basic types for document and subject corpora"""

import abc
import collections


Document = collections.namedtuple('Document', 'text uris labels')


class DocumentCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for document corpora"""

    @property
    @abc.abstractmethod
    def documents(self):
        """Iterate through the document corpus, yielding Document objects."""
        pass


Subject = collections.namedtuple('Subject', 'uri label text')


class SubjectCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for subject corpora"""

    @property
    @abc.abstractmethod
    def subjects(self):
        """Iterate through the subject corpus, yielding Subject objects."""
        pass
