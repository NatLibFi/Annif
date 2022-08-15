"""Basic types for document and subject corpora"""

import abc
import collections


Document = collections.namedtuple('Document', 'text uris labels')


class DocumentCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for document corpora"""

    _subject_index = None

    @property
    @abc.abstractmethod
    def documents(self):
        """Iterate through the document corpus, yielding Document objects."""
        pass  # pragma: no cover

    def set_subject_index(self, subject_index):
        """Set a subject index for looking up labels that are necessary for
        conversion"""

        self._subject_index = subject_index

    def _create_document(self, text, uris, labels):
        """Create a new Document instance from possibly incomplete
        information. URIs for labels and vice versa are looked up from the
        subject index, if available."""

        if self._subject_index:
            if not uris and labels:
                uris = set((self._subject_index.labels_to_uris(labels)))
            if not labels and uris:
                labels = set((self._subject_index.uris_to_labels(uris)))

        return Document(text=text, uris=uris, labels=labels)

    def is_empty(self):
        """Check if there are no documents to iterate."""
        try:
            next(self.documents)
            return False
        except StopIteration:
            return True


Subject = collections.namedtuple('Subject', 'uri label notation text')


class SubjectCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for subject corpora"""

    @property
    @abc.abstractmethod
    def subjects(self):
        """Iterate through the subject corpus, yielding Subject objects."""
        pass  # pragma: no cover
