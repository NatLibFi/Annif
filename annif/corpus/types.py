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

        sidx = self._subject_index

        if not uris and labels and sidx:
            uris = set((sidx[sidx.by_label(label)][0]
                        for label in labels
                        if sidx.by_label(label)))
        if not labels and uris and sidx:
            labels = set((sidx[sidx.by_uri(uri)][1]
                          for uri in uris
                          if sidx.by_uri(uri)))

        return Document(text=text, uris=uris, labels=labels)


Subject = collections.namedtuple('Subject', 'uri label text')


class SubjectCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for subject corpora"""

    @property
    @abc.abstractmethod
    def subjects(self):
        """Iterate through the subject corpus, yielding Subject objects."""
        pass  # pragma: no cover
