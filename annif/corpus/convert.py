"""Mixin classes for converting between SubjectCorpus and DocumentCorpus"""

import collections
import os.path
import tempfile
from .types import Document, DocumentCorpus, SubjectCorpus


class SubjectWriter:
    """Writes a single subject file into a SubjectDirectory, performing
    buffering to limit the number of I/O operations."""

    _buffer = None

    BUFFER_SIZE = 100

    def __init__(self, path, uri, label):
        self._path = path
        self._buffer = ["{} {}".format(uri, label)]
        self._created = False

    def _flush(self):
        if self._created:
            mode = 'a'
        else:
            mode = 'w'

        with open(self._path, mode, encoding='utf-8') as subjfile:
            for text in self._buffer:
                print(text, file=subjfile)
        self._buffer = []
        self._created = True

    def write(self, text):
        self._buffer.append(text)
        if len(self._buffer) >= self.BUFFER_SIZE:
            self._flush()

    def close(self):
        self._flush()


class DocumentToSubjectCorpusMixin(SubjectCorpus):
    """Mixin class for enabling a DocumentCorpus to act as a SubjectCorpus"""

    _subject_corpus = None
    _temp_directory = None
    _subject_writer = None

    @property
    def subjects(self):
        if self._subject_corpus is None:
            self._generate_corpus_from_documents()
        return self._subject_corpus.subjects

    def _subject_filename(self, subject_id):
        filename = '{:08d}.txt'.format(subject_id)
        return os.path.join(self._temp_directory.name, filename)

    def _create_subject(self, subject_id, uri, label):
        filename = self._subject_filename(subject_id)
        self._subject_writer[subject_id] = SubjectWriter(filename, uri, label)

    def _add_text_to_subject(self, subject_id, text):
        self._subject_writer[subject_id].write(text)

    def _generate_corpus_from_documents(self):
        self._temp_directory = tempfile.TemporaryDirectory()
        self._subject_writer = {}

        for subject_id, subject_info in enumerate(self._subject_index):
            uri, label = subject_info
            self._create_subject(subject_id, uri, label)

        for doc in self.documents:
            for uri in doc.uris:
                subject_id = self._subject_index.by_uri(uri)
                if subject_id is None:
                    continue
                self._add_text_to_subject(subject_id, doc.text)

        for subject_id, _ in enumerate(self._subject_index):
            self._subject_writer[subject_id].close()

        from .subject import SubjectDirectory
        self._subject_corpus = SubjectDirectory(self._temp_directory.name)


class SubjectToDocumentCorpusMixin(DocumentCorpus):
    """Mixin class for enabling a SubjectCorpus to act as a DocumentCorpus"""

    _document_uris = None
    _document_labels = None

    @property
    def documents(self):
        if self._document_uris is None:
            self._generate_corpus_from_subjects()
        for text, uris in self._document_uris.items():
            labels = self._document_labels[text]
            yield Document(text=text, uris=uris, labels=labels)

    def _generate_corpus_from_subjects(self):
        self._document_uris = collections.defaultdict(set)
        self._document_labels = collections.defaultdict(set)
        for subj in self.subjects:
            for line in subj.text.splitlines():
                self._document_uris[line].add(subj.uri)
                self._document_labels[line].add(subj.label)
