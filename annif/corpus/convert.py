"""Mixin classes for converting between SubjectCorpus and DocumentCorpus"""

import collections
import os.path
import tempfile
from .types import Document, DocumentCorpus, SubjectCorpus


class DocumentToSubjectCorpusMixin(SubjectCorpus):
    """Mixin class for enabling a DocumentCorpus to act as a SubjectCorpus"""

    _subject_index = None
    _subject_corpus = None
    _temp_directory = None

    @property
    def subjects(self):
        if self._subject_corpus is None:
            self._generate_corpus_from_documents()
        return self._subject_corpus.subjects

    def set_subject_index(self, subject_index):
        """Set a subject index for looking up labels that are necessary for
        conversion"""

        self._subject_index = subject_index

    def _subject_filename(self, subject_id):
        filename = '{:08d}.txt'.format(subject_id)
        return os.path.join(self._temp_directory.name, filename)

    def _create_subject(self, subject_id, uri, label):
        filename = self._subject_filename(subject_id)
        with open(filename, 'w') as subjfile:
            print("{} {}".format(uri, label), file=subjfile)

    def _add_text_to_subject(self, subject_id, text):
        filename = self._subject_filename(subject_id)
        with open(filename, 'a') as subjfile:
            print(text, file=subjfile)

    def _generate_corpus_from_documents(self):
        self._temp_directory = tempfile.TemporaryDirectory()

        for subject_id, subject_info in enumerate(self._subject_index):
            uri, label = subject_info
            self._create_subject(subject_id, uri, label)

        for text, uris in self.documents:
            for uri in uris:
                subject_id = self._subject_index.by_uri(uri)
                if subject_id is None:
                    continue
                self._add_text_to_subject(subject_id, text)

        from .subject import SubjectDirectory
        self._subject_corpus = SubjectDirectory(self._temp_directory.name)


class SubjectToDocumentCorpusMixin(DocumentCorpus):
    """Mixin class for enabling a SubjectCorpus to act as a DocumentCorpus"""

    _document_subjects = None

    @property
    def documents(self):
        if self._document_subjects is None:
            self._generate_corpus_from_subjects()
        for text, uris in self._document_subjects.items():
            yield Document(text=text, uris=uris)

    def _generate_corpus_from_subjects(self):
        self._document_subjects = collections.defaultdict(set)
        for subj in self.subjects:
            for line in subj.text.splitlines():
                self._document_subjects[line].add(subj.uri)
