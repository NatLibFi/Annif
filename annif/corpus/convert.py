"""Mixin classes for converting between SubjectCorpus and DocumentCorpus"""

import collections
import os.path
import tempfile
import annif.util
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

    def _add_subject(self, subject_id, uri, text):
        filename = '{}.txt'.format(annif.util.localname(uri))
        path = os.path.join(self._temp_directory.name, filename)
        if not os.path.exists(path):
            subject_id = self._subject_index.by_uri(uri)
            label = self._subject_index[subject_id][1]
            with open(path, 'w') as subjfile:
                print("{} {}".format(uri, label), file=subjfile)
        with open(path, 'a') as subjfile:
            print(text, file=subjfile)

    def _generate_corpus_from_documents(self):
        self._temp_directory = tempfile.TemporaryDirectory()

        for text, uris in self.documents:
            for uri in uris:
                subject_id = self._subject_index.by_uri(uri)
                if subject_id is None:
                    continue
                self._add_subject(subject_id, uri, text)

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
