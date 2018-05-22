"""Mixin classes for converting between SubjectCorpus and DocumentCorpus"""

import os.path
import tempfile
import annif.util
from .subject import SubjectCorpus, SubjectDirectory


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

    def _add_subject(self, uri, text):
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
                self._add_subject(uri, text)

        self._subject_corpus = SubjectDirectory(self._temp_directory.name)
