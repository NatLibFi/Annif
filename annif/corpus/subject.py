"""Classes for supporting subject corpora expressed as directories or files"""

import glob
import os.path
import annif.util
import numpy as np
from annif import logger
from .types import Subject, SubjectCorpus
from .convert import SubjectToDocumentCorpusMixin


class SubjectDirectory(SubjectCorpus, SubjectToDocumentCorpusMixin):
    """A subject corpus in the form of a directory with .txt files."""

    def __init__(self, path):
        self.path = path
        self._filenames = sorted(glob.glob(os.path.join(path, '*.txt')))

    @property
    def subjects(self):
        for filename in self._filenames:
            with open(filename, encoding='utf-8') as subjfile:
                uri, label = subjfile.readline().strip().split(' ', 1)
                text = ' '.join(subjfile.readlines())
                yield Subject(uri=uri, label=label, text=text)


class SubjectFileTSV(SubjectCorpus, SubjectToDocumentCorpusMixin):
    """A subject corpus stored in a TSV file."""

    def __init__(self, path):
        self.path = path

    @property
    def subjects(self):
        with open(self.path, encoding='utf-8') as subjfile:
            for line in subjfile:
                uri, label = line.strip().split(None, 1)
                clean_uri = annif.util.cleanup_uri(uri)
                yield Subject(uri=clean_uri, label=label, text=None)


class SubjectIndex:
    """An index that remembers the associations between integers subject IDs
    and their URIs and labels."""

    def __init__(self, corpus):
        """Initialize the subject index from a subject corpus."""
        self._uris = []
        self._labels = []
        self._uri_idx = {}
        self._label_idx = {}
        for subject_id, subject in enumerate(corpus.subjects):
            self._uris.append(subject.uri)
            self._labels.append(subject.label)
            self._uri_idx[subject.uri] = subject_id
            self._label_idx[subject.label] = subject_id

    def __len__(self):
        return len(self._uris)

    def __getitem__(self, subject_id):
        return (self._uris[subject_id], self._labels[subject_id])

    def by_uri(self, uri):
        """return the subject index of a subject by its URI"""
        try:
            return self._uri_idx[uri]
        except KeyError:
            logger.warning('Unknown subject URI <%s>', uri)
            return None

    def by_label(self, label):
        """return the subject index of a subject by its label"""
        try:
            return self._label_idx[label]
        except KeyError:
            logger.warning('Unknown subject label "%s"', label)
            return None

    def save(self, path):
        """Save this subject index into a file."""

        with open(path, 'w', encoding='utf-8') as subjfile:
            for subject_id in range(len(self)):
                line = "<{}>\t{}".format(
                    self._uris[subject_id], self._labels[subject_id])
                print(line, file=subjfile)

    @classmethod
    def load(cls, path):
        """Load a subject index from a TSV file and return it."""

        corpus = SubjectFileTSV(path)
        return cls(corpus)


class SubjectSet:
    """Represents a set of subjects for a document."""

    def __init__(self, subj_data=None):
        """Create a SubjectSet and optionally initialize it from a tuple
        (URIs, labels)"""

        uris, labels = subj_data or ([], [])
        self.subject_uris = set(uris)
        self.subject_labels = set(labels)

    @classmethod
    def from_string(cls, subj_data):
        sset = cls()
        for line in subj_data.splitlines():
            sset._parse_line(line)
        return sset

    def _parse_line(self, line):
        vals = line.split("\t")
        for val in vals:
            val = val.strip()
            if val == '':
                continue
            if val.startswith('<') and val.endswith('>'):  # URI
                self.subject_uris.add(val[1:-1])
                continue
            self.subject_labels.add(val)
            return

    def has_uris(self):
        """returns True if the URIs for all subjects are known"""
        return len(self.subject_uris) >= len(self.subject_labels)

    def as_vector(self, subject_index):
        """Return the hits as a one-dimensional NumPy array in sklearn
           multilabel indicator format, using a subject index as the source
           of subjects."""

        vector = np.zeros(len(subject_index), dtype=np.int8)
        if self.has_uris():
            for uri in self.subject_uris:
                subject_id = subject_index.by_uri(uri)
                if subject_id is not None:
                    vector[subject_id] = 1
        else:
            for label in self.subject_labels:
                subject_id = subject_index.by_label(label)
                if subject_id is not None:
                    vector[subject_id] = 1
        return vector
