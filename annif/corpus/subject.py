"""Classes for supporting subject corpora expressed as directories or files"""

import glob
import os.path
import annif.util
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
            with open(filename) as subjfile:
                uri, label = subjfile.readline().strip().split(' ', 1)
                text = ' '.join(subjfile.readlines())
                yield Subject(uri=uri, label=label, text=text)


class SubjectFileTSV(SubjectCorpus, SubjectToDocumentCorpusMixin):
    """A subject corpus stored in a TSV file."""

    def __init__(self, path):
        self.path = path

    @property
    def subjects(self):
        with open(self.path) as subjfile:
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
        for subject_id, subject in enumerate(corpus.subjects):
            self._uris.append(subject.uri)
            self._labels.append(subject.label)
            self._uri_idx[subject.uri] = subject_id

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

    def save(self, path):
        """Save this subject index into a file."""

        with open(path, 'w') as subjfile:
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
    def __init__(self, subj_data):
        self.subject_uris = set()
        self.subject_labels = set()
        self._parse(subj_data)

    def _parse(self, subj_data):
        for line in subj_data.splitlines():
            self._parse_line(line)

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
        return len(self.subject_uris) == len(self.subject_labels)
