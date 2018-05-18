"""Classes for supporting subject corpora expressed as directories or files"""

import collections
import glob
import os.path


Subject = collections.namedtuple('Subject', 'uri label text')


class SubjectDirectory:
    def __init__(self, path):
        self.path = path
        self._filenames = sorted(glob.glob(os.path.join(path, '*.txt')))

    def __iter__(self):
        """Iterate through the subject directory, yielding Subject objects."""

        for filename in self._filenames:
            with open(filename) as subjfile:
                uri, label = subjfile.readline().strip().split(' ', 1)
                text = ' '.join(subjfile.readlines())
                yield Subject(uri=uri, label=label, text=text)


class SubjectIndex:
    """A class that remembers the associations between integers subject IDs
    and their URIs and labels."""

    def __init__(self, corpus):
        """Initialize the subject index from a subject corpus."""
        self._uris = []
        self._labels = []
        self._uri_idx = {}
        for subject_id, subject in enumerate(corpus):
            self._uris.append(subject.uri)
            self._labels.append(subject.label)
            self._uri_idx[subject.uri] = subject_id

    def __len__(self):
        return len(self._uris)

    def __getitem__(self, subject_id):
        return (self._uris[subject_id], self._labels[subject_id])

    def by_uri(self, uri):
        """return the subject index of a subject by its URI"""
        return self._uri_idx.get(uri, None)

    def save(self, path):
        """Save this subject index into a file."""

        with open(path, 'w') as subjfile:
            for subject_id in range(len(self)):
                line = "<{}>\t{}".format(
                    self._uris[subject_id], self._labels[subject_id])
                print(line, file=subjfile)

    @classmethod
    def load(cls, path):
        """Load a subject index from a file and return it."""

        def file_as_corpus(path):
            with open(path) as subjfile:
                for line in subjfile:
                    uri, label = line.strip().split(None, 1)
                    uri = uri[1:-1]
                    yield Subject(uri=uri, label=label, text=None)

        return cls(file_as_corpus(path))
