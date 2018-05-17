"""A directory of files as a subject corpus"""


import collections
import glob
import os.path
import rdflib
import rdflib.util
from rdflib.namespace import SKOS, RDF


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
        for subject_id, subject in enumerate(corpus):
            self._uris.append(subject.uri)
            self._labels.append(subject.label)

    def __len__(self):
        return len(self._uris)

    def __getitem__(self, subject_id):
        return (self._uris[subject_id], self._labels[subject_id])

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


class SubjectIndexSKOS (SubjectIndex):
    """A subject index that uses SKOS files instead of TSV files"""

    @classmethod
    def load(cls, path, language):
        """Load subjects from a SKOS file and return a subject index."""

        def skos_file_as_corpus(path, language):
            graph = rdflib.Graph()
            format = rdflib.util.guess_format(path)
            graph.load(path, format=format)
            for concept in graph.subjects(RDF.type, SKOS.Concept):
                labels = graph.preferredLabel(concept, lang=language)
                if len(labels) > 0:
                    label = labels[0][1]
                    yield Subject(uri=str(concept), label=label, text=None)

        return cls(skos_file_as_corpus(path, language))
