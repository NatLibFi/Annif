"""A directory of files as a subject corpus"""


import glob
import os.path
import re


class Subject:
    def __init__(self, uri, label, text):
        self.uri = uri
        self.label = label
        self.text = text


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
                yield Subject(uri, label, text)


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
                    yield Subject(uri, label, None)

        return cls(file_as_corpus(path))


class VectorCorpus:
    """A class that wraps a subject corpus so it can be iterated as lists of
    vectors, by using a dictionary to map words to integers."""

    def __init__(self, corpus, dictionary, analyzer):
        self.corpus = corpus
        self.dictionary = dictionary
        self.analyzer = analyzer

    def __iter__(self):
        """Iterate through the subject directory, yielding vectors that are
        derived from subjects using the given analyzer and dictionary."""

        for subject in self.corpus:
            yield self.dictionary.doc2bow(
                self.analyzer.tokenize_words(subject.text))
