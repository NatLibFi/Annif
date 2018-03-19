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
        self._subjects = {}

    def __iter__(self):
        """Iterate through the subject directory, yielding Subject objects."""

        for filename in self._filenames:
            with open(filename) as subjfile:
                uri, label = subjfile.readline().strip().split(' ', 1)
                text = ' '.join(subjfile.readlines())
                yield Subject(uri, label, text)

    def tokens(self, analyzer):
        """Iterate through the subject directory, yielding lists of tokens
        that are derived from subjects using the given analyzer."""

        for subject in self:
            yield analyzer.tokenize_words(subject.text)
