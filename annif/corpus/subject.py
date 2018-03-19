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

    def __iter__(self):
        """Iterate through the directory, yielding Subject objects."""

        for filename in glob.glob(os.path.join(self.path, '*.txt')):
            with open(filename) as subjfile:
                uri, label = subjfile.readline().strip().split(' ', 1)
                text = ' '.join(subjfile.readlines())
                yield Subject(uri, label, text)
