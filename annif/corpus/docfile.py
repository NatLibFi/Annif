"""A TSV file as a corpus of documents (usually only their titles) with
subjects"""

import annif.util


class DocumentFile:
    """A TSV file as a corpus of documents with subjects"""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        """Iterate through the file, yielding tuples of (text, subjects) where
        subjects is a list of concept URIs."""

        with open(self.path) as tsvfile:
            for line in tsvfile:
                text, uris = line.split('\t', maxsplit=1)
                subjects = [annif.util.cleanup_uri(uri)
                            for uri in uris.split()]
                yield (text, subjects)
