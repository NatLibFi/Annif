"""A TSV file as a corpus of documents (usually only their titles) with
subjects"""


def cleanup_uri(uri):
    """remove angle brackets from a URI, if any"""
    if uri.startswith('<') and uri.endswith('>'):
        return uri[1:-1]
    return uri


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
                subjects = [cleanup_uri(uri) for uri in uris.split()]
                yield (text, subjects)
