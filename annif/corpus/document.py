"""Clases for supporting document corpora"""

import collections
import glob
import os.path
import re
import annif.util
from .types import Document, DocumentCorpus
from .convert import DocumentToSubjectCorpusMixin


class DocumentDirectory:
    """A directory of files as a full text document corpus"""

    def __init__(self, path, require_subjects=False):
        self.path = path
        self.require_subjects = require_subjects

    def __iter__(self):
        """Iterate through the directory, yielding tuples of (docfile,
        subjectfile) containing file paths. If there is no key file and
        require_subjects is False, the subjectfile will be returned as None."""

        for filename in glob.glob(os.path.join(self.path, '*.txt')):
            tsvfilename = re.sub(r'\.txt$', '.tsv', filename)
            if os.path.exists(tsvfilename):
                yield (filename, tsvfilename)
                continue
            keyfilename = re.sub(r'\.txt$', '.key', filename)
            if os.path.exists(keyfilename):
                yield (filename, keyfilename)
                continue
            if not self.require_subjects:
                yield (filename, None)


class DocumentFile(DocumentCorpus, DocumentToSubjectCorpusMixin):
    """A TSV file as a corpus of documents with subjects"""

    def __init__(self, path):
        self.path = path

    @property
    def documents(self):
        with open(self.path) as tsvfile:
            for line in tsvfile:
                text, uris = line.split('\t', maxsplit=1)
                subjects = [annif.util.cleanup_uri(uri)
                            for uri in uris.split()]
                yield Document(text=text, uris=subjects)
