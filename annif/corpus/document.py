"""Clases for supporting document corpora"""

import glob
import gzip
import os.path
import re
from itertools import islice

import annif.util

from .subject import SubjectSet
from .types import Document, DocumentCorpus

logger = annif.logger


class DocumentDirectory(DocumentCorpus):
    """A directory of files as a full text document corpus"""

    def __init__(self, path, subject_index, language, require_subjects=False):
        self.path = path
        self.subject_index = subject_index
        self.language = language
        self.require_subjects = require_subjects

    def __iter__(self):
        """Iterate through the directory, yielding tuples of (docfile,
        subjectfile) containing file paths. If there is no key file and
        require_subjects is False, the subjectfile will be returned as None."""

        for filename in sorted(glob.glob(os.path.join(self.path, "*.txt"))):
            tsvfilename = re.sub(r"\.txt$", ".tsv", filename)
            if os.path.exists(tsvfilename):
                yield (filename, tsvfilename)
                continue
            keyfilename = re.sub(r"\.txt$", ".key", filename)
            if os.path.exists(keyfilename):
                yield (filename, keyfilename)
                continue
            if not self.require_subjects:
                yield (filename, None)

    @property
    def documents(self):
        for docfilename, keyfilename in self:
            with open(docfilename, errors="replace", encoding="utf-8-sig") as docfile:
                text = docfile.read()
            with open(keyfilename, encoding="utf-8-sig") as keyfile:
                subjects = SubjectSet.from_string(
                    keyfile.read(), self.subject_index, self.language
                )
            yield Document(text=text, subject_set=subjects)


class DocumentFile(DocumentCorpus):
    """A TSV file as a corpus of documents with subjects"""

    def __init__(self, path, subject_index):
        self.path = path
        self.subject_index = subject_index

    @property
    def documents(self):
        if self.path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(self.path, mode="rt", encoding="utf-8-sig") as tsvfile:
            for line in tsvfile:
                yield from self._parse_tsv_line(line)

    def _parse_tsv_line(self, line):
        if "\t" in line:
            text, uris = line.split("\t", maxsplit=1)
            subject_ids = {
                self.subject_index.by_uri(annif.util.cleanup_uri(uri))
                for uri in uris.split()
            }
            yield Document(text=text, subject_set=SubjectSet(subject_ids))
        else:
            logger.warning('Skipping invalid line (missing tab): "%s"', line.rstrip())


class DocumentList(DocumentCorpus):
    """A document corpus based on a list of other iterable of Document
    objects"""

    def __init__(self, documents):
        self._documents = documents

    @property
    def documents(self):
        yield from self._documents


class TransformingDocumentCorpus(DocumentCorpus):
    """A document corpus that wraps another document corpus but transforms the
    documents using a given transform function"""

    def __init__(self, corpus, transform_fn):
        self._orig_corpus = corpus
        self._transform_fn = transform_fn

    @property
    def documents(self):
        for doc in self._orig_corpus.documents:
            yield Document(
                text=self._transform_fn(doc.text), subject_set=doc.subject_set
            )


class LimitingDocumentCorpus(DocumentCorpus):
    """A document corpus that wraps another document corpus but limits the
    number of documents to a given limit"""

    def __init__(self, corpus, docs_limit):
        self._orig_corpus = corpus
        self.docs_limit = docs_limit

    @property
    def documents(self):
        for doc in islice(self._orig_corpus.documents, self.docs_limit):
            yield doc
