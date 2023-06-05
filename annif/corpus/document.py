"""Clases for supporting document corpora"""
from __future__ import annotations

import glob
import gzip
import os.path
import re
from itertools import islice
from typing import TYPE_CHECKING

import annif.util

from .subject import SubjectSet
from .types import Document, DocumentCorpus

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annif.corpus.subject import SubjectIndex

logger = annif.logger


class DocumentDirectory(DocumentCorpus):
    """A directory of files as a full text document corpus"""

    def __init__(
        self,
        path: str,
        subject_index: SubjectIndex | None = None,
        language: str | None = None,
        require_subjects: bool = False,
    ) -> None:
        self.path = path
        self.subject_index = subject_index
        self.language = language
        self.require_subjects = require_subjects

    def __iter__(self) -> Iterator[tuple[str, str] | tuple[str, None]]:
        """Iterate through the directory, yielding tuples of (docfile,
        subjectfile) containing file paths. If require_subjects is False, the
        subjectfile will be returned as None."""

        for filename in sorted(glob.glob(os.path.join(self.path, "*.txt"))):
            if self.require_subjects:
                tsvfilename = re.sub(r"\.txt$", ".tsv", filename)
                if os.path.exists(tsvfilename):
                    yield (filename, tsvfilename)
                    continue
                keyfilename = re.sub(r"\.txt$", ".key", filename)
                if os.path.exists(keyfilename):
                    yield (filename, keyfilename)
                    continue
            else:
                yield (filename, None)

    @property
    def documents(self) -> Iterator[Document]:
        for docfilename, subjfilename in self:
            with open(docfilename, errors="replace", encoding="utf-8-sig") as docfile:
                text = docfile.read()
            if subjfilename is None:
                yield Document(text=text, subject_set=None)
                continue
            with open(subjfilename, encoding="utf-8-sig") as subjfile:
                subjects = SubjectSet.from_string(
                    subjfile.read(), self.subject_index, self.language
                )
            yield Document(text=text, subject_set=subjects)


class DocumentFile(DocumentCorpus):
    """A TSV file as a corpus of documents with subjects"""

    def __init__(self, path: str, subject_index: SubjectIndex) -> None:
        self.path = path
        self.subject_index = subject_index

    @property
    def documents(self) -> Iterator[Document]:
        if self.path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(self.path, mode="rt", encoding="utf-8-sig") as tsvfile:
            for line in tsvfile:
                yield from self._parse_tsv_line(line)

    def _parse_tsv_line(self, line: str) -> Iterator[Document]:
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
