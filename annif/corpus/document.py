"""Classes for supporting document corpora"""

from __future__ import annotations

import csv
import glob
import gzip
import os.path
import re
from itertools import islice
from typing import TYPE_CHECKING

import annif.util
from annif.exception import OperationFailedException

from .json import json_file_to_document, json_to_document
from .types import Document, DocumentCorpus, SubjectSet

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annif.vocab import SubjectIndex

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

    def __iter__(self) -> Iterator[str]:
        """Iterate through the directory, yielding file paths with corpus documents."""

        # txt files
        for filename in sorted(glob.glob(os.path.join(self.path, "*.txt"))):
            yield filename

        # json files
        for filename in sorted(glob.glob(os.path.join(self.path, "*.json"))):
            yield filename

    @staticmethod
    def _get_subject_filename(filename: str) -> str | None:
        tsvfilename = re.sub(r"\.txt$", ".tsv", filename)
        if os.path.exists(tsvfilename):
            return tsvfilename

        keyfilename = re.sub(r"\.txt$", ".key", filename)
        if os.path.exists(keyfilename):
            return keyfilename

        return None

    def _read_txt_file(self, filename: str) -> Document | None:
        with open(filename, errors="replace", encoding="utf-8-sig") as docfile:
            text = docfile.read()
        if not self.require_subjects:
            return Document(text=text, subject_set=None, file_path=filename)

        subjfilename = self._get_subject_filename(filename)
        if subjfilename is None:
            # subjects required but not found, skipping this docfile
            return None

        with open(subjfilename, encoding="utf-8-sig") as subjfile:
            subjects = SubjectSet.from_string(
                subjfile.read(), self.subject_index, self.language
            )
        return Document(text=text, subject_set=subjects, file_path=filename)

    @property
    def documents(self) -> Iterator[Document]:
        for docfilename in self:
            if docfilename.endswith(".txt"):
                doc = self._read_txt_file(docfilename)
            else:
                doc = json_file_to_document(
                    docfilename,
                    self.subject_index,
                    self.language,
                    self.require_subjects,
                )

            if doc is not None:
                yield doc


class DocumentFileTSV(DocumentCorpus):
    """A TSV file as a corpus of documents with subjects"""

    def __init__(
        self, path: str, subject_index: SubjectIndex, require_subjects=True
    ) -> None:
        self.path = path
        self.subject_index = subject_index
        self.require_subjects = require_subjects

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
            if self.require_subjects:
                logger.warning(
                    'Skipping invalid line (missing tab): "%s"', line.rstrip()
                )
            else:
                yield Document(text=line.strip())


class DocumentFileCSV(DocumentCorpus):
    """A CSV file as a corpus of documents with subjects"""

    def __init__(
        self, path: str, subject_index: SubjectIndex, require_subjects=True
    ) -> None:
        self.path = path
        self.subject_index = subject_index
        self.require_subjects = require_subjects

    @property
    def documents(self) -> Iterator[Document]:
        if self.path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(self.path, mode="rt", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            if not self._check_fields(reader):
                if self.require_subjects:
                    raise OperationFailedException(
                        f"Cannot parse CSV file {self.path}. "
                        + "The file must have a header row that defines at least "
                        + "the columns 'text' and 'subject_uris'."
                    )
                else:
                    raise OperationFailedException(
                        f"Cannot parse CSV file {self.path}. "
                        + "The file must have a header row that defines at least "
                        + "the column 'text'."
                    )
            for row in reader:
                yield from self._parse_row(row)

    def _parse_row(self, row: dict[str, str]) -> Iterator[Document]:
        if self.require_subjects:
            subject_ids = {
                self.subject_index.by_uri(annif.util.cleanup_uri(uri))
                for uri in (row["subject_uris"] or "").strip().split()
            }
        else:
            subject_ids = set()
        metadata = {
            key: val for key, val in row.items() if key not in ("text", "subject_uris")
        }
        yield Document(
            text=(row["text"] or ""),
            subject_set=SubjectSet(subject_ids),
            metadata=metadata,
        )

    def _check_fields(self, reader: csv.DictReader) -> bool:
        fns = reader.fieldnames
        if self.require_subjects:
            return fns is not None and "text" in fns and "subject_uris" in fns
        else:
            return fns is not None and "text" in fns

    @staticmethod
    def is_csv_file(path: str) -> bool:
        """return True if the path looks like a CSV file"""

        path_lc = path.lower()
        return path_lc.endswith(".csv") or path_lc.endswith(".csv.gz")


class DocumentFileJSONL(DocumentCorpus):
    """A JSON Lines file as a corpus of documents with subjects"""

    def __init__(
        self,
        path: str,
        subject_index: SubjectIndex,
        language: str,
        require_subjects=True,
    ) -> None:
        self.path = path
        self.subject_index = subject_index
        self.language = language
        self.require_subjects = require_subjects

    @property
    def documents(self) -> Iterator[Document]:
        if self.path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(self.path, mode="rt", encoding="utf-8") as jsonlfile:
            for line in jsonlfile:
                doc = json_to_document(
                    self.path,
                    line,
                    self.subject_index,
                    self.language,
                    self.require_subjects,
                )
                if doc is not None:
                    yield doc

    @staticmethod
    def is_jsonl_file(path: str) -> bool:
        """return True if the path looks like a JSONL file"""

        path_lc = path.lower()
        return path_lc.endswith(".jsonl") or path_lc.endswith(".jsonl.gz")


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
            yield self._transform_fn(doc)


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
