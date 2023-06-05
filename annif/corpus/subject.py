"""Classes for supporting subject corpora expressed as directories or files"""
from __future__ import annotations

import csv
import os.path
from typing import TYPE_CHECKING, Any

import annif
import annif.util

from .skos import serialize_subjects_to_skos
from .types import Subject, SubjectCorpus

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    import numpy as np

logger = annif.logger.getChild("subject")
logger.addFilter(annif.util.DuplicateFilter())


class SubjectFileTSV(SubjectCorpus):
    """A monolingual subject vocabulary stored in a TSV file."""

    def __init__(self, path: str, language: str) -> None:
        """initialize the SubjectFileTSV given a path to a TSV file and the
        language of the vocabulary"""

        self.path = path
        self.language = language

    def _parse_line(self, line: str) -> Iterator[Subject]:
        vals = line.strip().split("\t", 2)
        clean_uri = annif.util.cleanup_uri(vals[0])
        label = vals[1] if len(vals) >= 2 else None
        labels = {self.language: label} if label else None
        notation = vals[2] if len(vals) >= 3 else None
        yield Subject(uri=clean_uri, labels=labels, notation=notation)

    @property
    def languages(self) -> list[str]:
        return [self.language]

    @property
    def subjects(self) -> Generator:
        with open(self.path, encoding="utf-8-sig") as subjfile:
            for line in subjfile:
                yield from self._parse_line(line)

    def save_skos(self, path: str) -> None:
        """Save the contents of the subject vocabulary into a SKOS/Turtle
        file with the given path name."""
        serialize_subjects_to_skos(self.subjects, path)


class SubjectFileCSV(SubjectCorpus):
    """A multilingual subject vocabulary stored in a CSV file."""

    def __init__(self, path: str) -> None:
        """initialize the SubjectFileCSV given a path to a CSV file"""
        self.path = path

    def _parse_row(self, row: dict[str, str]) -> Iterator[Subject]:
        labels = {
            fname.replace("label_", ""): value or None
            for fname, value in row.items()
            if fname.startswith("label_")
        }

        # if there are no labels in any language, set labels to None
        # indicating a deprecated subject
        if set(labels.values()) == {None}:
            labels = None

        yield Subject(
            uri=annif.util.cleanup_uri(row["uri"]),
            labels=labels,
            notation=row.get("notation", None) or None,
        )

    @property
    def languages(self) -> list[str]:
        # infer the supported languages from the CSV column names
        with open(self.path, encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile)
            fieldnames = next(reader, None)

        return [
            fname.replace("label_", "")
            for fname in fieldnames
            if fname.startswith("label_")
        ]

    @property
    def subjects(self) -> Generator:
        with open(self.path, encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                yield from self._parse_row(row)

    def save_skos(self, path: str) -> None:
        """Save the contents of the subject vocabulary into a SKOS/Turtle
        file with the given path name."""
        serialize_subjects_to_skos(self.subjects, path)

    @staticmethod
    def is_csv_file(path: str) -> bool:
        """return True if the path looks like a CSV file"""

        return os.path.splitext(path)[1].lower() == ".csv"


class SubjectIndex:
    """An index that remembers the associations between integers subject IDs
    and their URIs and labels."""

    def __init__(self) -> None:
        self._subjects = []
        self._uri_idx = {}
        self._label_idx = {}
        self._languages = None

    def load_subjects(self, corpus: SubjectCorpus) -> None:
        """Initialize the subject index from a subject corpus"""

        self._languages = corpus.languages
        for subject in corpus.subjects:
            self.append(subject)

    def __len__(self) -> int:
        return len(self._subjects)

    @property
    def languages(self) -> list[str] | None:
        return self._languages

    def __getitem__(self, subject_id: int) -> Subject:
        return self._subjects[subject_id]

    def append(self, subject: Subject) -> None:
        if self._languages is None and subject.labels is not None:
            self._languages = list(subject.labels.keys())

        subject_id = len(self._subjects)
        self._uri_idx[subject.uri] = subject_id
        if subject.labels:
            for lang, label in subject.labels.items():
                self._label_idx[(label, lang)] = subject_id
        self._subjects.append(subject)

    def contains_uri(self, uri: str) -> bool:
        return uri in self._uri_idx

    def by_uri(self, uri: str, warnings: bool = True) -> int | None:
        """return the subject ID of a subject by its URI, or None if not found.
        If warnings=True, log a warning message if the URI cannot be found."""
        try:
            return self._uri_idx[uri]
        except KeyError:
            if warnings:
                logger.warning("Unknown subject URI <%s>", uri)
            return None

    def by_label(self, label: str | None, language: str) -> int | None:
        """return the subject ID of a subject by its label in a given
        language"""
        try:
            return self._label_idx[(label, language)]
        except KeyError:
            logger.warning('Unknown subject label "%s"@%s', label, language)
            return None

    def deprecated_ids(self) -> list[int]:
        """return indices of deprecated subjects"""

        return [
            subject_id
            for subject_id, subject in enumerate(self._subjects)
            if subject.labels is None
        ]

    @property
    def active(self) -> list[tuple[int, Subject]]:
        """return a list of (subject_id, subject) tuples of all subjects that
        are not deprecated"""

        return [
            (subj_id, subject)
            for subj_id, subject in enumerate(self._subjects)
            if subject.labels is not None
        ]

    def save(self, path: str) -> None:
        """Save this subject index into a file with the given path name."""

        fieldnames = ["uri", "notation"] + [f"label_{lang}" for lang in self._languages]

        with open(path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for subject in self:
                row = {"uri": subject.uri, "notation": subject.notation or ""}
                if subject.labels:
                    for lang, label in subject.labels.items():
                        row[f"label_{lang}"] = label
                writer.writerow(row)

    @classmethod
    def load(cls, path: str) -> SubjectIndex:
        """Load a subject index from a CSV file and return it."""

        corpus = SubjectFileCSV(path)
        subject_index = cls()
        subject_index.load_subjects(corpus)
        return subject_index


class SubjectSet:
    """Represents a set of subjects for a document."""

    def __init__(self, subject_ids: Any | None = None) -> None:
        """Create a SubjectSet and optionally initialize it from an iterable
        of subject IDs"""

        if subject_ids:
            # use set comprehension to eliminate possible duplicates
            self._subject_ids = list(
                {subject_id for subject_id in subject_ids if subject_id is not None}
            )
        else:
            self._subject_ids = []

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int) -> int:
        return self._subject_ids[idx]

    def __bool__(self) -> bool:
        return bool(self._subject_ids)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SubjectSet):
            return self._subject_ids == other._subject_ids

        return False

    @classmethod
    def from_string(
        cls, subj_data: str, subject_index: SubjectIndex, language: str
    ) -> SubjectSet:
        subject_ids = set()
        for line in subj_data.splitlines():
            uri, label = cls._parse_line(line)
            if uri is not None:
                subject_ids.add(subject_index.by_uri(uri))
            else:
                subject_ids.add(subject_index.by_label(label, language))
        return cls(subject_ids)

    @staticmethod
    def _parse_line(
        line: str,
    ) -> tuple[str | None, str | None]:
        uri = label = None
        vals = line.split("\t")
        for val in vals:
            val = val.strip()
            if val == "":
                continue
            if val.startswith("<") and val.endswith(">"):  # URI
                uri = val[1:-1]
                continue
            label = val
            break
        return uri, label

    def as_vector(
        self, size: int | None = None, destination: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the hits as a one-dimensional NumPy array in sklearn
        multilabel indicator format. Use destination array if given (not
        None), otherwise create and return a new one of the given size."""

        if destination is None:
            import numpy as np

            assert size is not None and size > 0
            destination = np.zeros(size, dtype=bool)

        destination[list(self._subject_ids)] = True

        return destination
