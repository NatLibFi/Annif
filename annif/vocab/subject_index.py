"""Subject index functionality for Annif"""

from __future__ import annotations

import csv

import annif
import annif.util

from .subject_file import VocabFileCSV
from .types import Subject, SubjectIndex, VocabSource

logger = annif.logger
logger.addFilter(annif.util.DuplicateFilter())


class SubjectIndexFile(SubjectIndex):
    """SubjectIndex implementation backed by a file."""

    def __init__(self) -> None:
        self._subjects = []
        self._uri_idx = {}
        self._label_idx = {}
        self._languages = None

    def load_subjects(self, vocab_source: VocabSource) -> None:
        """Initialize the subject index from a subject corpus"""

        self._languages = vocab_source.languages
        for subject in vocab_source.subjects:
            self.append(subject)

    def __len__(self) -> int:
        return len(self._subjects)

    @property
    def languages(self) -> list[str] | None:
        return self._languages

    def __getitem__(self, subject_id: int) -> Subject:
        subject = self._subjects[subject_id]
        if subject.labels is None:
            raise IndexError(f"Subject is deprecated: {subject_id}")
        return subject

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
        try:
            subject_id = self._uri_idx[uri]
            if self._subjects[subject_id].labels is None:  # deprecated
                return None
            return subject_id
        except KeyError:
            if warnings:
                logger.warning("Unknown subject URI <%s>", uri)
            return None

    def by_label(self, label: str | None, language: str) -> int | None:
        try:
            return self._label_idx[(label, language)]
        except KeyError:
            logger.warning('Unknown subject label "%s"@%s', label, language)
            return None

    @property
    def active(self) -> list[tuple[int, Subject]]:
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

        vocab_file = VocabFileCSV(path)
        subject_index = cls()
        subject_index.load_subjects(vocab_file)
        return subject_index


class SubjectIndexFilter(SubjectIndex):
    """SubjectIndex implementation that filters another SubjectIndex based
    on a list of subject URIs to exclude."""

    def __init__(self, subject_index: SubjectIndex, exclude: list[str]):
        self._subject_index = subject_index
        self._exclude = set(exclude)

    def __len__(self) -> int:
        return len(self._subject_index)

    @property
    def languages(self) -> list[str] | None:
        return self._subject_index.languages

    def __getitem__(self, subject_id: int) -> Subject:
        subject = self._subject_index[subject_id]
        if subject.uri in self._exclude:
            raise IndexError(f"Subject is excluded: {subject.uri}")
        return subject

    def contains_uri(self, uri: str) -> bool:
        if uri in self._exclude:
            return False
        return self._subject_index.contains_uri(uri)

    def by_uri(self, uri: str, warnings: bool = True) -> int | None:
        """return the subject ID of a subject by its URI, or None if not found.
        If warnings=True, log a warning message if the URI cannot be found."""

        if uri in self._exclude:
            return None
        return self._subject_index.by_uri(uri, warnings)

    def by_label(self, label: str | None, language: str) -> int | None:
        """return the subject ID of a subject by its label in a given
        language"""

        subject_id = self._subject_index.by_label(label, language)
        subject = self._subject_index[subject_id]
        if subject is not None and subject.uri not in self._exclude:
            return subject_id
        return None

    @property
    def active(self) -> list[tuple[int, Subject]]:
        """return a list of (subject_id, Subject) tuples of all subjects that
        are available for use"""

        return [
            (subject_id, subject)
            for subject_id, subject in self._subject_index.active
            if subject.uri not in self._exclude
        ]
