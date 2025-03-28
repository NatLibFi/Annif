"""Classes for supporting vocabulary files in CSV or TSV format"""

from __future__ import annotations

import csv
import os.path
from typing import TYPE_CHECKING

import annif
import annif.util

from .skos import serialize_subjects_to_skos
from .types import Subject, VocabSource

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


class VocabFileTSV(VocabSource):
    """A monolingual subject vocabulary stored in a TSV file."""

    def __init__(self, path: str, language: str) -> None:
        """initialize the VocabFileTSV given a path to a TSV file and the
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


class VocabFileCSV(VocabSource):
    """A multilingual subject vocabulary stored in a CSV file."""

    def __init__(self, path: str) -> None:
        """initialize the VocabFileCSV given a path to a CSV file"""
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
