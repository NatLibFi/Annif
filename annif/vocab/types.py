"""Type declarations for vocabulary functionality"""

from __future__ import annotations

import abc

from annif.corpus import Subject


class SubjectIndex(metaclass=abc.ABCMeta):
    """Base class for an index that remembers the associations between
    integer subject IDs and their URIs and labels."""

    @abc.abstractmethod
    def __len__(self) -> int:
        pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def languages(self) -> list[str] | None:
        pass  # pragma: no cover

    @abc.abstractmethod
    def __getitem__(self, subject_id: int) -> Subject:
        pass  # pragma: no cover

    @abc.abstractmethod
    def contains_uri(self, uri: str) -> bool:
        pass  # pragma: no cover

    @abc.abstractmethod
    def by_uri(self, uri: str, warnings: bool = True) -> int | None:
        """return the subject ID of a subject by its URI, or None if not found.
        If warnings=True, log a warning message if the URI cannot be found."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def by_label(self, label: str | None, language: str) -> int | None:
        """return the subject ID of a subject by its label in a given
        language"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def active(self) -> list[tuple[int, Subject]]:
        """return a list of (subject_id, Subject) tuples of all subjects that
        are available for use"""
        pass  # pragma: no cover
