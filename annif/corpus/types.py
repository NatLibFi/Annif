"""Basic types for document corpora"""

from __future__ import annotations

import abc
import collections
from itertools import islice
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from annif.vocab import SubjectIndex


class Document:
    def __init__(self, text, subject_set=None, metadata=None):
        self.text = text
        self.subject_set = subject_set if subject_set is not None else set()
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return (
            f"Document(text={self.text!r}, "
            f"subject_set={self.subject_set!r}, "
            f"metadata={self.metadata!r})"
        )


class DocumentCorpus(metaclass=abc.ABCMeta):
    """Abstract base class for document corpora"""

    DOC_BATCH_SIZE = 32

    @property
    @abc.abstractmethod
    def documents(self):
        """Iterate through the document corpus, yielding Document objects."""
        pass  # pragma: no cover

    @property
    def doc_batches(self) -> collections.abc.Iterator[list[Document]]:
        """Iterate through the document corpus in batches, yielding lists of Document
        objects."""
        it = iter(self.documents)
        while True:
            docs_batch = list(islice(it, self.DOC_BATCH_SIZE))
            if not docs_batch:
                return
            yield docs_batch

    def is_empty(self) -> bool:
        """Check if there are no documents to iterate."""
        try:
            next(self.documents)
            return False
        except StopIteration:
            return True


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
