"""Representing suggested subjects."""
from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from annif.corpus.subject import SubjectIndex

SubjectSuggestion = collections.namedtuple("SubjectSuggestion", "subject_id score")


def vector_to_suggestions(vector: np.ndarray, limit: int) -> Iterator:
    limit = min(len(vector), limit)
    topk_idx = np.argpartition(vector, -limit)[-limit:]
    return (
        SubjectSuggestion(subject_id=idx, score=float(vector[idx])) for idx in topk_idx
    )


def filter_suggestion(
    preds: csr_array,
    limit: int | None = None,
    threshold: float = 0.0,
) -> csr_array:
    """filter a 2D sparse suggestion array (csr_array), retaining only the
    top K suggestions with a score above or equal to the threshold for each
    individual prediction; the rest will be left as zeros"""

    if limit == 0:
        return csr_array(preds.shape, dtype=np.float32)  # empty

    data, rows, cols = [], [], []
    for row in range(preds.shape[0]):
        arow = preds.getrow(row)
        if limit is not None and limit < len(arow.data):
            topk_idx = arow.data.argpartition(-limit)[-limit:]
        else:
            topk_idx = range(len(arow.data))
        for idx in topk_idx:
            if arow.data[idx] >= threshold:
                data.append(arow.data[idx])
                rows.append(row)
                cols.append(arow.indices[idx])
    return csr_array((data, (rows, cols)), shape=preds.shape, dtype=np.float32)


class SuggestionResult:
    """Suggestions for a single document, backed by a row of a sparse array."""

    def __init__(self, array: csr_array, idx: int) -> None:
        self._array = array
        self._idx = idx

    def __iter__(self):
        _, cols = self._array[[self._idx], :].nonzero()
        suggestions = [
            SubjectSuggestion(subject_id=col, score=float(self._array[self._idx, col]))
            for col in cols
        ]
        return iter(
            sorted(suggestions, key=lambda suggestion: suggestion.score, reverse=True)
        )

    def as_vector(self) -> np.ndarray:
        return self._array[[self._idx], :].toarray()[0]

    def __len__(self) -> int:
        _, cols = self._array[[self._idx], :].nonzero()
        return len(cols)


class SuggestionBatch:
    """Subject suggestions for a batch of documents."""

    def __init__(self, array: csr_array) -> None:
        """Create a new SuggestionBatch from a csr_array"""
        assert isinstance(array, csr_array)
        self.array = array

    @classmethod
    def from_sequence(
        cls,
        suggestion_results: Sequence[Iterable[SubjectSuggestion]],
        subject_index: SubjectIndex,
        limit: int | None = None,
    ) -> SuggestionBatch:
        """Create a new SuggestionBatch from a sequence where each item is
        a sequence of SubjectSuggestion objects."""

        deprecated = set(subject_index.deprecated_ids())
        data, rows, cols = [], [], []
        for idx, result in enumerate(suggestion_results):
            for suggestion in itertools.islice(result, limit):
                if suggestion.subject_id in deprecated or suggestion.score <= 0.0:
                    continue
                data.append(min(suggestion.score, 1.0))
                rows.append(idx)
                cols.append(suggestion.subject_id)
        return cls(
            csr_array(
                (data, (rows, cols)),
                shape=(len(suggestion_results), len(subject_index)),
                dtype=np.float32,
            )
        )

    @classmethod
    def from_averaged(
        cls, batches: list[SuggestionBatch], weights: list[float]
    ) -> SuggestionBatch:
        """Create a new SuggestionBatch where the subject scores are the
        weighted average of scores in several SuggestionBatches"""

        avg_array = sum(
            [batch.array * weight for batch, weight in zip(batches, weights)]
        ) / sum(weights)
        return SuggestionBatch(avg_array)

    def filter(
        self, limit: int | None = None, threshold: float = 0.0
    ) -> SuggestionBatch:
        """Return a subset of the hits, filtered by the given limit and
        score threshold, as another SuggestionBatch object."""

        return SuggestionBatch(filter_suggestion(self.array, limit, threshold))

    def __getitem__(self, idx: int) -> SuggestionResult:
        if idx < 0 or idx >= len(self):
            raise IndexError
        return SuggestionResult(self.array, idx)

    def __len__(self) -> int:
        return self.array.shape[0]


class SuggestionResults:
    """Subject suggestions for a potentially very large number of documents."""

    def __init__(self, batches: Iterable[SuggestionBatch]) -> None:
        """Initialize a new SuggestionResults from an iterable that provides
        SuggestionBatch objects."""

        self.batches = batches

    def filter(
        self, limit: int | None = None, threshold: float = 0.0
    ) -> SuggestionResults:
        """Return a view of these suggestions, filtered by the given limit
        and/or threshold, as another SuggestionResults object."""

        return SuggestionResults(
            (batch.filter(limit, threshold) for batch in self.batches)
        )

    def __iter__(self) -> itertools.chain:
        return iter(itertools.chain.from_iterable(self.batches))
