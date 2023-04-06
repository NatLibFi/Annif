"""Representing suggested subjects."""

import collections
import itertools

import numpy as np
from scipy.sparse import csr_array, dok_array

SubjectSuggestion = collections.namedtuple("SubjectSuggestion", "subject_id score")


def vector_to_suggestions(vector, limit):
    limit = min(len(vector), limit)
    topk_idx = np.argpartition(vector, -limit)[-limit:]
    return (
        SubjectSuggestion(subject_id=idx, score=float(vector[idx])) for idx in topk_idx
    )


def filter_suggestion(preds, limit=None, threshold=0.0):
    """filter a 2D sparse suggestion array (csr_array), retaining only the
    top K suggestions with a score above or equal to the threshold for each
    individual prediction; the rest will be left as zeros"""

    filtered = dok_array(preds.shape, dtype=np.float32)
    for row in range(preds.shape[0]):
        arow = preds.getrow(row)
        top_k = arow.data.argsort()[::-1]
        if limit is not None:
            top_k = top_k[:limit]
        for idx in top_k:
            val = arow.data[idx]
            if val < threshold:
                break
            filtered[row, arow.indices[idx]] = val
    return filtered.tocsr()


class SuggestionResult:
    """Suggestions for a single document, backed by a row of a sparse array."""

    def __init__(self, array, idx):
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

    def as_vector(self):
        return self._array[[self._idx], :].toarray()[0]

    def __len__(self):
        _, cols = self._array[[self._idx], :].nonzero()
        return len(cols)


class SuggestionBatch:
    """Subject suggestions for a batch of documents."""

    def __init__(self, array):
        """Create a new SuggestionBatch from a csr_array"""
        assert isinstance(array, csr_array)
        self.array = array

    @classmethod
    def from_sequence(cls, suggestion_results, subject_index, limit=None):
        """Create a new SuggestionBatch from a sequence where each item is
        a sequence of SubjectSuggestion objects."""

        deprecated = set(subject_index.deprecated_ids())

        ar = dok_array((len(suggestion_results), len(subject_index)), dtype=np.float32)
        for idx, result in enumerate(suggestion_results):
            for suggestion in itertools.islice(result, limit):
                if suggestion.subject_id in deprecated or suggestion.score <= 0.0:
                    continue
                ar[idx, suggestion.subject_id] = min(suggestion.score, 1.0)
        return cls(ar.tocsr())

    @classmethod
    def from_averaged(cls, batches, weights):
        """Create a new SuggestionBatch where the subject scores are the
        weighted average of scores in several SuggestionBatches"""

        avg_array = sum(
            [batch.array * weight for batch, weight in zip(batches, weights)]
        ) / sum(weights)
        return SuggestionBatch(avg_array)

    def filter(self, limit=None, threshold=0.0):
        """Return a subset of the hits, filtered by the given limit and
        score threshold, as another SuggestionBatch object."""

        return SuggestionBatch(filter_suggestion(self.array, limit, threshold))

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        return SuggestionResult(self.array, idx)

    def __len__(self):
        return self.array.shape[0]


class SuggestionResults:
    """Subject suggestions for a potentially very large number of documents."""

    def __init__(self, batches):
        """Initialize a new SuggestionResults from an iterable that provides
        SuggestionBatch objects."""

        self.batches = batches

    def filter(self, limit=None, threshold=0.0):
        """Return a view of these suggestions, filtered by the given limit
        and/or threshold, as another SuggestionResults object."""

        return SuggestionResults(
            (batch.filter(limit, threshold) for batch in self.batches)
        )

    def __iter__(self):
        return iter(itertools.chain.from_iterable(self.batches))
