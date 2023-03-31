"""Representing suggested subjects."""

import abc
import collections
import itertools

import numpy as np
from scipy.sparse import dok_array

SubjectSuggestion = collections.namedtuple("SubjectSuggestion", "subject_id score")
WeightedSuggestionsBatch = collections.namedtuple(
    "WeightedSuggestionsBatch", "hit_sets weight subjects"
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

    def as_vector(self, size):
        return self._array[[self._idx], :].toarray()[0]

    def __len__(self):
        _, cols = self._array[[self._idx], :].nonzero()
        return len(cols)


class SuggestionBatch:
    """Subject suggestions for a batch of documents."""

    def __init__(self, array):
        """Create a new SuggestionBatch from a csr_array"""
        self.array = array

    @staticmethod
    def _vector_to_suggestions(vector):
        hits = []
        for subject_id in np.argsort(vector)[::-1]:
            score = vector[subject_id]
            if score <= 0.0:
                break  # we can skip the remaining ones
            hits.append(SubjectSuggestion(subject_id=subject_id, score=float(score)))
        return hits

    @classmethod
    def from_sequence(cls, suggestion_results, subject_index, limit=None):
        """Create a new SuggestionBatch from a sequence of SuggestionResult objects."""

        deprecated = set(subject_index.deprecated_ids())

        # create a dok_array for fast construction
        ar = dok_array((len(suggestion_results), len(subject_index)), dtype=np.float32)
        for idx, result in enumerate(suggestion_results):
            if isinstance(result, np.ndarray):
                result = cls._vector_to_suggestions(result)
            for suggestion in itertools.islice(result, limit):
                if suggestion.subject_id in deprecated or suggestion.score < 0.0:
                    continue
                ar[idx, suggestion.subject_id] = min(suggestion.score, 1.0)
        return cls(ar.tocsr())

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
