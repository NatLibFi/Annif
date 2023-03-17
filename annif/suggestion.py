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


class SuggestionResult(metaclass=abc.ABCMeta):
    """Abstract base class for a set of hits returned by an analysis
    operation."""

    @abc.abstractmethod
    def as_list(self):
        """Return the hits as an ordered sequence of SubjectSuggestion objects,
        highest scores first."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def as_vector(self, size, destination=None):
        """Return the hits as a one-dimensional score vector of given size.
        If destination array is given (not None) it will be used, otherwise a
        new array will be created."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def filter(self, subject_index, limit=None, threshold=0.0):
        """Return a subset of the hits, filtered by the given limit and
        score threshold, as another SuggestionResult object."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def __len__(self):
        """Return the number of hits with non-zero scores."""
        pass  # pragma: no cover


class VectorSuggestionResult(SuggestionResult):
    """SuggestionResult implementation based primarily on NumPy vectors."""

    def __init__(self, vector):
        vector_f32 = vector.astype(np.float32)
        # limit scores to the range 0.0 .. 1.0
        self._vector = np.minimum(np.maximum(vector_f32, 0.0), 1.0)
        self._subject_order = None
        self._lsr = None

    def _vector_to_list_suggestion(self):
        hits = []
        for subject_id in self.subject_order:
            score = self._vector[subject_id]
            if score <= 0.0:
                break  # we can skip the remaining ones
            hits.append(SubjectSuggestion(subject_id=subject_id, score=float(score)))
        return ListSuggestionResult(hits)

    @property
    def subject_order(self):
        if self._subject_order is None:
            self._subject_order = np.argsort(self._vector)[::-1]
        return self._subject_order

    def as_list(self):
        if self._lsr is None:
            self._lsr = self._vector_to_list_suggestion()
        return self._lsr.as_list()

    def as_vector(self, size, destination=None):
        if destination is not None:
            np.copyto(destination, self._vector)
            return destination
        return self._vector

    def filter(self, subject_index, limit=None, threshold=0.0):
        mask = self._vector > threshold
        deprecated_ids = subject_index.deprecated_ids()
        if limit is not None:
            limit_mask = np.zeros_like(self._vector, dtype=bool)
            deprecated_set = set(deprecated_ids)
            top_k_subjects = itertools.islice(
                (subj for subj in self.subject_order if subj not in deprecated_set),
                limit,
            )
            limit_mask[list(top_k_subjects)] = True
            mask = mask & limit_mask
        else:
            deprecated_mask = np.ones_like(self._vector, dtype=bool)
            deprecated_mask[deprecated_ids] = False
            mask = mask & deprecated_mask
        vsr = VectorSuggestionResult(self._vector * mask)
        return ListSuggestionResult(vsr.as_list())

    def __len__(self):
        return (self._vector > 0.0).sum()


class ListSuggestionResult(SuggestionResult):
    """SuggestionResult implementation based primarily on lists of hits."""

    def __init__(self, hits):
        self._list = [self._enforce_score_range(hit) for hit in hits if hit.score > 0.0]
        self._vector = None

    @staticmethod
    def _enforce_score_range(hit):
        if hit.score > 1.0:
            return hit._replace(score=1.0)
        return hit

    def _list_to_vector(self, size, destination):
        if destination is None:
            destination = np.zeros(size, dtype=np.float32)

        for hit in self._list:
            if hit.subject_id is not None:
                destination[hit.subject_id] = hit.score
        return destination

    def as_list(self):
        return self._list

    def as_vector(self, size, destination=None):
        if self._vector is None:
            self._vector = self._list_to_vector(size, destination)
        return self._vector

    def filter(self, subject_index, limit=None, threshold=0.0):
        hits = sorted(self._list, key=lambda hit: hit.score, reverse=True)
        filtered_hits = [
            hit
            for hit in hits
            if hit.score >= threshold and hit.score > 0.0 and hit.subject_id is not None
        ]
        if limit is not None:
            filtered_hits = filtered_hits[:limit]
        return ListSuggestionResult(filtered_hits)

    def __len__(self):
        return len(self._list)


class SparseSuggestionResult(SuggestionResult):
    """SuggestionResult implementation backed by a single row of a sparse array."""

    def __init__(self, array, idx):
        self._array = array
        self._idx = idx

    def as_list(self):
        _, cols = self._array[[self._idx], :].nonzero()
        suggestions = [
            SubjectSuggestion(subject_id=col, score=float(self._array[self._idx, col]))
            for col in cols
        ]
        return sorted(
            suggestions, key=lambda suggestion: suggestion.score, reverse=True
        )

    def as_vector(self, size, destination=None):
        if destination is not None:
            print("as_vector called with destination not None")
            return None
        return self._array[[self._idx], :].toarray()[0]

    def filter(self, subject_index, limit=None, threshold=0.0):
        lsr = ListSuggestionResult(self.as_list())
        return lsr.filter(subject_index, limit, threshold)

    def __len__(self):
        _, cols = self._array[[self._idx], :].nonzero()
        return len(cols)


class SuggestionBatch:
    """Subject suggestions for a batch of documents."""

    def __init__(self, array):
        """Create a new SuggestionBatch from a csr_array"""
        self.array = array

    @classmethod
    def from_sequence(cls, suggestion_results, vocab_size):
        """Create a new SuggestionBatch from a sequence of SuggestionResult objects."""

        # create a dok_array for fast construction
        ar = dok_array((len(suggestion_results), vocab_size), dtype=np.float32)
        for idx, result in enumerate(suggestion_results):
            for suggestion in result.as_list():
                ar[idx, suggestion.subject_id] = suggestion.score
        return cls(ar.tocsr())

    def filter(self, limit=None, threshold=0.0):
        """Return a subset of the hits, filtered by the given limit and
        score threshold, as another SuggestionBatch object."""

        from annif.util import filter_suggestion

        return SuggestionBatch(filter_suggestion(self.array, limit, threshold))

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        return SparseSuggestionResult(self.array, idx)

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
