"""Representing suggested subjects."""

import abc
import collections
import itertools

import numpy as np

SubjectSuggestion = collections.namedtuple("SubjectSuggestion", "subject_id score")
WeightedSuggestion = collections.namedtuple(
    "WeightedSuggestion", "hits weight subjects"
)


class SuggestionFilter:
    """A reusable filter for filtering SubjectSuggestion objects."""

    def __init__(self, subject_index, limit=None, threshold=0.0):
        self._subject_index = subject_index
        self._limit = limit
        self._threshold = threshold

    def __call__(self, orighits):
        return LazySuggestionResult(
            lambda: orighits.filter(self._subject_index, self._limit, self._threshold)
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


class LazySuggestionResult(SuggestionResult):
    """SuggestionResult implementation that wraps another SuggestionResult which
    is initialized lazily only when it is actually accessed. Method calls
    will be proxied to the wrapped SuggestionResult."""

    def __init__(self, construct):
        """Create the proxy object. The given construct function will be
        called to create the actual SuggestionResult when it is needed."""
        self._construct = construct
        self._object = None

    def _initialize(self):
        if self._object is None:
            self._object = self._construct()

    def as_list(self):
        self._initialize()
        return self._object.as_list()

    def as_vector(self, size, destination=None):
        self._initialize()
        return self._object.as_vector(size, destination)

    def filter(self, subject_index, limit=None, threshold=0.0):
        self._initialize()
        return self._object.filter(subject_index, limit, threshold)

    def __len__(self):
        self._initialize()
        return len(self._object)


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
