"""Index for fast matching of token sets."""

import collections


class TokenSet:
    """Represents a set of tokens (expressed as integer token IDs) that can
    be matched with another set of tokens. A TokenSet can optionally
    be associated with a subject from the vocabulary."""

    def __init__(self, tokens, subject_id=None, is_pref=False):
        self._tokens = set(tokens)
        self.subject_id = subject_id
        self.is_pref = is_pref

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens)

    def contains(self, other):
        """Returns True iff the tokens in the other TokenSet are all
        included within this TokenSet."""

        return other._tokens.issubset(self._tokens)

    def sample(self):
        """Return an arbitrary token from this TokenSet, or None if empty"""
        try:
            return next(iter(self._tokens))
        except StopIteration:
            return None


class TokenSetIndex:
    """A searchable index of TokenSets (representing vocabulary terms)"""

    def __init__(self):
        self._index = collections.defaultdict(set)

    def __len__(self):
        return len(self._index)

    def add(self, tset):
        """Add a TokenSet into this index"""
        token = tset.sample()
        if token is not None:
            self._index[token].add(tset)

    def _find_subj_tsets(self, tset):
        subj_tsets = {}

        for token in tset:
            for ts in self._index[token]:
                if not tset.contains(ts):
                    continue
                if ts.subject_id not in subj_tsets or \
                   not subj_tsets[ts.subject_id].is_pref:
                    subj_tsets[ts.subject_id] = ts

        return subj_tsets

    def _find_subj_ambiguity(self, subj_tsets):
        subj_ambiguity = collections.Counter()

        for ts in subj_tsets.values():
            for other in subj_tsets.values():
                if ts == other:
                    continue
                if other.contains(ts):
                    subj_ambiguity.update([ts.subject_id])

        return subj_ambiguity

    def search(self, tset):
        """Return the TokenSets that are contained in the given TokenSet.
        The matches are returned as a list of (TokenSet, ambiguity) pairs
        where ambiguity is an integer indicating the number of other TokenSets
        that also match the same tokens."""

        subj_tsets = self._find_subj_tsets(tset)
        subj_ambiguity = self._find_subj_ambiguity(subj_tsets)

        return [(ts, subj_ambiguity[ts.subject_id])
                for uri, ts in subj_tsets.items()]
