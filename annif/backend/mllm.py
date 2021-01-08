"""Maui-like Lexical Matching backend"""

import collections
from rdflib import URIRef
from rdflib.namespace import SKOS
from sklearn.feature_extraction.text import CountVectorizer
from . import backend

Term = collections.namedtuple('Term', 'subject_id label is_pref')


class TokenSet:
    """Represents a set of tokens (expressed as integer token IDs) that can
    be matched with another set of tokens. A TokenSet can optionally
    be associated with a subject from the vocabulary."""

    def __init__(self, tokens, subject_id=None, is_pref=None):
        self._tokens = set(tokens)
        self.subject_id = subject_id
        self.is_pref = is_pref

    def contains(self, other):
        """Returns True iff the tokens in the other TokenSet are all
        included within this TokenSet."""

        return other._tokens.issubset(self.tokens)

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

    def add(self, tset):
        """Add a TokenSet into this index"""
        token = tset.sample()
        if token is not None:
            self._index[token].add(tset)

    def search(self, tset):
        """Return the TokenSets that are contained in the given TokenSet.
        The matches are returned as a list of (TokenSet, ambiguity) pairs
        where ambiguity is an integer indicating the number of other TokenSets
        that also match the same tokens."""

        subj_tsets = {}
        subj_ambiguity = collections.Counter()

        for token in tset.tokens:
            for ts in self._index[token]:
                if not tset.contains(ts):
                    continue
                if ts.subject_id not in subj_tsets or \
                   not subj_tsets[ts.subject_id].is_pref:
                    subj_tsets[ts.subject_id] = ts

        for ts in subj_tsets.values():
            for other in subj_tsets.values():
                if ts == other:
                    continue
                if other.contains(ts):
                    subj_ambiguity.update([ts.subject_id])

        return [(ts, subj_ambiguity[ts.subject_id])
                for uri, ts in subj_tsets.items()]


class MLLMBackend(backend.AnnifBackend):
    """Maui-like Lexical Matching backend for Annif"""
    name = "mllm"
    needs_subject_index = True

    def initialize(self):
        pass

    def _train(self, corpus, params):
        graph = self.project.vocab.as_graph()
        self.info('starting train')
        terms = []
        for subj_id, (uri, pref, _) in enumerate(self.project.vocab.subjects):
            if pref is None:
                continue  # deprecated subject
            terms.append(Term(subject_id=subj_id, label=pref, is_pref=True))
            alts = graph.preferredLabel(URIRef(uri),
                                        lang=params['language'],
                                        labelProperties=[SKOS.altLabel])
            for label, _ in alts:
                terms.append(Term(subject_id=subj_id,
                                  label=str(label),
                                  is_pref=False))

        vectorizer = CountVectorizer(
            binary=True,
            tokenizer=self.project.analyzer.tokenize_words
        )
        label_corpus = vectorizer.fit_transform((t.label for t in terms))
        self.info(label_corpus.shape)

        index = TokenSetIndex()
        for term, label_matrix in zip(terms, label_corpus):
            tokens = label_matrix.nonzero()[1]
            tset = TokenSet(tokens, term.subject_id, term.is_pref)
            index.add(tset)
        self.info(index._index)

    def _suggest(self, text, params):
        pass
